import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
    elif device.type=='mps':torch.mps.synchronize()

@torch.no_grad()
def _evaluate_episode(model,cache,device,split='test',source=False,reset_boundaries=False,probe_limit=None,resource_guard=None,source_probe_logits=None):
    """Domain annotations are exclusively evaluator metadata.

    reset_boundaries is an explicitly oracle-assisted diagnostic, not a deployable
    baseline. Cached timing excludes prefix/preprocessing and is labeled as such.
    """
    if split not in {'val','test'}:raise ValueError('Use held-out val/test for goal evaluation')
    model.eval(); state=model.policy.initial_state()
    if resource_guard: resource_guard.sample('before_evaluation')
    stream=[i for i,r in enumerate(cache.rows) if r['split']==split and r['role']=='stream']
    probes=defaultdict(list);clean=[]
    for i,r in enumerate(cache.rows):
        if r['split']!=split:continue
        if r['role']=='past':probes[r['domain']].append(i)
        if r['role']=='clean':clean.append(i)
    if not stream:raise ValueError(f'No {split} stream')
    results=[];visits=defaultdict(int);segment_pos=0;last=None;seen=set();history=defaultdict(list)
    def probe_logits(i,is_source):
        key=cache.rows[i]['id']
        if is_source and source_probe_logits is not None and key in source_probe_logits:
            return source_probe_logits[key].to(device)
        y,_,_=model.step(cache.feature(i,device),state,write=False,source=is_source)
        if is_source and source_probe_logits is not None:source_probe_logits[key]=y.cpu()
        return y
    def probe():
        for domain in seen:
            ids=probes[domain][:probe_limit]
            if ids:
                correct=[]
                for i in ids:
                    y=probe_logits(i,source)
                    correct.append(int(y.argmax(-1).item()==cache.rows[i]['label']))
                    if resource_guard: resource_guard.sample('evaluation_past_probe')
                history[domain].append(float(np.mean(correct)))
    for i in stream:
        row=cache.rows[i];domain=row['domain']
        if domain!=last:
            if last is not None:probe()
            visits[domain]+=1;segment_pos=0;last=domain;seen.add(domain)
            if reset_boundaries:state=model.policy.initial_state()
        feature=cache.feature(i,device)
        synchronize(device);start=time.perf_counter()
        logits,context=model.predict(feature,state,source=source)
        prediction=int(logits.argmax(-1).item())  # Commit before persistent write.
        if not source:state,_=model.observe(state,logits,context)
        synchronize(device);elapsed=time.perf_counter()-start
        if resource_guard: resource_guard.sample('evaluation_stream')
        results.append({'id':row['id'],'parent_id':row['parent_id'],'episode':row['episode'],'domain':domain,'visit':visits[domain],
                        'visit_pos':segment_pos,'correct':int(prediction==row['label']),'prediction':prediction,
                        'suffix_step_ms':1000*elapsed,'slot':context[1],'read_strength':float(context[2])})
        segment_pos+=1
    probe()
    per_domain={d:float(np.mean([r['correct'] for r in results if r['domain']==d])) for d in seen}
    revisit=[r['correct'] for r in results if r['visit']>1 and r['visit_pos']<128]
    clean_current=[];clean_source=[]
    for i in clean[:probe_limit]:
        f=cache.feature(i,device)
        for target,is_source in [(clean_current,False),(clean_source,True)]:
            y=probe_logits(i,source or is_source)
            target.append(int(y.argmax(-1).item()==cache.rows[i]['label']))
            if resource_guard: resource_guard.sample('evaluation_clean_probe')
    report = {'split':split,'cache_meta':cache.meta,'device':str(device),'precision':'fp32','source':source,'oracle_boundary_reset':reset_boundaries,
            'probe_counts':{'clean':len(clean[:probe_limit]),'past':{d:len(probes[d][:probe_limit]) for d in sorted(seen)}},
            'n':len(results),'top1':float(np.mean([r['correct'] for r in results])),
            'worst_domain_top1':min(per_domain.values()),'domain_top1':per_domain,
            'revisit_top1':float(np.mean(revisit)) if revisit else None,
            'clean_top1':float(np.mean(clean_current)) if clean_current else None,
            'clean_source_top1':float(np.mean(clean_source)) if clean_source else None,
            'forgetting':{d:max(scores)-scores[-1] for d,scores in history.items()},
            'probe_accuracy_history':dict(history),'state_bytes':state.nbytes(),
            'suffix_p95_ms':float(np.percentile([r['suffix_step_ms'] for r in results],95)),
            'e2e_p95_ms':None,'timing_note':'Cached suffix+policy only; NOT end-to-end latency.',
            'rejected':state.rejected,'predictions':results}
    if resource_guard: report['memory'] = resource_guard.sample('evaluation_complete')
    return report

EVALUATION_PROTOCOL = 'independent-episodes-all-probes-v2'

@torch.no_grad()
def evaluate(model,cache,device,split='test',source=False,reset_boundaries=False,
             probe_limit=None,resource_guard=None,expected_probes=None):
    """Reset independent episodes; preserve state across domains within an episode.

    Shared held-out probes are evaluated after each independent episode. Aggregate
    accuracy/retention is episode-macro, with image-micro accuracy also reported.
    None means ALL probes. Explicit finite limits must be satisfiable, never silently
    truncated. expected_probes pins available denominators for a dataset protocol.
    """
    if probe_limit is not None and (type(probe_limit) is not int or probe_limit < 1):
        raise ValueError('probe_limit must be None (all) or a positive integer')
    episodes={}; pools=defaultdict(list);clean=[]
    for i,r in enumerate(cache.rows):
        if r['split']!=split:continue
        if r['role']=='stream':episodes.setdefault(r['episode'],[]).append(i)
        elif r['role']=='past':pools[r['domain']].append(i)
        elif r['role']=='clean':clean.append(i)
    if not episodes:raise ValueError(f'No {split} stream')
    counts={'clean':len(clean),'past':{d:len(v) for d,v in sorted(pools.items())}}
    if expected_probes is not None and counts!=expected_probes:
        raise ValueError(f'Probe cardinality mismatch: {counts} != {expected_probes}')
    if probe_limit is not None and any(n<probe_limit for n in [len(clean),*[len(v) for v in pools.values()]]):
        raise ValueError('Insufficient probes for explicit probe_limit')
    class EpisodeCache:
        def __init__(self, indices):
            self.indices=indices;self.rows=[cache.rows[i] for i in indices];self.meta=cache.meta
        def feature(self,i,device):return cache.feature(self.indices[i],device)
    probe_ids=clean+[i for v in pools.values() for i in v]
    reports=[];source_probe_logits={}
    for ep,ids in episodes.items():
        report=_evaluate_episode(model,EpisodeCache(ids+probe_ids),device,split,source,
                                 reset_boundaries,probe_limit,resource_guard,source_probe_logits)
        report['episode']=ep;reports.append(report)
    result=dict(reports[0]);predictions=[p for r in reports for p in r['predictions']]
    def mean(key):
        values=[r[key] for r in reports if r[key] is not None]
        return float(np.mean(values)) if values else None
    for key in ['top1','revisit_top1','clean_top1','clean_source_top1']:
        result[key]=mean(key)
    domains=sorted({d for r in reports for d in r['domain_top1']})
    result['domain_top1']={d:float(np.mean([r['domain_top1'][d] for r in reports if d in r['domain_top1']])) for d in domains}
    result['worst_domain_top1']=min(result['domain_top1'].values())
    result['forgetting']={d:float(np.mean([r['forgetting'][d] for r in reports if d in r['forgetting']])) for d in domains if any(d in r['forgetting'] for r in reports)}
    result['probe_accuracy_history']={str(r['episode']):r['probe_accuracy_history'] for r in reports}
    result.update(protocol=EVALUATION_PROTOCOL,aggregation='episode_macro',episode_count=len(reports),
                  n=len(predictions),predictions=predictions,available_probe_counts=counts,
                  top1_micro=float(np.mean([p['correct'] for p in predictions])),
                  rejected=sum(r['rejected'] for r in reports),state_bytes=max(r['state_bytes'] for r in reports),
                  suffix_p95_ms=float(np.percentile([p['suffix_step_ms'] for p in predictions],95)),
                  episodes=[{k:v for k,v in r.items() if k not in {'predictions','cache_meta','memory'}} for r in reports])
    result.pop('episode',None)
    if resource_guard:result['memory']=resource_guard.sample('all_episodes_complete')
    return result

def compare(meta,source,fixed,reset,output,timing_report=None):
    reports=[json.loads(Path(p).read_text()) for p in [meta,source,fixed,reset]]
    m,s,f,r=reports
    ids=[x['id'] for x in m['predictions']]
    for other in reports[1:]:
        if other.get('protocol')!=m.get('protocol'):raise ValueError('Evaluation protocol mismatch')
        if other['cache_meta']!=m['cache_meta'] or [x['id'] for x in other['predictions']]!=ids:raise ValueError('Unpaired data/order/cache')
    if not s['source'] or not r['oracle_boundary_reset']:raise ValueError('Expected source and reset diagnostic reports')
    if f.get('policy_config',{}).get('mode')!='fixed':raise ValueError('Use an independently trained, validation-tuned fixed policy')
    def gate(value,ok):return {'status':'NOT_MEASURED' if value is None else ('PASS' if ok(value) else 'FAIL'),'value':value}
    gain=m['top1']-s['top1'];meta_gain=m['top1']-f['top1']
    clean_drop=None if m['clean_top1'] is None else m['clean_source_top1']-m['clean_top1']
    revisit=None if m['revisit_top1'] is None or r['revisit_top1'] is None else m['revisit_top1']-r['revisit_top1']
    latency=None if m['e2e_p95_ms'] is None or s['e2e_p95_ms'] is None else m['e2e_p95_ms']/s['e2e_p95_ms']
    if timing_report:
        timing=json.loads(Path(timing_report).read_text())
        if timing['weights_sha256']!=m['cache_meta']['weights_sha256'] or timing['checkpoint_sha256']!=m['checkpoint_sha256']:raise ValueError('Timing weights mismatch')
        if timing['device']!=m['device'] or timing['precision']!=m['precision']:raise ValueError('Timing device/precision mismatch')
        if not set(timing['image_ids'])<=set(ids):raise ValueError('Timing images differ from evaluated stream')
        latency=timing['p95_ratio']
    gates={'online_gain':gate(gain,lambda x:x>=.02),'meta_over_fixed':gate(meta_gain,lambda x:x>=.01),
           'clean_drop':gate(clean_drop,lambda x:x<=.01),'revisit_gain':gate(revisit,lambda x:x>=.02),
           'state_bytes':gate(m['state_bytes'],lambda x:x<=8*2**20),'end_to_end_p95_ratio':gate(latency,lambda x:x<=1.3)}
    result={'status':'PER_RUN_PASS' if all(v['status']=='PASS' for v in gates.values()) else 'NOT_PASSED',
            'study_status':'MULTI_SEED_CONFIRMATION_REQUIRED',
            'gates':gates,'note':'One run only. Require paired 5 training seeds x 3 stream orders; no IID-image significance claim.'}
    Path(output).write_text(json.dumps(result,indent=2));return result
