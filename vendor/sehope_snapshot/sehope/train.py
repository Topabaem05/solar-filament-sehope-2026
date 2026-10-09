"""Offline truncated-BPTT meta-training. Labels reach losses, never state update."""
import json
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .checkpoint import save_checkpoint,restore

def snapshot(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def train(model,cache,config,weights_sha,output,device,resume=None,warm_start=None,resource_guard=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    traincfg=config['training']; window=int(traincfg['tbptt'])
    if window<2: raise ValueError('TBPTT >=2 is required to learn future effects')
    params={n:p for n,p in model.policy.named_parameters() if p.requires_grad}
    if not params: raise ValueError('No trainable policy parameters')
    optimizer=torch.optim.AdamW(params.values(),lr=traincfg['lr'],weight_decay=traincfg.get('weight_decay',1e-4),foreach=False)
    start_epoch,updates=0,0
    if resume and warm_start: raise ValueError('Choose resume or warm-start')
    if resume:
        p=restore(resume,model,weights_sha,optimizer,cache.meta)
        if p['config']['policy']!=config['policy'] or p['config']['adapted_blocks']!=config['adapted_blocks']: raise ValueError('Resume architecture changed')
        start_epoch,updates=p['next_epoch'],p['updates']
        for group in optimizer.param_groups: group['lr']=traincfg['lr']
    if warm_start: restore(warm_start,model,weights_sha)
    if start_epoch>=traincfg['epochs']:raise ValueError('epochs is the total target; set it above the resumed next_epoch')
    frozen=snapshot(model.backbone)
    before={n:p.detach().cpu().clone() for n,p in params.items()}
    gradient_max={n:0. for n in params}
    validation_source=None;best_val=-float('inf')
    if resume:
        best_val=p.get('best_val',-float('inf'))
    episodes=defaultdict(list);past=[];clean=[]
    for i,r in enumerate(cache.rows):
        if r['split']!='train':continue
        if r['role']=='stream': episodes[r['episode']].append(i)
        elif r['role']=='past':past.append(i)
        else:clean.append(i)
    if not episodes or not past or not clean: raise ValueError('Train stream, disjoint past probes and clean probes are required')
    model.train()
    began=time.monotonic();epoch_seconds=[];stale=0;meaningful_best=best_val;stop_reason='TARGET_EPOCHS'
    training_seconds=0.;validation_seconds=0.;update_times=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch,traincfg['epochs']):
        if epoch_seconds and time.monotonic()-began+1.2*max(epoch_seconds)>traincfg.get('max_seconds',21600):
            stop_reason='EPOCH_BUDGET';break
        epoch_began=time.monotonic();totals=[]
        for ep,indices in sorted(episodes.items()):
            state=model.policy.initial_state(); seen=set()
            for start in range(0,len(indices),window):
                update_began=time.monotonic()
                if time.monotonic()-began>traincfg.get('max_seconds',21600):raise TimeoutError('Training invocation budget exceeded; only previous epoch checkpoint is resumable')
                if resource_guard: resource_guard.sample('before_window')
                ids=indices[start:start+window]
                optimizer.zero_grad(set_to_none=True); loss=next(iter(params.values())).sum()*0
                for i in ids:
                    r=cache.rows[i];seen.add(r['domain'])
                    # Prediction at t uses P[t-1], then write P[t].
                    logits,state,cost=model.step(cache.feature(i,device),state)
                    target=torch.tensor([r['label']],device=device)
                    loss=loss+F.cross_entropy(logits,target)/len(ids)+traincfg.get('lambda_budget',.001)*cost/len(ids)
                    if resource_guard: resource_guard.sample('stream_forward')
                # Probes read the current state without modifying it.
                eligible=[i for i in past if cache.rows[i]['domain'] in seen]
                if not eligible: raise ValueError('No past probe for seen training domains')
                for pool,scale in [(eligible,traincfg.get('lambda_past',.5)),(clean,traincfg.get('lambda_clean',.5))]:
                    j=pool[int(torch.randint(len(pool),(1,)).item())]
                    logits,_,_=model.step(cache.feature(j,device),state,write=False)
                    loss=loss+scale*F.cross_entropy(logits,torch.tensor([cache.rows[j]['label']],device=device))
                    if resource_guard: resource_guard.sample('probe_forward')
                loss=loss+traincfg.get('lambda_state',.0001)*(state.values.square()/(model.policy.radius.square()+1e-20)).mean()
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite objective')
                loss.backward()
                if resource_guard: resource_guard.sample('backward')
                for n,p in params.items():
                    if p.grad is not None:
                        if not bool(torch.isfinite(p.grad).all()): raise FloatingPointError(f'Nonfinite gradient {n}')
                        gradient_max[n]=max(gradient_max[n],float(p.grad.norm()))
                norm=torch.nn.utils.clip_grad_norm_(list(params.values()),traincfg.get('grad_clip',1.),error_if_nonfinite=True)
                optimizer.step(); state=state.detach();updates+=1
                totals.append(float(loss.detach()))
                if any(not bool(torch.isfinite(p).all()) for p in params.values()):raise FloatingPointError('Nonfinite weight')
                peak=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None
                if peak and peak>traincfg.get('max_cuda_gib',14):raise RuntimeError('Memory budget exceeded; lower TBPTT and retry from checkpoint')
                reserved=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None
                seconds=time.monotonic()-update_began;training_seconds+=seconds;update_times.append(seconds)
                record={'train_seconds':seconds,'epoch':epoch,'episode':ep,'update':updates,'loss':totals[-1],'grad_norm':float(norm),'peak_cuda_gib':peak,'peak_cuda_reserved_gib':reserved,'rejected':state.rejected}
                if resource_guard: record['memory'] = resource_guard.sample('optimizer')
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if any(not torch.equal(frozen[k],v.detach().cpu()) for k,v in model.backbone.state_dict().items()): raise AssertionError('Frozen parameter or buffer changed')
        if (epoch+1)%traincfg.get('validate_every',1)==0 and any(r['split']=='val' and r['role']=='stream' for r in cache.rows):
            validation_began=time.monotonic()
            if resource_guard: resource_guard.sample('before_validation')
            from .evaluate import evaluate
            eval_options=config.get('evaluation',{})
            if validation_source is None: validation_source=evaluate(model,cache,device,'val',source=True,resource_guard=resource_guard,**eval_options)
            val=evaluate(model,cache,device,'val',resource_guard=resource_guard,**eval_options)
            if resource_guard: resource_guard.sample('validation')
            clean_ok=val['clean_top1'] is not None and val['clean_source_top1']-val['clean_top1']<=.01
            val['source_top1']=validation_source['top1']
            val['constrained_selection_eligible']=clean_ok
            (out/f'validation_epoch_{epoch+1}.json').write_text(json.dumps(val,indent=2))
            if clean_ok and val['top1']>best_val:
                best_val=val['top1']
                save_checkpoint(out/'best_val.pt',model,optimizer,config,weights_sha,cache.meta,epoch+1,updates,best_val)
            if clean_ok and val['top1']>=meaningful_best+traincfg.get('min_delta',.002):
                meaningful_best=val['top1'];stale=0
            else:stale+=1
            validation_seconds+=time.monotonic()-validation_began
            model.train()
        save_checkpoint(out/'last.pt',model,optimizer,config,weights_sha,cache.meta,epoch+1,updates,best_val)
        epoch_seconds.append(time.monotonic()-epoch_began)
        timing={'training_seconds':training_seconds,'validation_seconds':validation_seconds,'epoch_seconds':epoch_seconds,'first_100_update_mean_seconds':sum(update_times[:100])/len(update_times[:100]),'observed_updates':len(update_times),'completed_epoch':epoch+1}
        (out/'timing.json').write_text(json.dumps(timing,indent=2))
        print(json.dumps({'epoch':epoch,'mean_loss':sum(totals)/len(totals),'updates':updates}),flush=True)
        if epoch+1>=traincfg.get('min_epochs',3) and stale>=traincfg.get('patience',3):
            stop_reason='PLATEAU';break
    audit={n:{'max_grad_norm':gradient_max[n],'weight_delta_norm':float((p.detach().cpu()-before[n]).norm())} for n,p in params.items()}
    active_writer=any(v['max_grad_norm']>0 and v['weight_delta_norm']>0 for n,v in audit.items() if n.startswith('writer'))
    active_controller=any(v['max_grad_norm']>0 and v['weight_delta_norm']>0 for n,v in audit.items() if not n.startswith('writer'))
    passed=active_writer and (active_controller or model.policy.mode=='fixed')
    report={'stop_reason':stop_reason,'best_checkpoint_eligible':(out/'best_val.pt').is_file(),'status':'PASS' if passed else 'FAIL','frozen_unchanged':True,'weights':audit,
            'note':'Gradient/weight updates do not establish generalization or retention.'}
    (out/'weight_audit.json').write_text(json.dumps(report,indent=2))
    if not (out/'best_val.pt').is_file():raise RuntimeError('No clean-constrained validation checkpoint; candidate rejected')
    if not passed:raise AssertionError('Required learning path inactive; inspect weight_audit.json')
    return report
