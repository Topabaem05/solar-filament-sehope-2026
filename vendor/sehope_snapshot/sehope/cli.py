import argparse
import json
from pathlib import Path
import torch
from .backbone import SplitBackbone,load_tiny,sha256
from .model import SEHOPE
from .data import FeatureCache,build_cache
from .checkpoint import restore

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['cache','train','eval','inspect','device-check','compare'])
    p.add_argument('--config',default='configs/v100.json')
    p.add_argument('--weights',default='weights/visionhope_tiny.pth')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    p.add_argument('--cache');p.add_argument('--manifest');p.add_argument('--output',default='outputs/run')
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--warm-start')
    p.add_argument('--split',default='test',choices=['val','test'])
    p.add_argument('--source',action='store_true');p.add_argument('--reset-boundaries',action='store_true')
    p.add_argument('--fp16-cache',action='store_true')
    p.add_argument('--upstream-prefix',action='store_true')
    p.add_argument('--reports',nargs=4,metavar=('META','SOURCE','FIXED','RESET'))
    p.add_argument('--timing-report')
    a=p.parse_args()
    torch.set_num_threads(2)
    config=json.loads(Path(a.config).read_text());torch.manual_seed(config.get('seed',42))
    device=torch.device(a.device)
    if a.command=='compare':
        from .evaluate import compare
        print(json.dumps(compare(*a.reports,a.output,a.timing_report),indent=2));return
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no silent CPU fallback')
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable; no silent CPU fallback')
    resource_guard = None
    if device.type=='cuda':
        from .resources import configure_cuda
        resource_guard=configure_cuda(config)
    if device.type=='mps':
        from .resources import configure_mps
        resource_guard = configure_mps(config)
    if device.type=='cpu' and a.command=='cache':
        from .resources import configure_host
        resource_guard=configure_host(config)
    if a.command=='device-check':
        from .validation import device_check
        print(json.dumps(device_check(device,a.output),indent=2));return
    digest=sha256(a.weights)
    expected=json.loads(Path('weights_manifest.json').read_text())['sha256']
    if digest!=expected:raise ValueError('Official checkpoint SHA256 mismatch')
    backbone=SplitBackbone(load_tiny(a.weights),config['adapted_blocks']).to(device)
    if a.command=='inspect':
        m=SEHOPE(backbone,**config['policy']).to(device)
        print(json.dumps({'sha256':digest,'backbone_parameters':sum(p.numel() for p in backbone.parameters()),
            'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
            'state_bytes':m.policy.initial_state().nbytes(),'state_shape':list(backbone.anchors.shape)},indent=2));return
    if a.command=='cache':
        if a.upstream_prefix:
            if device.type!='cuda':raise ValueError('Official prefix requires CUDA')
            from .upstream import official_prefix
            backbone=official_prefix(a.weights,config['adapted_blocks']).to(device)
        batch_size=config.get('runtime',{}).get('cpu_prefix_batch_size',1) if device.type=='cpu' else 1
        print(json.dumps(build_cache(a.manifest,a.output,backbone,digest,device,'official' if a.upstream_prefix else 'reference',a.fp16_cache,resource_guard=resource_guard,batch_size=batch_size),indent=2));return
    if not a.cache:raise ValueError('--cache required')
    cache=FeatureCache(a.cache,digest,config['adapted_blocks'])
    model=SEHOPE(backbone,**config['policy']).to(device)
    if a.command=='train':
        from .train import train
        print(json.dumps(train(model,cache,config,digest,a.output,device,a.resume,a.warm_start,resource_guard=resource_guard),indent=2))
    elif a.command=='eval':
        if not a.checkpoint and not a.source:raise ValueError('Specify trained checkpoint or --source')
        if a.checkpoint:
            ck=restore(a.checkpoint,model,digest)
            if ck['config']['policy']!=config['policy']:raise ValueError('Use checkpoint policy configuration')
        from .evaluate import evaluate
        report=evaluate(model,cache,device,a.split,a.source,a.reset_boundaries,resource_guard=resource_guard,**config.get('evaluation',{}))
        report['policy_config']=config['policy'];report['checkpoint']=a.checkpoint
        report['checkpoint_sha256']=sha256(a.checkpoint) if a.checkpoint else None
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        Path(a.output).write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))

if __name__=='__main__':main()
