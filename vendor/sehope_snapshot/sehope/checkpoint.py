import os
from pathlib import Path
import torch

def cpu_tree(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x

def save_checkpoint(path,model,optimizer,config,weights_sha,cache_meta,next_epoch,updates,best_val=-float('inf')):
    rng={'cpu':torch.get_rng_state()}
    if torch.cuda.is_available(): rng['cuda']=torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available(): rng['mps']=torch.mps.get_rng_state()
    payload=cpu_tree({'format_version':1,'policy':model.policy.state_dict(),'optimizer':optimizer.state_dict(),
        'config':config,'weights_sha256':weights_sha,'cache_meta':cache_meta,'next_epoch':next_epoch,
        'updates':updates,'best_val':best_val,'rng':rng,'torch_version':str(torch.__version__),
        'stream_state':None,'checkpoint_boundary':'epoch; all episodes complete'})
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); os.replace(tmp,path)

def restore(path,model,weights_sha,optimizer=None,cache_meta=None):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p['format_version']!=1 or p['weights_sha256']!=weights_sha: raise ValueError('Checkpoint/backbone mismatch')
    if cache_meta is not None and p['cache_meta']!=cache_meta: raise ValueError('Resume cache changed; use --warm-start for new source data')
    model.policy.load_state_dict(p['policy'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(p['optimizer'])
        device=next(model.policy.parameters()).device
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor) and k!='step': state[k]=v.to(device)
        torch.set_rng_state(p['rng']['cpu'])
        if 'cuda' in p['rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(p['rng']['cuda'])
        if 'mps' in p['rng'] and torch.backends.mps.is_available(): torch.mps.set_rng_state(p['rng']['mps'])
    return p
