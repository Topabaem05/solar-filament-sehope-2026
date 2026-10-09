import copy
import json
from pathlib import Path
import torch
from .memory import MemoryPolicy
from .recurrence import srnl_reference

def device_check(device,output):
    """Small FP32 forward, final-state, gradient and optimizer test vs CPU."""
    torch.manual_seed(29)
    x=torch.randn(1,2,2,5,4)*.2; q=torch.randn_like(x)
    states=[torch.randn(1,2,2,4,4)*.03 for _ in range(3)]+[torch.randn(1,2,2,4)*.03 for _ in range(2)]
    def run(dev):
        a=x.to(dev).requires_grad_(); b=q.to(dev).requires_grad_()
        ss=[s.to(dev).requires_grad_() for s in states]
        y,end=srnl_reference(a,b,ss,3)
        loss=y.square().sum()+sum(z.square().sum() for z in end)
        grads=torch.autograd.grad(loss,[a,b,*ss])
        return [y.detach().cpu(),*[e.detach().cpu() for e in end],*[g.detach().cpu() for g in grads]]
    reference,actual=run(torch.device('cpu')),run(device)
    for a,b in zip(reference,actual):torch.testing.assert_close(a,b,rtol=2e-3,atol=2e-5)
    anchors=torch.randn(1,4,2,4,4)*.03
    policy=MemoryPolicy(anchors,channels=8,descriptor_dim=8,hidden=8).to(device)
    opt=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=.001,foreach=False)
    s=policy.initial_state();d=policy.descriptor(torch.randn(1,8,3,3,device=device))
    for _ in range(4):
        _,j,_,sim=policy.read(d,s)
        s,_=policy.update(d,s,j,sim,torch.randn(1,4,device=device),torch.randn_like(policy.radius.expand(policy.shape))*.02)
    (s.values.square().sum()+s.hidden.square().mean()).backward();opt.step()
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    result={'status':'PASS','device':str(device),'torch':str(torch.__version__),
            'max_abs_diff':max(float((a-b).abs().max()) for a,b in zip(reference,actual)),
            'tests':['SRNL outputs','all five final states','input/query/initial-state gradients','GRU+Writer optimizer step'],
            'limitation':'Small operator test; not full model accuracy or throughput.'}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));return result
