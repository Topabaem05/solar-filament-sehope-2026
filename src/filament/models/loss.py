"""Single-class instance matching and losses for the memory-off baseline."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class InstanceLoss(nn.Module):
    def __init__(self, *, balanced_mask_bce=False):
        super().__init__()
        self.balanced_mask_bce=balanced_mask_bce

    def forward(self, output, targets):
        if len(targets)!=output['mask_logits'].shape[0]:
            raise ValueError('One target mask set is required per image.')
        losses=[]
        matches=[]
        for b,target in enumerate(targets):
            target=target.to(output['mask_logits'].device,dtype=torch.float32)
            if target.ndim!=3 or not torch.isfinite(target).all() or not ((target==0)|(target==1)).all():
                raise ValueError('Targets must be [instances,H,W] binary masks.')
            if len(target) and not (target.flatten(1).sum(1)>0).all():
                raise ValueError('Remove and report empty targets after spatial transforms.')
            logits=F.interpolate(output['mask_logits'][b,None],size=target.shape[-2:],mode='bilinear',align_corners=False)[0]
            obj=output['object_logits'][b]
            if len(target)>len(obj):
                raise ValueError('Target count exceeds query capacity.')
            labels=torch.zeros_like(obj)
            if len(target):
                flat=logits.flatten(1)
                gt=target.flatten(1)
                prob=flat.sigmoid()
                # Exact dense pair costs; point sampling can replace this after profiling.
                bce=F.softplus(flat).mean(1)[:,None]-flat@gt.T/gt.shape[1]
                if self.balanced_mask_bce:
                    pos=gt.sum(1);neg=gt.shape[1]-pos
                    pc=F.softplus(-flat)@gt.T/pos.clamp_min(1)[None]
                    nc=F.softplus(flat)@(1-gt).T/neg.clamp_min(1)[None]
                    bce=(pc*(pos>0)[None]+nc*(neg>0)[None])/((pos>0).float()+(neg>0).float())[None]
                dice=1-(2*prob@gt.T+1)/(prob.sum(1)[:,None]+gt.sum(1)[None]+1)
                cost=-obj.sigmoid()[:,None]+5*bce+5*dice
                qi,ti=linear_sum_assignment(cost.detach().cpu().numpy())
                labels[qi]=1
                mask_loss=F.binary_cross_entropy_with_logits(logits[qi],target[ti])
                if self.balanced_mask_bce:
                    # The selected pair costs remain differentiable after matching.
                    mask_loss=bce[qi,ti].mean()
                p=logits[qi].sigmoid().flatten(1); t=target[ti].flatten(1)
                dice_loss=(1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()
            else:
                qi,ti=[],[]
                mask_loss=logits.sum()*0
                dice_loss=logits.sum()*0
            object_weights=torch.where(labels.bool(),1.,.1)
            object_loss=(F.binary_cross_entropy_with_logits(obj,labels,reduction='none')
                         *object_weights).sum()/object_weights.sum()
            losses.append(object_loss+5*mask_loss+5*dice_loss)
            matches.append(len(qi))
        return torch.stack(losses).mean(), {'matched_instances':sum(matches)}
