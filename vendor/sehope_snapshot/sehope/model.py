import torch
from torch import nn
from .memory import MemoryPolicy

class SEHOPE(nn.Module):
    def __init__(self, backbone, **policy_config):
        super().__init__()
        self.backbone = backbone
        self.policy = MemoryPolicy(backbone.anchors,channels=backbone.model.embed_dims[-1],**policy_config)
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def predict(self, feature, state, *, source=False):
        desc = self.policy.descriptor(feature)
        offset,j,rho,sim = self.policy.read(desc,state)
        if source: offset = torch.zeros_like(offset)
        logits,end,start = self.backbone.suffix(feature,offset)
        if not bool(torch.isfinite(logits).all()):
            logits,_,_ = self.backbone.suffix(feature,torch.zeros_like(offset))
            if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Source fallback also nonfinite')
            return logits, (desc,-1,rho,sim,None)
        return logits,(desc,j,rho,sim,end-start)
    def observe(self, state, logits, context):
        desc,j,_,sim,delta = context
        if delta is None:
            from dataclasses import replace
            return replace(state,rejected=state.rejected+1), logits.new_tensor(0.)
        return self.policy.update(desc,state,j,sim,logits,delta)
    def step(self,feature,state,*,write=True,source=False):
        logits,context = self.predict(feature,state,source=source)
        if not write or source: return logits,state,logits.new_tensor(0.)
        state,cost = self.observe(state,logits,context)
        return logits,state,cost
