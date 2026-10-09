from dataclasses import dataclass, fields, replace
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class MemoryState:
    values: torch.Tensor
    keys: torch.Tensor
    valid: torch.Tensor
    last_used: torch.Tensor
    hidden: torch.Tensor
    tick: int = 0
    rejected: int = 0
    def detach(self):
        return replace(self, **{f.name: getattr(self,f.name).detach() for f in fields(self) if isinstance(getattr(self,f.name),torch.Tensor)})
    def nbytes(self):
        return sum(getattr(self,f.name).numel()*getattr(self,f.name).element_size() for f in fields(self) if isinstance(getattr(self,f.name),torch.Tensor))

def project(x, radius):
    norm = torch.linalg.vector_norm(x, dim=(-2,-1), keepdim=True)
    return x * (radius / norm.clamp_min(1e-20)).clamp(max=1)

class MemoryPolicy(nn.Module):
    def __init__(self, anchors, channels=512, slots=4, descriptor_dim=128, hidden=64,
                 epsilon=.1, step_fraction=.25, threshold=.75, mode='meta', fixed_beta=.05, fixed_read=.5):
        super().__init__()
        if mode not in {'meta','mlp','fixed','copy'}: raise ValueError(mode)
        if not 0 < epsilon < 1 or not 0 < step_fraction <= 1 or slots < 1: raise ValueError('Memory bounds')
        self.mode, self.slots, self.threshold = mode, slots, threshold
        self.fixed_beta, self.step_fraction = fixed_beta, step_fraction
        if not 0<=fixed_read<=1 or not 0<=fixed_beta<=1:raise ValueError('Fixed policy controls must lie in [0,1]')
        self.fixed_read=fixed_read
        self.shape, self.hidden_size = tuple(anchors.shape), hidden
        radius = epsilon * torch.linalg.vector_norm(anchors.detach(),dim=(-2,-1),keepdim=True)
        self.register_buffer('radius', torch.where(radius > 1e-10, radius, torch.zeros_like(radius)))
        # Fixed projection: hard slot selection cannot train this through argmax.
        generator = torch.Generator().manual_seed(404)
        self.register_buffer('projection', torch.randn(2*channels,descriptor_dim,generator=generator)/math.sqrt(2*channels))
        self.gru = nn.GRUCell(descriptor_dim+4, hidden)
        self.mlp = nn.Sequential(nn.Linear(descriptor_dim+4,hidden),nn.Tanh())
        self.read_head = nn.Linear(descriptor_dim+hidden+1,1)
        self.write_head = nn.Linear(hidden,6)  # beta, gamma, eta and 3 rule logits
        count = math.prod(self.shape[:-2])
        d = self.shape[-1]
        self.writer_left = nn.Parameter(torch.eye(d))
        self.writer_right = nn.Parameter(torch.eye(d))
        self.writer_gain = nn.Parameter(torch.full(self.shape[:-2]+(1,1),.1))
        self.writer_context = nn.Linear(descriptor_dim,count)
        nn.init.normal_(self.writer_context.weight, std=.001)
        nn.init.zeros_(self.writer_context.bias)
        if mode != 'meta': self.gru.requires_grad_(False)
        if mode != 'mlp': self.mlp.requires_grad_(False)
        if mode in {'fixed','copy'}:
            self.read_head.requires_grad_(False); self.write_head.requires_grad_(False)
        if mode == 'copy':
            for p in self.parameters(): p.requires_grad_(False)
    def initial_state(self):
        r = self.radius
        return MemoryState(r.new_zeros(self.slots,*self.shape),r.new_zeros(self.slots,self.projection.shape[1]),
                           torch.zeros(self.slots,dtype=torch.bool,device=r.device),
                           torch.full((self.slots,),-1,dtype=torch.long,device=r.device),r.new_zeros(1,self.hidden_size))
    def descriptor(self, feature):
        x = feature.detach().float()
        z = torch.cat((x.mean((2,3)),x.var((2,3),unbiased=False).add(1e-6).sqrt()),-1)
        return F.normalize(z @ self.projection,dim=-1)[0]
    def read(self, descriptor, state):
        similarity = state.keys @ descriptor
        similarity = similarity.masked_fill(~state.valid, -2)
        j = int(similarity.argmax().item())
        matched = bool(state.valid[j].item()) and float(similarity[j].item()) >= self.threshold
        if not matched: return torch.zeros_like(state.values[0]), -1, descriptor.new_tensor(0.), descriptor.new_tensor(-1.)
        if self.mode in {'fixed','copy'}: rho = descriptor.new_tensor(self.fixed_read)
        else:
            # No current entropy is used before a prediction exists.
            inp = torch.cat((descriptor,state.hidden[0],similarity[j:j+1]))
            rho = self.read_head(inp).sigmoid().squeeze()
        return rho * state.values[j], j, rho, similarity[j]
    def update(self, descriptor, state, selected, similarity, logits, delta):
        # No labels, domain id, future inputs or scoring callbacks in this API.
        if not all(bool(torch.isfinite(t).all()) for t in [descriptor,logits,delta]):
            return replace(state,rejected=state.rejected+1), self.radius.new_tensor(0.)
        if selected < 0:
            free = (~state.valid).nonzero().flatten()
            j = int(free[0].item()) if len(free) else int(state.last_used.argmin().item())
            old = torch.zeros_like(state.values[j])
        else:
            j, old = selected, state.values[selected]
        p = logits.float().softmax(-1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum()/math.log(max(2,logits.shape[-1]))
        conflict = F.cosine_similarity(old.flatten(),delta.flatten(),dim=0,eps=1e-12)
        dn = delta.norm()/(self.radius.norm()+1e-12)
        stats = torch.stack([entropy, similarity, dn.log1p().clamp(max=20), conflict])
        inp = torch.cat([descriptor,stats]).unsqueeze(0)
        if self.mode == 'meta': hidden = self.gru(inp,state.hidden)
        elif self.mode == 'mlp': hidden = self.mlp(inp)
        else: hidden = state.hidden
        if self.mode in {'fixed','copy'}:
            beta, gamma, eta = self.fixed_beta, 1., self.fixed_beta
            mix = delta.new_tensor([0.,1.,0.])
        else:
            controls = self.write_head(hidden)[0]
            beta, gamma, eta = controls[:3].sigmoid().unbind()
            mix = controls[3:].softmax(-1)
        if self.mode == 'copy': candidate = delta
        else:
            context = self.writer_context(descriptor).reshape(self.shape[:-2]+(1,1)).tanh()
            candidate = (self.writer_gain+context) * (self.writer_left @ delta @ self.writer_right)
        candidate = project(candidate,self.radius)
        avg = old + beta*(candidate-old)
        # Unit key uses unlabeled current fast-memory change; fixed coordinate space.
        key = F.normalize(delta.mean(-2),dim=-1,eps=1e-12)
        error = ((candidate-gamma*old) @ key.unsqueeze(-1))
        corrected = gamma*old + eta*error*key.unsqueeze(-2)
        proposed = mix[0]*old + mix[1]*avg + mix[2]*corrected
        if not bool(torch.isfinite(proposed).all()) or not bool(torch.isfinite(hidden).all()):
            return replace(state,rejected=state.rejected+1), self.radius.new_tensor(0.)
        value = project(old+project(proposed-old,self.radius*self.step_fraction),self.radius)
        selector = F.one_hot(torch.tensor(j,device=value.device),self.slots).to(value.dtype)
        values = state.values*(1-selector.reshape(-1,*([1]*len(self.shape)))) + value.unsqueeze(0)*selector.reshape(-1,*([1]*len(self.shape)))
        key_new = descriptor if selected < 0 else F.normalize(.95*state.keys[j]+.05*descriptor,dim=0)
        keys = state.keys*(1-selector[:,None])+key_new[None]*selector[:,None]
        valid, times = state.valid.clone(),state.last_used.clone()
        valid[j],times[j] = True,state.tick
        changed = ((value-old).square()/(self.radius.square()+1e-20)).mean()
        return MemoryState(values,keys,valid,times,hidden,state.tick+1,state.rejected), changed
