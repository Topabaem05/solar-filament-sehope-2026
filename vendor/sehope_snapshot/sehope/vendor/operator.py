"""Portable four-direction operator with explicit differentiable Content state.

Parameter names and spatial restoration match upstream VisionHOPE (MIT).
"""
import torch
from torch import nn
from ..recurrence import srnl_reference

class SRNL(nn.Module):
    def __init__(self, dim, head_dim=16, chunk_size=None, directions=4):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.num_heads, self.directions = dim // head_dim, directions
        self.chunk_size = chunk_size
        for name in ['m', 'v', 'k']:
            setattr(self, f'M_{name}_0', nn.Parameter(torch.empty(directions, self.num_heads, head_dim, head_dim)))
        for name in ['eta', 'alpha']:
            setattr(self, f'M_{name}_0', nn.Parameter(torch.zeros(directions, self.num_heads, head_dim)))
        self.reset_parameters()
    @property
    def initial_memories(self):
        return tuple(getattr(self, f'M_{n}_0') for n in ['m', 'v', 'k', 'eta', 'alpha'])
    def reset_parameters(self):
        for s in self.initial_memories[:3]: nn.init.trunc_normal_(s, std=.01)
        for s in self.initial_memories[3:]: nn.init.zeros_(s)

class VisionHOPEOperator(nn.Module):
    def __init__(self, dim, head_dim=16, chunk_size=None, direction_merge='scale', share_direction_skip=True, **kwargs):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.num_heads = dim // head_dim
        self.direction_merge, self.share_direction_skip = direction_merge, share_direction_skip
        self.srnl = SRNL(dim, head_dim, chunk_size)
        self.W_q = nn.Linear(dim, dim)
        self.d_skip = nn.Parameter(torch.zeros((self.num_heads, head_dim) if share_direction_skip else (4, self.num_heads, head_dim)))
        self.direction_scale = nn.Parameter(torch.ones(4, dim)) if direction_merge == 'scale' else None
        self.direction_logits = nn.Parameter(torch.zeros(4, dim)) if direction_merge == 'softmax' else None
        self.reset_parameters()
    def reset_parameters(self):
        self.srnl.reset_parameters()
        nn.init.trunc_normal_(self.W_q.weight, std=.02)
        nn.init.zeros_(self.W_q.bias)
    def direction_weights(self):
        if self.direction_merge == 'scale': return self.direction_scale
        if self.direction_merge == 'softmax': return 4 * self.direction_logits.softmax(0)
        return self.d_skip.new_ones(4, self.dim)
    @staticmethod
    def serialize(x):
        a, b = x.flatten(2).transpose(1, 2), x.transpose(2, 3).flatten(2).transpose(1, 2)
        return torch.stack([a, a.flip(1), b, b.flip(1)], 1)
    def forward_state(self, x, content=None):
        batch, channels, h, w = x.shape
        xs = self.serialize(x).reshape(batch, 4, h*w, self.num_heads, self.head_dim).transpose(2, 3)
        qhw = self.W_q(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(batch, channels, h, w)
        qs = self.serialize(qhw).reshape(batch, 4, h*w, self.num_heads, self.head_dim).transpose(2, 3)
        states = self.srnl.initial_memories
        m = states[0] if content is None else content
        if m.ndim == 4: m = m.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        out, final = [], []
        # Pair equal row/column lengths to avoid four separate recurrent graphs.
        for lo, hi, chunk in [(0, 2, w), (2, 4, h)]:
            ss = (m[:, lo:hi],) + tuple(s[None, lo:hi] for s in states[1:])
            y, end = srnl_reference(xs[:, lo:hi], qs[:, lo:hi], ss, self.srnl.chunk_size or chunk)
            out.append(y); final.append(end[0])
        y = torch.cat(out, 1)
        skip = self.d_skip.reshape(1, 1 if self.share_direction_skip else 4, self.num_heads, 1, self.head_dim)
        y = (y + skip * xs).transpose(2, 3).flatten(3)
        views = [y[:, 0].reshape(batch,h,w,channels).permute(0,3,1,2),
                 y[:, 1].flip(1).reshape(batch,h,w,channels).permute(0,3,1,2),
                 y[:, 2].reshape(batch,w,h,channels).permute(0,3,2,1),
                 y[:, 3].flip(1).reshape(batch,w,h,channels).permute(0,3,2,1)]
        weights = self.direction_weights()
        result = sum(weights[i][None,:,None,None] * v for i,v in enumerate(views))
        return result, torch.cat(final, 1), m
    def forward(self, x, *, channels_last=None):
        return self.forward_state(x)[0]
