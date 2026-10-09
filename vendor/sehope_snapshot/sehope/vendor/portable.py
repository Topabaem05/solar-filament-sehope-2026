"""Pure PyTorch replacements for optional CUDA utilities. No kernel compilation."""
import torch
from torch import nn

def can_fuse_residual_add(*args): return False
def can_fuse_layernorm2d(*args): return False
def visionhope_residual_add(*args): raise RuntimeError("CUDA fusion is disabled")
def visionhope_layernorm2d(*args): raise RuntimeError("CUDA fusion is disabled")

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = True
    def forward(self, x):
        if not self.training or self.drop_prob == 0: return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim-1)).bernoulli_(keep)
        return x * mask / keep

class GlobalResponseNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
    def forward(self, x):
        x = x.float()
        gx = torch.linalg.vector_norm(x, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(1, keepdim=True) + self.eps)
        return x * (1 + self.gamma * nx) + self.beta
