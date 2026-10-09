import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.hierarchical import HierarchicalVisionHOPE

UPSTREAM_COMMIT = '98cfb5d6004472a0ba3d212d3cf884393e39c134'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_tiny(path):
    model = HierarchicalVisionHOPE()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'state_dict'}: raise ValueError('Expected official state_dict container')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model

def stateful_block(block, x, content):
    if block.cpe is not None: x = x + block.cpe(x)
    local = block._local_features(block.norm1(x))
    branch, end, init = block.operator.forward_state(local, content)
    if block.local_position is not None: branch = branch + block.local_position(local)
    branch = F.conv2d(branch, block.proj_out.weight[:, :, None, None], block.proj_out.bias)
    x = block._residual_add(x, branch, block.gamma1)
    x = block._residual_add(x, block.ffn(block.norm2(x)), block.gamma2)
    return x, end, init

class SplitBackbone(nn.Module):
    def __init__(self, model, adapted_blocks=2):
        super().__init__()
        if not 1 <= adapted_blocks <= len(model.stages[-1]): raise ValueError('adapted_blocks')
        self.model, self.adapted_blocks = model, adapted_blocks
        self.cut = len(model.stages[-1]) - adapted_blocks
        self.requires_grad_(False)
        self.eval()
    def train(self, mode=True):
        # Freeze BN running buffers and DropPath as well as weights.
        super().train(False)
        return self
    @property
    def anchors(self):
        return torch.stack([b.operator.srnl.M_m_0 for b in self.model.stages[-1][self.cut:]])
    @torch.no_grad()
    def prefix(self, x):
        for i in range(3): x = self.model.stages[i](self.model.downsample_layers[i](x))
        x = self.model.downsample_layers[3](x)
        for b in self.model.stages[3][:self.cut]: x = b(x)
        return x.detach()
    def suffix(self, feature, offsets):
        if feature.shape[0] != 1: raise ValueError('One independent image per stream step required')
        x, ends, starts = feature, [], []
        for i,b in enumerate(self.model.stages[-1][self.cut:]):
            x, end, init = stateful_block(b, x, self.anchors[i] + offsets[i])
            ends.append(end[0]); starts.append(init[0])
        m = self.model
        x = m.head_act(m.head_norm(m.head_pre_logits(m.norm(x)))).mean((2,3))
        return m.head(x), torch.stack(ends), torch.stack(starts)
