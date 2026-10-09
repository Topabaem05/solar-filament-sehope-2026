"""Optional unmodified official CUDA prefix for one-time feature extraction."""
from torch import nn

class OfficialPrefix(nn.Module):
    def __init__(self,model,blocks):
        super().__init__();self.model=model;self.adapted_blocks=blocks
        self.requires_grad_(False);self.eval()
    def prefix(self,x):
        for i in range(3):x=self.model.stages[i](self.model.downsample_layers[i](x))
        x=self.model.downsample_layers[3](x)
        for block in self.model.stages[3][:-self.adapted_blocks]:x=block(x)
        return x

def official_prefix(weights,blocks):
    from visionhope.models import create_model
    return OfficialPrefix(create_model('visionhope_tiny',checkpoint_path=weights),blocks)
