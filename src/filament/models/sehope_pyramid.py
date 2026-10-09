"""Read an explicit local SE-HOPE checkout and expose spatial features."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
from torch import nn


class SEHopePyramid(nn.Module):
    def __init__(self, source_root, weights=None):
        super().__init__()
        source_root = Path(source_root).resolve()
        manifest = json.loads((source_root/'weights_manifest.json').read_text())
        weights = Path(weights) if weights else source_root/'weights'/manifest['file']
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Backbone weight SHA256 differs from the source manifest.')
        sys.path.insert(0,str(source_root))
        try:
            module = importlib.import_module('sehope.backbone')
        finally:
            sys.path.remove(str(source_root))
        if Path(module.__file__).resolve() != source_root/'sehope/backbone.py':
            raise ValueError('A different SE-HOPE checkout is already imported.')
        self.model = module.load_tiny(weights)
        self.stateful_block = module.stateful_block
        self.cut = len(self.model.stages[3])-2
        self.requires_grad_(False)
        self.eval()
        self.provenance = {'source_root':str(source_root), 'weights_sha256':manifest['sha256'],
                           'source_sha256':{str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in sorted((source_root/'sehope').rglob('*.py'))}}

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def anchors(self):
        return torch.stack([b.operator.srnl.M_m_0 for b in self.model.stages[3][self.cut:]])

    @torch.no_grad()
    def prefix(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected normalized [B,3,H,W] grayscale-replicated input.')
        features = []
        x = image
        for i in range(3):
            x = self.model.stages[i](self.model.downsample_layers[i](x))
            features.append(x.detach())
        x = self.model.downsample_layers[3](x)
        for block in self.model.stages[3][:self.cut]:
            x = block(x)
        return (*features,x.detach())

    def suffix(self, prefix, offsets=None):
        x = prefix[-1]
        if offsets is None:
            offsets = torch.zeros_like(self.anchors)
        if offsets.shape != self.anchors.shape:
            raise ValueError('Incorrect Content offset shape.')
        ends, starts = [], []
        for i,block in enumerate(self.model.stages[3][self.cut:]):
            x,end,initial = self.stateful_block(block,x,self.anchors[i]+offsets[i])
            ends.append(end)
            starts.append(initial)
        return (*prefix[:3],x), torch.stack(ends)-torch.stack(starts)

    def forward(self, image, offsets=None):
        return self.suffix(self.prefix(image), offsets)[0]
