"""Small FPN + mask-query baseline. This is not a Mask2Former reproduction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError('Decoder width must be divisible by four.')
    yy,xx = torch.meshgrid(torch.linspace(0,1,height,device=device),
                           torch.linspace(0,1,width,device=device),indexing='ij')
    scale = 2*math.pi * 10000**(-torch.arange(channels//4,device=device)/(channels//4))
    x,y = xx.flatten()[:,None]*scale,yy.flatten()[:,None]*scale
    return torch.cat([x.sin(),x.cos(),y.sin(),y.cos()],1).to(dtype)[None]


class InstanceDecoder(nn.Module):
    def __init__(self, channels=(64,128,256,512), width=128, queries=64, layers=3):
        super().__init__()
        if width % 8:
            raise ValueError('FPN width must be divisible by eight.')
        self.width = width
        self.laterals = nn.ModuleList([nn.Conv2d(c,width,1) for c in channels])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=1),
                                      nn.GroupNorm(8,width),nn.GELU()) for _ in channels])
        self.query = nn.Embedding(queries,width)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(width,4,4*width,
                                                   dropout=0.,batch_first=True,norm_first=True),layers)
        self.object_head = nn.Linear(width,1)
        self.mask_head = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))

    def forward(self, features):
        if len(features)!=4:
            raise ValueError('Four feature scales are required.')
        lateral = [layer(x) for layer,x in zip(self.laterals,features)]
        pyramid = [None]*4
        pyramid[3] = self.smooth[3](lateral[3])
        for i in range(2,-1,-1):
            fused = lateral[i]+F.interpolate(pyramid[i+1],size=lateral[i].shape[-2:],mode='bilinear',align_corners=False)
            pyramid[i] = self.smooth[i](fused)
        # Cross attention reads coarser levels; mask logits retain stride-4 detail.
        memory = torch.cat([x.flatten(2).transpose(1,2)+position_embedding(*x.shape[-2:],self.width,x.device,x.dtype)
                            for x in pyramid[2:]],dim=1)
        query = self.query.weight[None].expand(features[0].shape[0],-1,-1)
        query = self.decoder(query,memory)
        masks = torch.einsum('bqc,bchw->bqhw',self.mask_head(query),pyramid[0])
        return {'object_logits':self.object_head(query).squeeze(-1),'mask_logits':masks,'embeddings':query}
