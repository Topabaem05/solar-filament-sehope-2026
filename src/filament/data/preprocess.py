"""Explicit grayscale preprocessing for the initial resize-only baseline."""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

PREPROCESS_VERSION='grayscale_repeat3_mean0.5_std0.25_bilinear_v1'


def load_image(path, size):
    if size < 32 or size % 32:
        raise ValueError('Input size must be a positive multiple of 32, at least 32.')
    with Image.open(path) as image:
        if image.mode != 'L':
            raise ValueError('Expected single-channel grayscale JPEG.')
        if image.size != (2048,2048):
            raise ValueError('Competition images must have native dimensions 2048 x 2048.')
        image=image.resize((size,size),Image.Resampling.BILINEAR)
        pixels=np.asarray(image,dtype=np.float32).copy()/255
    return ((torch.from_numpy(pixels)[None].repeat(3,1,1)-.5)/.25)[None]


def resized_targets(dataset, record, size):
    masks=torch.from_numpy(dataset.masks(record).copy()).float()
    if not len(masks):
        return torch.empty(0,size,size),0
    masks=F.interpolate(masks[:,None],size=(size,size),mode='nearest-exact')[:,0]
    keep=masks.flatten(1).sum(1)>0
    return masks[keep],int((~keep).sum())
