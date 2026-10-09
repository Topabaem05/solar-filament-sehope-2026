"""Resize-only baseline instance decoding; native-tile refinement is separate."""
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.nn import functional as F

from filament.io.rle import checked_rle, encode_mask


@torch.no_grad()
def decode_instances(output, shape=(2048,2048), *, score_threshold=.5, mask_threshold=.5, nms_iou=.8):
    if any(not 0 < x < 1 for x in [score_threshold,mask_threshold,nms_iou]):
        raise ValueError('Thresholds must be strictly between zero and one.')
    if output['mask_logits'].shape[0]!=1:
        raise ValueError('Decode one original image at a time.')
    objects=output['object_logits'][0].sigmoid().cpu()
    masks=output['mask_logits'][0].cpu()
    if not torch.isfinite(objects).all() or not torch.isfinite(masks).all():
        raise ValueError('Nonfinite predictions.')
    candidates=[]
    for query in (objects>=score_threshold).nonzero().flatten():
        probability=F.interpolate(masks[query][None,None],size=shape,mode='bilinear',align_corners=False)[0,0].sigmoid()
        binary=probability>=mask_threshold
        if not binary.any():
            continue
        confidence=float(objects[query]*probability[binary].mean())
        if confidence < score_threshold:
            continue
        counts=encode_mask(binary.numpy().astype(np.uint8))
        candidates.append((confidence,counts))
    kept=[]
    for confidence,counts in sorted(candidates,reverse=True):
        current=checked_rle(counts,shape)
        if kept and float(np.max(mask_util.iou([current],[r for _,r in kept],[0]*len(kept)))) > nms_iou:
            continue
        kept.append((counts,current))
    return [counts for counts,_ in kept]
