"""Whole-image native-grid inference; the predictor receives pixels only.

Association currently uses local mask evidence only. Global context, detail
refinement and OOF threshold calibration are subsequent experimental stages.
"""
import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.nn import functional as F

from filament.association import TilePrediction, associate_instances, fuse_cluster
from filament.data.tiles import native_grid, normalized_tile, validate_coverage
from filament.io.rle import checked_rle, encode_mask

NATIVE_PREPROCESS_VERSION = 'native_grayscale_repeat3_mean0.5_std0.25_v1'


def _thresholds(*values):
    if any(not 0 < value < 1 for value in values):
        raise ValueError('Thresholds must be strictly between zero and one.')


@torch.no_grad()
def decode_tile(output, tile, *, object_threshold=.5, score_threshold=.5,
                mask_threshold=.5, nms_iou=.8):
    _thresholds(object_threshold, score_threshold, mask_threshold, nms_iou)
    logits, masks = output['object_logits'], output['mask_logits']
    if (logits.ndim != 2 or logits.shape[0] != 1 or masks.ndim != 4
            or masks.shape[:2] != logits.shape):
        raise ValueError('Expected one tile with matching object/mask query dimensions.')
    if not torch.isfinite(logits).all() or not torch.isfinite(masks).all():
        raise ValueError('Nonfinite predictions.')
    objects = logits[0].sigmoid().cpu()
    masks = masks[0].cpu()
    stats = dict(queries=len(objects), below_object=0, empty_mask=0,
                 below_score=0, local_nms_removed=0, retained=0)
    candidates = []
    for query, object_score in enumerate(objects):
        if object_score < object_threshold:
            stats['below_object'] += 1
            continue
        p = F.interpolate(masks[query][None, None], size=tile.shape,
                          mode='bilinear', align_corners=False)[0, 0].sigmoid().numpy()
        binary = p >= mask_threshold
        if not binary.any():
            stats['empty_mask'] += 1
            continue
        score = float(object_score) * float(p[binary].mean())
        if score < score_threshold:
            stats['below_score'] += 1
            continue
        rle = checked_rle(encode_mask(binary.astype(np.uint8)), tile.shape)
        candidates.append((TilePrediction(tile, query, score, p), rle))
    kept = []
    for pred, rle in sorted(candidates, key=lambda c: (-c[0].score, c[0].query_id)):
        if kept and float(np.max(coco_mask.iou([rle], [r for _, r in kept], [0] * len(kept)))) > nms_iou:
            stats['local_nms_removed'] += 1
        else:
            kept.append((pred, rle))
    stats['retained'] = len(kept)
    return [p for p, _ in kept], stats


@torch.no_grad()
def predict_native(pixels, predictor, *, tile_size=512, stride=384,
                   object_threshold=.5, score_threshold=.5, mask_threshold=.5,
                   nms_iou=.8, overlap_iou=.5, min_shared_pixels=4):
    """Return full-size RLEs and filtering diagnostics, including empty images.

    ``predictor`` maps a normalized CPU tensor [1,3,H,W] to query outputs. It
    handles model/device placement. Neither annotations nor crop proposals are
    accepted here. Every grid tile runs, regardless of prior detections.
    """
    _thresholds(object_threshold, score_threshold, mask_threshold, nms_iou)
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError('Expected original uint8 grayscale image.')
    shape = pixels.shape
    tiles = native_grid(shape, tile_size, stride)
    coverage = validate_coverage(tiles, shape)
    all_predictions, per_tile = [], []
    for tile in tiles:
        output = predictor(normalized_tile(pixels, tile))
        local, stats = decode_tile(output, tile, object_threshold=object_threshold,
                                   score_threshold=score_threshold,
                                   mask_threshold=mask_threshold, nms_iou=nms_iou)
        all_predictions.extend(local)
        per_tile.append({'tile_index': tile.index, 'y': tile.y, 'x': tile.x, **stats})
    clusters, association = associate_instances(all_predictions, shape,
                                                mask_threshold=mask_threshold,
                                                overlap_iou=overlap_iou,
                                                min_shared_pixels=min_shared_pixels)
    # Fuse and immediately encode one cluster at a time, avoiding a stack of
    # dense full-resolution probabilities for every predicted instance.
    encoded, empty = [], 0
    for cluster in clusters:
        fused = fuse_cluster(cluster, shape, mask_threshold=mask_threshold)
        if not fused.mask.any():
            empty += 1
            continue
        counts = encode_mask(fused.mask.astype(np.uint8))
        encoded.append((fused.score, fused.members, counts, checked_rle(counts, shape)))
    kept, suppressed = [], 0
    for candidate in sorted(encoded, key=lambda c: (-c[0], c[1])):
        if kept and float(np.max(coco_mask.iou([candidate[3]], [p[3] for p in kept], [0] * len(kept)))) > nms_iou:
            suppressed += 1
        else:
            kept.append(candidate)
    report = {'preprocessing': NATIVE_PREPROCESS_VERSION, 'shape': list(shape),
              'tile_count': len(tiles), 'coverage_minimum': int(coverage.min()),
              'tile_size': tile_size, 'stride': stride,
              'thresholds': dict(object=object_threshold, score=score_threshold,
                                 mask=mask_threshold, nms=nms_iou,
                                 overlap_iou=overlap_iou, min_shared_pixels=min_shared_pixels),
              'per_tile': per_tile, 'association': association,
              'empty_fused_instances': empty, 'final_nms_removed': suppressed,
              'output_instances': len(kept),
              'instance_members': [list(p[1]) for p in kept]}
    return [p[2] for p in kept], report
