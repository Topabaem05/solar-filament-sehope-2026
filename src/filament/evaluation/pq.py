"""Official public-notebook PQ arithmetic plus a separate one-to-one diagnostic."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from filament.io.rle import checked_rle, validate_predictions


def overlaps(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    return np.asarray(coco_mask.iou(pred, gt, [0]*len(gt)), dtype=np.float64).T


def summarize(matrices, *, one_to_one=False):
    tp = fp = fn = 0
    iou_sum = 0.
    gt_degrees, pred_degrees = Counter(), Counter()
    for raw in matrices:
        m = np.asarray(raw, dtype=np.float64)
        if m.ndim != 2 or not np.isfinite(m).all() or ((m < 0) | (m > 1)).any():
            raise ValueError('Invalid overlap matrix.')
        hits = m > .5
        if one_to_one and min(m.shape) > 0:
            # Maximize valid pair cardinality first, then summed IoU.
            score = hits * (min(m.shape)+1+m)
            rows, cols = linear_sum_assignment(-score)
            selected = np.zeros_like(hits)
            selected[rows, cols] = hits[rows, cols]
            hits = selected
        tp += int(hits.sum())
        fp += int((hits.sum(0) == 0).sum())
        fn += int((hits.sum(1) == 0).sum())
        iou_sum += float(m[hits].sum())
        gt_degrees.update(map(int, (m > 0).sum(1)))
        pred_degrees.update(map(int, (m > 0).sum(0)))
    denominator = tp + .5*fp + .5*fn
    return {'pq': iou_sum/denominator if denominator else 0., 'tp': tp, 'fp': fp, 'fn': fn,
            'matched_iou_sum': iou_sum, 'denominator': denominator,
            'gt_overlap_degree_histogram': dict(sorted(gt_degrees.items())),
            'prediction_overlap_degree_histogram': dict(sorted(pred_degrees.items())),
            'split_gt_count': sum(n for degree, n in gt_degrees.items() if degree > 1),
            'merged_prediction_count': sum(n for degree, n in pred_degrees.items() if degree > 1)}


def evaluate(dataset, predictions, filenames):
    filenames = sorted(filenames)
    if not set(filenames) <= set(dataset.filenames()) or len(set(filenames)) != len(filenames):
        raise ValueError('Unknown or duplicated evaluation image.')
    shapes = {dataset.shape(r) for n in filenames for r in dataset.by_filename[n]}
    if len(shapes) != 1:
        raise ValueError('Evaluation requires one image shape and a nonempty image set.')
    shape = shapes.pop()
    validate_predictions(predictions, [Path(n).stem for n in filenames], shape)
    matrices, per_record = [], []
    for filename in filenames:
        pred = [checked_rle(s, shape) for s in predictions[Path(filename).stem]]
        for record_id in dataset.by_filename[filename]:
            m = overlaps(dataset.rles(record_id), pred)
            matrices.append(m)
            per_record.append({'annotator_image_id': record_id, **summarize([m])})
    return {'source': 'public_notebook_arithmetic_not_verified_kaggle_server',
            'original_images': len(filenames), 'annotator_image_records': len(matrices),
            'official': summarize(matrices), 'one_to_one_diagnostic': summarize(matrices, one_to_one=True),
            'per_record': per_record}
