"""Strict mask/COCO-RLE/CSV contracts, independent of model code."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


def checked_rle(counts: str | bytes, shape: tuple[int, int]) -> dict:
    """Validate decoded run lengths before handing data to the native decoder."""
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Mask shape must contain two positive integers.')
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    if not isinstance(counts, str) or not counts:
        raise ValueError('Expected nonempty compressed COCO counts.')
    runs, value, shift, total = [], 0, 0, 0
    for char in counts:
        c = ord(char) - 48
        if not 0 <= c <= 63:
            raise ValueError('Invalid compressed COCO character.')
        value |= (c & 31) << shift
        shift += 5
        if shift > 40:
            raise ValueError('Invalid COCO run length.')
        if c & 32:
            continue
        if c & 16:
            value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError('Negative COCO run length.')
        total += value
        if total > shape[0] * shape[1]:
            raise ValueError('RLE exceeds image dimensions.')
        runs.append(value)
        value, shift = 0, 0
    if shift or total != shape[0] * shape[1]:
        raise ValueError('Incomplete RLE or incompatible image dimensions.')
    return {'size': list(shape), 'counts': counts.encode('ascii')}


def encode_mask(mask: np.ndarray, *, shape=None, allow_empty=False) -> str:
    mask = np.asarray(mask)
    if mask.ndim != 2 or (shape is not None and tuple(mask.shape) != tuple(shape)):
        raise ValueError('Incorrect mask dimensions.')
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Masks must be finite and binary.')
    if not allow_empty and not mask.any():
        raise ValueError('Empty instance masks are not predictions.')
    return coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))['counts'].decode('ascii')


def decode_mask(counts: str | bytes, shape=(2048, 2048), *, allow_empty=False) -> np.ndarray:
    rle = checked_rle(counts, tuple(shape))
    if not allow_empty and int(coco_mask.area(rle)) == 0:
        raise ValueError('Empty instance masks are not predictions.')
    return coco_mask.decode(rle)


def validate_predictions(predictions: dict[str, list[str]], image_ids, shape=(2048, 2048)):
    expected = set(image_ids)
    if set(predictions) != expected:
        raise ValueError(f'Image coverage differs: missing={sorted(expected-set(predictions))[:3]}, '
                         f'unknown={sorted(set(predictions)-expected)[:3]}')
    for image_id, counts_list in predictions.items():
        if not image_id or '_' in image_id or '/' in image_id or '\\' in image_id:
            raise ValueError('Image IDs must be original filename stems.')
        fingerprints = set()
        for counts in counts_list:
            mask = decode_mask(counts, shape)
            # Canonicalize so alternate run encodings of identical pixels also collide.
            canonical = encode_mask(mask)
            if canonical in fingerprints:
                raise ValueError(f'Duplicate instance mask for {image_id}.')
            fingerprints.add(canonical)


def write_submission(path, predictions, image_ids, shape=(2048, 2048)):
    """Write CSV plus a manifest that explicitly records zero-detection images."""
    validate_predictions(predictions, image_ids, shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for image_id in sorted(predictions):
            for i, counts in enumerate(predictions[image_id], 1):
                writer.writerow([f'{image_id}_{i}', counts])
    temp.replace(path)
    manifest = {'shape': list(shape), 'image_counts': {k: len(v) for k, v in sorted(predictions.items())},
                'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'zero_detection_policy': 'zero_rows_in_csv_explicit_in_manifest',
                'kaggle_server_empty_image_contract': 'not_verified'}
    path.with_suffix(path.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def read_submission(path, image_ids, shape=(2048, 2048), *, require_manifest=True):
    path = Path(path)
    predictions = {image_id: [] for image_id in image_ids}
    seen_ids = set()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Incorrect CSV columns.')
        for row in reader:
            row_id = row['filament_id']
            if None in row or row_id in seen_ids or '_' not in row_id:
                raise ValueError('Malformed row or duplicate filament_id.')
            image_id, suffix = row_id.rsplit('_', 1)
            if image_id not in predictions or not suffix:
                raise ValueError('Unknown image ID or empty suffix.')
            seen_ids.add(row_id)
            predictions[image_id].append(row['segmentation_rle'])
    validate_predictions(predictions, image_ids, shape)
    if require_manifest:
        manifest = json.loads(path.with_suffix(path.suffix + '.manifest.json').read_text())
        if manifest['shape'] != list(shape) or manifest['image_counts'] != {k: len(v) for k, v in predictions.items()}:
            raise ValueError('CSV coverage does not match manifest.')
        if manifest['csv_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('CSV hash does not match manifest.')
    return predictions
