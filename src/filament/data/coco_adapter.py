"""Preserve each annotator's instance set; expose only segmentation targets."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


def polygon_to_rle(segmentation, shape):
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError('This dataset adapter expects nonempty COCO polygons.')
    for polygon in segmentation:
        coords = np.asarray(polygon)
        if coords.ndim != 1 or len(coords) < 6 or len(coords) % 2 or not np.isfinite(coords).all():
            raise ValueError('Invalid polygon coordinates.')
    parts = coco_mask.frPyObjects(segmentation, *shape)
    rle = coco_mask.merge(parts)
    if int(coco_mask.area(rle)) == 0:
        raise ValueError('Polygon rasterizes to an empty mask.')
    return rle


class AnnotationSet:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        self.images = {}
        self.by_filename = defaultdict(list)
        self.annotations = defaultdict(list)
        self._rles = {}
        for image in data['images']:
            if image['id'] in self.images:
                raise ValueError('Duplicate annotator-image ID.')
            filename = image['file_name']
            if Path(filename).name != filename:
                raise ValueError('file_name must be a basename.')
            if any(type(image[k]) is not int or image[k] <= 0 for k in ('height', 'width')):
                raise ValueError('Invalid image shape.')
            self.images[image['id']] = image
            self.by_filename[filename].append(image['id'])
        seen = set()
        for annotation in data['annotations']:
            if annotation['id'] in seen or annotation['image_id'] not in self.images:
                raise ValueError('Duplicate annotation ID or orphan reference.')
            seen.add(annotation['id'])
            self.annotations[annotation['image_id']].append(annotation)
        for filename, records in self.by_filename.items():
            if len({self.shape(r) for r in records}) != 1:
                raise ValueError(f'Annotators disagree on dimensions: {filename}')
        for records in self.by_filename.values():
            records.sort()

    def shape(self, record_id):
        image = self.images[record_id]
        return image['height'], image['width']

    def rles(self, record_id):
        if record_id not in self._rles:
            self._rles[record_id] = [polygon_to_rle(a['segmentation'], self.shape(record_id))
                                     for a in self.annotations[record_id]]
        return self._rles[record_id]

    def masks(self, record_id):
        rles = self.rles(record_id)
        if not rles:
            return np.zeros((0, *self.shape(record_id)), dtype=np.uint8)
        return coco_mask.decode(rles).transpose(2, 0, 1)

    def filenames(self):
        return sorted(self.by_filename)
