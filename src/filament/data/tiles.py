"""Native, unpadded tiles with explicit half-open image coordinates.

Inference grids depend only on image shape. Target cropping is training-only and
preserves annotation identity and whether foreground was removed by the crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, order=True)
class Tile:
    index: int
    y: int
    x: int
    height: int
    width: int

    def __post_init__(self):
        values = (self.index, self.y, self.x, self.height, self.width)
        if any(type(v) is not int for v in values):
            raise ValueError('Tile coordinates must be integers.')
        if min(self.index, self.y, self.x) < 0 or min(self.height, self.width) <= 0:
            raise ValueError('Invalid tile extent.')

    @property
    def shape(self):
        return self.height, self.width

    @property
    def slices(self):
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def validate_in(self, shape):
        if self.y + self.height > shape[0] or self.x + self.width > shape[1]:
            raise ValueError('Tile extends beyond the original image.')


def axis_starts(length, tile_size, stride):
    if any(type(v) is not int or v <= 0 for v in (length, tile_size, stride)):
        raise ValueError('Grid dimensions must be positive integers.')
    if stride > tile_size:
        raise ValueError('Stride cannot leave coverage gaps.')
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def native_grid(shape, tile_size=512, stride=384):
    if len(shape) != 2:
        raise ValueError('Expected image shape (height, width).')
    ys = axis_starts(shape[0], tile_size, stride)
    xs = axis_starts(shape[1], tile_size, stride)
    return [Tile(i, y, x, min(tile_size, shape[0]), min(tile_size, shape[1]))
            for i, (y, x) in enumerate((y, x) for y in ys for x in xs)]


def validate_coverage(tiles, shape):
    """Reject missing, duplicate or out-of-bounds tiles; return pixel counts."""
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('Invalid original shape.')
    coverage = np.zeros(shape, dtype=np.uint32)
    ids, extents = set(), set()
    for tile in tiles:
        tile.validate_in(shape)
        extent = (tile.y, tile.x, tile.height, tile.width)
        if tile.index in ids or extent in extents:
            raise ValueError('Duplicate tile ID or extent.')
        ids.add(tile.index)
        extents.add(extent)
        coverage[tile.slices] += 1
    if not np.all(coverage):
        raise ValueError('Tile grid does not cover every image pixel.')
    return coverage


def crop_native(array, tile):
    """Crop the last two axes without interpolation or padding."""
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError('Expected at least two spatial axes.')
    tile.validate_in(array.shape[-2:])
    return array[(..., *tile.slices)].copy()


def restore_native(array, tile, shape):
    array = np.asarray(array)
    if array.shape != tile.shape:
        raise ValueError('Local mask dimensions differ from tile dimensions.')
    tile.validate_in(shape)
    restored = np.zeros(shape, dtype=array.dtype)
    restored[tile.slices] = array
    return restored


@dataclass
class TileTargets:
    masks: np.ndarray
    annotation_ids: tuple
    truncated: np.ndarray
    retained_fraction: np.ndarray


def crop_targets(masks, annotation_ids, tile):
    masks = np.asarray(masks)
    if masks.ndim != 3 or len(masks) != len(annotation_ids):
        raise ValueError('Expected one annotation ID per N,H,W mask.')
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError('Annotation IDs must be unique within a record.')
    if not np.isin(masks, (0, 1)).all():
        raise ValueError('Targets must be finite binary masks.')
    original_area = masks.sum(axis=(1, 2))
    if np.any(original_area == 0):
        raise ValueError('Original instance masks must be nonempty.')
    cropped = crop_native(masks, tile)
    area = cropped.sum(axis=(1, 2))
    keep = area > 0
    return TileTargets(
        cropped[keep].astype(np.uint8),
        tuple(i for i, k in zip(annotation_ids, keep) if k),
        (area < original_area)[keep],
        (area[keep] / original_area[keep]).astype(np.float32),
    )


def load_native_image(path, expected_shape=(2048, 2048)):
    with Image.open(Path(path)) as image:
        if image.mode != 'L' or image.size[::-1] != expected_shape:
            raise ValueError('Unexpected native grayscale image format.')
        return np.asarray(image, dtype=np.uint8).copy()


def normalized_tile(pixels, tile):
    """Same mean/std as v0.1, but keep every native pixel in the tile."""
    import torch

    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError('Expected native uint8 grayscale pixels.')
    local = crop_native(pixels, tile).astype(np.float32) / 255
    return ((torch.from_numpy(local)[None].repeat(3, 1, 1) - .5) / .25)[None]


def positive_taper(shape, minimum=.05):
    if not 0 < minimum <= 1:
        raise ValueError('Taper minimum must be positive and at most one.')
    axes = [np.hanning(n) if n > 2 else np.ones(n) for n in shape]
    return np.maximum(np.outer(*axes), minimum).astype(np.float32)
