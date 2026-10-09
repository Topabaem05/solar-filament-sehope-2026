"""Conservative mask association across native tiles; no GT or distance bridge.

Pairwise Hungarian matching uses IoU only in the common field of view. Cluster
unions cannot contain two different queries from the same tile. Thresholds here
are starting values, not calibrated validation results.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from filament.data.tiles import Tile, positive_taper


@dataclass
class TilePrediction:
    tile: Tile
    query_id: int
    score: float
    probability: np.ndarray

    @property
    def key(self):
        return self.tile.index, self.query_id

    def validate(self, shape):
        self.tile.validate_in(shape)
        if type(self.query_id) is not int or self.query_id < 0:
            raise ValueError('Invalid query ID.')
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('Invalid prediction score.')
        p = np.asarray(self.probability)
        if p.shape != self.tile.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Expected finite tile probabilities in [0,1].')


def overlap_slices(a, b):
    y, x = max(a.y, b.y), max(a.x, b.x)
    y1, x1 = min(a.y + a.height, b.y + b.height), min(a.x + a.width, b.x + b.width)
    if y >= y1 or x >= x1:
        return None
    return ((slice(y - a.y, y1 - a.y), slice(x - a.x, x1 - a.x)),
            (slice(y - b.y, y1 - b.y), slice(x - b.x, x1 - b.x)))


def associate_instances(predictions, shape, *, mask_threshold=.5, overlap_iou=.5,
                        min_shared_pixels=4):
    if not 0 < mask_threshold < 1 or not 0 < overlap_iou <= 1:
        raise ValueError('Invalid association threshold.')
    if type(min_shared_pixels) is not int or min_shared_pixels < 1:
        raise ValueError('Association requires shared foreground pixels.')
    predictions = sorted(predictions, key=lambda p: p.key)
    keys, tiles, by_tile = set(), {}, {}
    binary = []
    for i, pred in enumerate(predictions):
        pred.validate(shape)
        if pred.key in keys:
            raise ValueError('Duplicate tile/query key.')
        keys.add(pred.key)
        if pred.tile.index in tiles and tiles[pred.tile.index] != pred.tile:
            raise ValueError('Same tile ID has conflicting geometry.')
        tiles[pred.tile.index] = pred.tile
        by_tile.setdefault(pred.tile.index, []).append(i)
        binary.append(np.asarray(pred.probability) >= mask_threshold)
        if not binary[-1].any():
            raise ValueError('Empty local predictions must be filtered before association.')

    edges = []
    for ta, tb in combinations(sorted(tiles), 2):
        shared = overlap_slices(tiles[ta], tiles[tb])
        if shared is None:
            continue
        aa, bb = by_tile[ta], by_tile[tb]
        scores = np.zeros((len(aa), len(bb)), dtype=np.float64)
        for row, a in enumerate(aa):
            ma = binary[a][shared[0]]
            for col, b in enumerate(bb):
                mb = binary[b][shared[1]]
                intersection = np.count_nonzero(ma & mb)
                if intersection < min_shared_pixels:
                    continue
                iou = intersection / np.count_nonzero(ma | mb)
                if iou >= overlap_iou:
                    scores[row, col] = iou
        # Invalid edges have zero reward and are discarded, including assignments
        # used merely to fill the rectangular matching matrix.
        rows, cols = linear_sum_assignment(-scores)
        edges.extend((float(scores[r, c]), aa[r], bb[c]) for r, c in zip(rows, cols)
                     if scores[r, c] > 0)

    parent = list(range(len(predictions)))
    tile_sets = [{p.tile.index} for p in predictions]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    accepted, conflicts = [], []
    for score, a, b in sorted(edges, key=lambda e: (-e[0], predictions[e[1]].key, predictions[e[2]].key)):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        item = {'a': list(predictions[a].key), 'b': list(predictions[b].key), 'overlap_iou': score}
        if tile_sets[ra] & tile_sets[rb]:
            conflicts.append(item)
            continue
        parent[rb] = ra
        tile_sets[ra] |= tile_sets[rb]
        accepted.append(item)
    grouped = {}
    for i, pred in enumerate(predictions):
        grouped.setdefault(find(i), []).append(pred)
    clusters = sorted(grouped.values(), key=lambda c: c[0].key)
    return clusters, {'local_instances': len(predictions), 'candidate_pair_matches': len(edges),
                      'clusters': len(clusters), 'accepted_edges': accepted,
                      'rejected_same_tile_conflicts': conflicts}


@dataclass
class FusedInstance:
    mask: np.ndarray
    score: float
    members: tuple


def fuse_cluster(cluster, shape, *, mask_threshold=.5, taper_minimum=.05):
    if not cluster or not 0 < mask_threshold < 1:
        raise ValueError('Expected nonempty cluster and valid threshold.')
    if len({p.tile.index for p in cluster}) != len(cluster):
        raise ValueError('Cluster cannot contain multiple queries from one tile.')
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape, dtype=np.float32)
    for pred in cluster:
        pred.validate(shape)
        weight = positive_taper(pred.tile.shape, taper_minimum)
        numerator[pred.tile.slices] += np.asarray(pred.probability, dtype=np.float32) * weight
        denominator[pred.tile.slices] += weight
    covered = denominator > 0
    probability = np.divide(numerator, denominator, out=numerator, where=covered)
    mask = covered & (probability >= mask_threshold)
    # Max avoids artificially rewarding detection count across overlapping tiles.
    # Calibration and duplicate suppression across distinct clusters are separate.
    return FusedInstance(mask, max(p.score for p in cluster), tuple(p.key for p in cluster))
