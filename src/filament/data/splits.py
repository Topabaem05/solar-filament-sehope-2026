"""Deterministic temporal/exact-duplicate groups with balanced partitions."""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def observation_time(filename):
    return datetime.strptime(Path(filename).stem[:14], '%Y%m%d%H%M%S')


def components(filenames, max_gap_hours=72, duplicate_groups=()):
    names = sorted(set(filenames), key=lambda n: (observation_time(n), n))
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in zip(names, names[1:]):
        if (observation_time(b)-observation_time(a)).total_seconds() <= max_gap_hours*3600:
            union(a, b)
    for group in duplicate_groups:
        if not set(group) <= set(names):
            raise ValueError('Duplicate group references unknown training images.')
        for name in group[1:]:
            union(group[0], name)
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    return sorted([sorted(g) for g in groups.values()])


def make_splits(dataset, *, seed=42, max_gap_hours=72, duplicate_groups=(), pixel_audit_sha256=None):
    groups = components(dataset.filenames(), max_gap_hours, duplicate_groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    # Image counts, instrument and year distributions; no masks used for allocation.
    keys = ['size'] + sorted({f'station:{Path(n).stem[-2:]}' for n in dataset.filenames()})
    keys += sorted({f'year:{n[:4]}' for n in dataset.filenames()})

    def vector(group):
        counts = Counter({'size': len(group)})
        counts.update(f'station:{Path(n).stem[-2:]}' for n in group)
        counts.update(f'year:{n[:4]}' for n in group)
        return np.array([counts[k] for k in keys], dtype=float)

    total = vector(dataset.filenames())
    fractions = np.array([.15, .85/3, .85/3, .85/3])
    targets = fractions[:, None]*total
    assigned = np.zeros_like(targets)
    records = []
    for group in groups:
        v = vector(group)
        delta = (((assigned+v-targets)**2 - (assigned-targets)**2) / np.maximum(targets, 1)).sum(1)
        partition = int(delta.argmin())
        assigned[partition] += v
        group_id = hashlib.sha256('\n'.join(group).encode()).hexdigest()[:16]
        for filename in group:
            records.append({'file_name': filename, 'group_id': group_id,
                            'partition': 'holdout' if partition == 0 else 'development',
                            'fold': None if partition == 0 else partition-1,
                            'annotator_image_ids': dataset.by_filename[filename]})
    manifest = {'schema_version': 1, 'seed': seed, 'annotation_sha256': dataset.sha256,
                'max_gap_hours': max_gap_hours, 'development_folds': 3,
                'pixel_audit_sha256': pixel_audit_sha256,
                'near_duplicate_review': 'provisional_until_registered_image_similarity_review',
                'records': sorted(records, key=lambda r: r['file_name'])}
    validate_splits(manifest, dataset, duplicate_groups)
    return manifest


def validate_splits(manifest, dataset, duplicate_groups=()):
    if manifest['annotation_sha256'] != dataset.sha256:
        raise ValueError('Annotation provenance differs.')
    records = manifest['records']
    by_name = {r['file_name']: r for r in records}
    if len(by_name) != len(records) or set(by_name) != set(dataset.filenames()):
        raise ValueError('Split must cover each original image exactly once.')
    for name, record in by_name.items():
        if record['annotator_image_ids'] != dataset.by_filename[name]:
            raise ValueError('Annotator records have been separated or dropped.')
        if record['partition'] not in {'holdout', 'development'}:
            raise ValueError('Unknown partition.')
        if ((record['partition'] == 'holdout' and record['fold'] is not None)
                or (record['partition'] == 'development' and record['fold'] not in range(3))):
            raise ValueError('Invalid fold assignment.')
    for group in components(dataset.filenames(), manifest['max_gap_hours'], duplicate_groups):
        assignments = {(by_name[n]['partition'], by_name[n]['fold']) for n in group}
        if len(assignments) != 1:
            raise ValueError('Temporal/exact-duplicate group crosses split boundaries.')
    # Also honor explicit groups, including future manual near-duplicate additions.
    explicit = defaultdict(set)
    for r in records:
        explicit[r['group_id']].add((r['partition'], r['fold']))
    if any(len(v) != 1 for v in explicit.values()):
        raise ValueError('Explicit group crosses partitions.')


def save_splits(path, manifest):
    path = Path(path)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)+'\n'
    if path.exists() and path.read_text() != payload:
        raise FileExistsError('Existing split is immutable; choose an explicitly versioned path.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def select_filenames(manifest, split, fold=0):
    if split not in {'train', 'val', 'holdout'} or fold not in range(3):
        raise ValueError('Invalid split or fold.')
    return [r['file_name'] for r in manifest['records']
            if (r['partition'] == 'holdout' if split == 'holdout' else
                r['partition'] == 'development' and ((r['fold'] == fold) == (split == 'val')))]


def finalize_reviewed_splits(manifest, dataset, review, audit_sha256):
    """Freeze existing assignments after a provenance-bound, bounded review.

    Never promote previously used development images into holdout. If newly
    confirmed related images cross partitions, fail and require a separately
    designed split migration instead of silently reassigning them.
    """
    if (review.get('status') != 'review_closed_for_v02_split_with_bounded_scope'
            or review.get('unresolved_cases') != []
            or not review.get('reviewed_pairs')):
        raise ValueError('Near-duplicate review is incomplete.')
    if review.get('audit_sha256') != audit_sha256:
        raise ValueError('Review does not match current registered audit hash.')
    groups = review.get('additional_duplicate_groups')
    if not isinstance(groups, list):
        raise ValueError('Reviewed duplicate groups must be explicit.')
    names = set(dataset.filenames())
    for pair in review['reviewed_pairs']:
        if pair.get('a') not in names or pair.get('b') not in names or not pair.get('decision'):
            raise ValueError('Review contains an invalid pair or missing decision.')
    validate_splits(manifest, dataset, groups)
    final = deepcopy(manifest)
    final['schema_version'] = 2
    final['near_duplicate_review'] = 'closed_bounded_registered_review_v02'
    final['registered_audit_sha256'] = audit_sha256
    final['reviewed_duplicate_groups'] = groups
    final['assignments_preserved_from_provisional'] = True
    return final
