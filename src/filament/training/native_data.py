"""Uniform original/annotator sampling with native-grid targets; no oracle val."""
import hashlib
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np

from filament.data.splits import select_filenames, validate_splits
from filament.data.tiles import crop_targets, load_native_image, native_grid


def choose_subset(manifest, *, fold, count_train, count_val, seed):
    rng = random.Random(seed)
    return (rng.sample(select_filenames(manifest, 'train', fold), count_train),
            rng.sample(select_filenames(manifest, 'val', fold), count_val))


class NativeTrainingData:
    def __init__(self, dataset, manifest, images, names, *, fold=0, tile_size=512, stride=384):
        validate_splits(manifest, dataset, manifest.get('reviewed_duplicate_groups', []))
        if manifest.get('near_duplicate_review') != 'closed_bounded_registered_review_v02':
            raise ValueError('Native training requires the finalized reviewed split.')
        if not names or len(set(names)) != len(names) or not set(names) <= set(select_filenames(manifest, 'train', fold)):
            raise ValueError('Training names include duplicate, validation or holdout images.')
        self.dataset, self.images, self.names = dataset, Path(images), list(names)
        self.tile_size, self.stride = tile_size, stride
        self.cache = OrderedDict()

    def original(self, name):
        if name not in self.names:
            raise ValueError('Cannot load a validation/holdout image into the training sampler.')
        if name not in self.cache:
            # Decode only one original at a time; release all annotator masks on
            # eviction. RLEs in AnnotationSet remain compressed and small.
            self.cache.clear()
            pixels = load_native_image(self.images / name, self.dataset.shape(self.dataset.by_filename[name][0]))
            tiles = native_grid(pixels.shape, self.tile_size, self.stride)
            targets = {}
            for record in self.dataset.by_filename[name]:
                masks = self.dataset.masks(record)
                ids = [a['id'] for a in self.dataset.annotations[record]]
                targets[record] = [crop_targets(masks, ids, tile) for tile in tiles]
            self.cache[name] = {'pixels': pixels, 'tiles': tiles, 'targets': targets,
                                'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
        return self.cache[name]

    def sample(self, name, rng, positive_fraction=.5):
        if not 0 <= positive_fraction <= 1:
            raise ValueError('Invalid positive sampler fraction.')
        original = self.original(name)
        record = rng.choice(self.dataset.by_filename[name])
        positives = [i for i, targets in enumerate(original['targets'][record]) if len(targets.masks)]
        positive_request = rng.random() < positive_fraction
        eligible = positives if positive_request and positives else range(len(original['tiles']))
        index = rng.choice(eligible)
        return original, record, original['tiles'][index], original['targets'][record][index], positive_request
