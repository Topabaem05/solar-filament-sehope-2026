"""Single-host run ownership, deadlines and a bounded frozen-feature cache."""
import fcntl
import hashlib
import json
import os
import socket
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


@contextmanager
def training_lock(path, run_id):
    """Keep inode stable; deleting a flock file can allow two lock owners."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another local training job holds the project lock.') from exc
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({'pid': os.getpid(), 'host': socket.gethostname(), 'run_id': run_id,
                       'started_unix': time.time()}, handle)
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TimeBudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        if not 0 < seconds <= 3300:
            raise ValueError('Pilot runtime must be positive and at most3300 seconds.')
        self.clock = clock
        self.start = clock()
        self.deadline = self.start + seconds

    @property
    def remaining(self):
        return self.deadline - self.clock()

    def check(self, reserve=0):
        if self.remaining <= reserve:
            raise TimeBudgetExceeded('Pilot time budget reached; no further work started.')


class FeatureCache:
    """In-memory only; never reused across backbone or preprocessing versions."""
    def __init__(self, provenance_key, max_originals=2, max_bytes=512 * 2**20):
        if not 1 <= max_originals <= 2 or max_bytes <= 0:
            raise ValueError('Feature cache must be bounded to at most two originals.')
        self.provenance_key = provenance_key
        self.max_originals, self.max_bytes = max_originals, max_bytes
        self.groups = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = self.peak_bytes = 0

    def get(self, image_key, tile_key, loader, *, provenance_key):
        if provenance_key != self.provenance_key:
            raise ValueError('Frozen feature cache provenance changed.')
        group = self.groups.get(image_key)
        if group is not None and tile_key in group:
            self.groups.move_to_end(image_key)
            self.hits += 1
            return group[tile_key]
        self.misses += 1
        features = tuple(f.detach().cpu().contiguous() for f in loader())
        size = sum(f.numel() * f.element_size() for f in features)
        if size > self.max_bytes:
            raise MemoryError('A feature entry exceeds the cache budget.')
        while self.groups and (self.bytes + size > self.max_bytes
                               or (image_key not in self.groups and len(self.groups) >= self.max_originals)):
            _, removed = self.groups.popitem(last=False)
            self.bytes -= sum(f.numel() * f.element_size() for fs in removed.values() for f in fs)
            self.evictions += 1
        self.groups.setdefault(image_key, {})[tile_key] = features
        self.groups.move_to_end(image_key)
        self.bytes += size
        self.peak_bytes = max(self.bytes, self.peak_bytes)
        return features

    def clear(self):
        self.groups.clear()
        self.bytes = 0


def implementation_hashes():
    package = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob('*.py'))}
