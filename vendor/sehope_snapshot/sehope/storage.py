"""Verified single-file NAS transfers; never remove an unverified local copy."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

GIB = 1024**3


def fingerprint(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        if sys.platform == 'darwin':
            fcntl.fcntl(f.fileno(), 48, 1)  # F_NOCACHE; not a server durability guarantee.
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_mount(mount, source):
    mount = Path(mount)
    expected = f'{source} on {mount} (smbfs,'
    lines = subprocess.check_output(['mount'], text=True).splitlines()
    if not any(line.startswith(expected) for line in lines):
        raise RuntimeError('Expected KPI NAS SMB mount is absent or changed')
    if not mount.is_mount():
        raise RuntimeError('NAS path is not a mounted filesystem')
    return mount.stat().st_dev


def inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe relative storage path')
    path = root / rel
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Storage path escapes root or is a symlink')
    return path


def check_space(root, incoming, reserve, limit=None):
    root = Path(root)
    if incoming < 0 or reserve < 0 or (limit is not None and limit <= 0):
        raise ValueError('Invalid storage budget')
    existing = 0
    if limit is not None and root.exists():
        for p in root.rglob('*'):
            if p.is_symlink():
                raise ValueError('Symlink inside managed staging directory')
            if p.is_file():
                existing += p.stat().st_size
    if limit is not None and existing + incoming > limit:
        raise RuntimeError('Managed local staging limit would be exceeded')
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < incoming + reserve:
        raise RuntimeError('Disk free-space reserve would be breached')


def verified_copy(source, destination, expected_sha, guard=lambda: None):
    source, destination = Path(source), Path(destination)
    guard()
    before = fingerprint(source)
    if sha256_file(source) != expected_sha:
        raise ValueError('Source SHA256 mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha:
            raise ValueError('Existing destination SHA256 mismatch; preserve both copies')
        return
    partial = destination.with_name(destination.name + '.partial')
    # A failed previous transfer is retained for diagnosis, not overwritten.
    with source.open('rb') as src, partial.open('xb') as dst:
        while block := src.read(1024 * 1024):
            guard()
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    guard()
    if fingerprint(source) != before or sha256_file(partial) != expected_sha:
        raise ValueError('Transfer read-back or source-stability check failed')
    os.rename(partial, destination)
    guard()
    if sha256_file(destination) != expected_sha:
        raise ValueError('Final destination SHA256 mismatch')


def evict_verified(local, archive, expected_sha, guard=lambda: None):
    local, archive = Path(local), Path(archive)
    guard()
    before = fingerprint(local)
    if sha256_file(local) != expected_sha or sha256_file(archive) != expected_sha:
        raise ValueError('Cannot evict: local or NAS SHA256 mismatch')
    guard()
    if fingerprint(local) != before:
        raise ValueError('Cannot evict: local file changed during verification')
    local.unlink()
