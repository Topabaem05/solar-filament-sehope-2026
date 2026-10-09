"""Frozen, unmodified official VisionHOPE CUDA backend, with explicit provenance.

This is a separate execution contract from portable_v1. Cross-backend numerical
equivalence has NOT passed; checkpoints cannot silently switch implementations.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

BACKEND = 'official_cuda_v1'
MANIFEST_SHA256 = '39edb9b7f43a5b87cba9e608716dbe8e8cf6c6e932b3b3e0610518142735fa03'
WEIGHTS_SHA256 = 'ada1efa033bac3652ec956674e1337e19f4f3897b6284f25834d43fe850b0c39'


def verify_official_source(source_root):
    root = Path(source_root).resolve()
    manifest_path = root.parent / 'SNAPSHOT.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Official source manifest differs from pinned snapshot.')
    manifest = json.loads(manifest_path.read_text())
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Official source cannot contain symlinks.')
        if path.is_file() and '__pycache__' not in path.parts:
            actual[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest['files']:
        raise ValueError('Official source files differ from pinned snapshot.')
    return manifest


def validate_fusion_policy(policy):
    if policy:
        raise ValueError('Only the unmodified official default fusion policy is supported.')


class OfficialCudaPyramid(nn.Module):
    def __init__(self, source_root, weights, *, device='cuda:0'):
        super().__init__()
        device = torch.device(device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Official CUDA backend requires CUDA; CPU fallback is not permitted.')
        if torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.4':
            raise ValueError('Official pilot requires pinned torch2.6.0 CUDA12.4 runtime.')
        source_root = Path(source_root).resolve()
        manifest = verify_official_source(source_root)
        if hashlib.sha256(Path(weights).read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError('Official backbone weight SHA256 differs.')
        sys.path.insert(0, str(source_root.parent))
        try:
            module = importlib.import_module('visionhope.models')
        finally:
            sys.path.remove(str(source_root.parent))
        if Path(module.__file__).resolve() != source_root / 'models/__init__.py':
            raise ValueError('Another official source is already imported.')
        self.model = module.create_model('visionhope_tiny', checkpoint_path=str(weights))
        expected = torch.load(weights, map_location='cpu', weights_only=True)['state_dict']
        actual = self.model.state_dict()
        if actual.keys() != expected.keys() or not all(torch.equal(actual[k], expected[k]) for k in actual):
            raise ValueError('Loaded official state differs from pretrained weights.')
        self.requires_grad_(False)
        self.eval()
        self.to(device)
        self.provenance = dict(backend=BACKEND, source_root=str(source_root),
            source_sha256=manifest['files'], weights_sha256=WEIGHTS_SHA256,
            snapshot_sha256=MANIFEST_SHA256, upstream_commit=manifest['commit'],
            execution_policy='official_default_fp32_frozen_no_offsets',
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device),
                         capability=list(torch.cuda.get_device_capability(device))))

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image, offsets=None):
        if offsets is not None:
            raise ValueError('Official S1 backend does not implement Content offsets.')
        if image.device.type != 'cuda' or image.dtype != torch.float32:
            raise ValueError('Official S1 backend requires CUDA FP32 input.')
        if image.ndim != 4 or image.shape[1] != 3 or any(n % 32 for n in image.shape[-2:]):
            raise ValueError('Expected normalized [B,3,H,W] with dimensions divisible by32.')
        return self.model.forward_intermediates(image)
