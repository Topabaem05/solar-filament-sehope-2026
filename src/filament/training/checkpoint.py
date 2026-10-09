"""Native checkpoint loading rejects oracle crops and stale preprocessing/code."""
from pathlib import Path

import torch

from filament.models.instance_decoder import InstanceDecoder
from filament.models.backbone_factory import create_backbone, validate_backend_provenance, backend_name, OFFICIAL
from filament.native_inference import NATIVE_PREPROCESS_VERSION
from filament.training.runtime import implementation_hashes, sha256

VIEW = 'native_grid_v02_s1'


def validate_checkpoint(checkpoint, *, code_hashes, split_sha256=None):
    if checkpoint.get('view') != VIEW:
        raise ValueError('Only native-grid training checkpoints are accepted; oracle crops are rejected.')
    if checkpoint.get('preprocessing') != NATIVE_PREPROCESS_VERSION:
        raise ValueError('Native preprocessing version differs.')
    if checkpoint.get('implementation_sha256') != code_hashes:
        raise ValueError('Implementation differs from checkpoint; use its saved source snapshot.')
    if split_sha256 is not None and checkpoint.get('split_sha256') != split_sha256:
        raise ValueError('Split provenance differs.')
    config = checkpoint['inference_config']
    if config['tile_size'] != checkpoint['training_config']['tile_size']:
        raise ValueError('Training and inference native scales differ.')
    validate_backend_provenance(checkpoint['training_config'], checkpoint.get('backbone_provenance', {}))


def load_native_checkpoint(path, source_root, weights, *, device='cpu', split_path=None, official_root=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, code_hashes=implementation_hashes(),
                        split_sha256=sha256(split_path) if split_path else None)
    validate_backend_provenance(checkpoint['training_config'], checkpoint['backbone_provenance'], device)
    backbone = create_backbone(checkpoint['training_config'], source_root, weights,
                               device=device, official_root=official_root)
    keys = ['source_sha256', 'weights_sha256']
    if backend_name(checkpoint['training_config']) == OFFICIAL:
        keys += ['backend', 'execution_policy', 'snapshot_sha256', 'runtime']
    for key in keys:
        if backbone.provenance[key] != checkpoint['backbone_provenance'][key]:
            raise ValueError(f'Backbone {key} differs from training.')
    decoder = InstanceDecoder(**checkpoint['decoder_config']).to(device).eval()
    decoder.load_state_dict(checkpoint['decoder'], strict=True)
    return backbone, decoder, checkpoint
