"""Select an explicit backend; never substitute a numerically different one."""
from filament.models.sehope_pyramid import SEHopePyramid

PORTABLE = 'portable_v1'
OFFICIAL = 'official_cuda_v1'


def backend_name(config):
    name = config.get('backbone_backend', PORTABLE)
    if name not in {PORTABLE, OFFICIAL}:
        raise ValueError('Unknown backbone backend.')
    return name


def validate_backend_provenance(config, provenance, device=None):
    name = backend_name(config)
    if provenance.get('backend', PORTABLE) != name:
        raise ValueError('Checkpoint backbone backend differs from training config.')
    if name == OFFICIAL:
        if provenance.get('execution_policy') != 'official_default_fp32_frozen_no_offsets':
            raise ValueError('Unknown official execution policy.')
        if device is not None and not str(device).startswith('cuda'):
            raise ValueError('Official checkpoint requires CUDA; no portable/CPU fallback.')


def create_backbone(config, source_root, weights, *, device, official_root=None):
    if backend_name(config) == OFFICIAL:
        if official_root is None:
            raise ValueError('Official source root must be explicit.')
        from filament.models.official_cuda_pyramid import OfficialCudaPyramid
        return OfficialCudaPyramid(official_root, weights, device=device)
    model = SEHopePyramid(source_root, weights)
    model.provenance['backend'] = PORTABLE
    return model.to(device)
