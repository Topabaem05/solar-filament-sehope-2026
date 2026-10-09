"""Replay a provenance-checked native checkpoint on complete images, without labels."""
import argparse
import time
from pathlib import Path

import torch

from filament.data.tiles import load_native_image
from filament.io.rle import read_submission, write_submission
from filament.native_inference import predict_native
from filament.training.checkpoint import load_native_checkpoint
from filament.training.runtime import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'source-root', 'weights', 'split', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--official-root', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Existing prediction preserved; choose a new output.')
    if len({p.stem for p in args.images}) != len(args.images):
        raise ValueError('Duplicate input image IDs.')
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    backbone, decoder, checkpoint = load_native_checkpoint(
        args.checkpoint, args.source_root, args.weights, device=args.device, split_path=args.split,
        official_root=args.official_root)
    predictions, diagnostics = {}, {}
    shape = tuple(checkpoint['training_config']['expected_shape'])
    start = time.monotonic()
    with torch.no_grad():
        for path in args.images:
            pixels = load_native_image(path, shape)
            rles, diagnostic = predict_native(
                pixels, lambda x: decoder(backbone(x.to(args.device))), **checkpoint['inference_config'])
            predictions[path.stem] = rles
            diagnostics[path.name] = diagnostic
    write_submission(args.output, predictions, predictions, shape)
    if read_submission(args.output, predictions, shape) != predictions:
        raise AssertionError('Replayed CSV differs on readback.')
    atomic_json(args.output.with_suffix('.replay.json'), dict(
        checkpoint_sha256=sha256(args.checkpoint), split_sha256=sha256(args.split),
        image_sha256={p.name: sha256(p) for p in args.images}, csv_sha256=sha256(args.output),
        diagnostics=diagnostics, device=args.device, seconds=time.monotonic()-start,
        annotation_input=False, submission_ready=False))


if __name__ == '__main__':
    main()
