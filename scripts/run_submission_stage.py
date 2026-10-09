"""Frozen-checkpoint full-image development or final inference, never training.

The inference loop accepts pixels and checkpoint settings only. Annotations are
opened by the evaluator after predictions have been written and read back.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from filament.data.tiles import load_native_image
from filament.io.rle import read_submission, write_submission
from filament.native_inference import predict_native
from filament.training.checkpoint import load_native_checkpoint
from filament.training.runtime import atomic_json, sha256


def check_inputs(directory, expected):
    actual = {p.name: p for p in Path(directory).glob('*.jpeg')}
    if not set(expected) <= set(actual):
        raise ValueError('Missing registered image files.')
    for name, digest in expected.items():
        if sha256(actual[name]) != digest:
            raise ValueError(f'Image digest differs: {name}')
    return [actual[n] for n in sorted(expected)]


def predict_phase(paths, backbone, decoder, checkpoint, output, deadline, device='cuda:0'):
    """Only image paths enter this function; no labels or target locations."""
    shape = tuple(checkpoint['training_config']['expected_shape'])
    predictions, diagnostics, seconds = {}, {}, []
    for path in paths:
        if time.monotonic() + 20 >= deadline:
            raise TimeoutError('Stage deadline reached; partial files are not a submission.')
        start = time.monotonic()
        with torch.no_grad():
            counts, diagnostic = predict_native(
                load_native_image(path, shape),
                lambda x: decoder(backbone(x.to(device))), **checkpoint['inference_config'])
        if diagnostic['coverage_minimum'] < 1:
            raise ValueError('Incomplete full-image coverage.')
        predictions[path.stem] = counts
        diagnostics[path.name] = diagnostic
        seconds.append(time.monotonic() - start)
        atomic_json(output.parent / (output.stem + '_progress.json'), {
            'processed': len(predictions), 'expected': len(paths), 'last_image': path.name,
            'total_image_seconds': sum(seconds), 'complete': False})
        if len(predictions) % 20 == 0 or len(predictions) == len(paths):
            print(json.dumps({'stage': output.stem, 'processed': len(predictions),
                              'expected': len(paths), 'seconds': sum(seconds)}), flush=True)
    write_submission(output, predictions, predictions, shape)
    if read_submission(output, predictions, shape) != predictions:
        raise ValueError('CSV round trip differs.')
    atomic_json(output.with_suffix('.diagnostics.json'), diagnostics)
    return predictions, {'images': len(paths), 'csv_sha256': sha256(output),
                         'image_seconds': sum(seconds), 'annotation_input': False,
                         'full_image_coverage': True}


def score_phase(annotations, predictions, paths, output):
    from filament.data.coco_adapter import AnnotationSet
    from filament.evaluation.pq import evaluate
    scores = evaluate(AnnotationSet(annotations), predictions, [p.name for p in paths])
    atomic_json(output, scores)
    return {k: v for k, v in scores.items() if k != 'per_record'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'checkpoint', 'weights', 'source-root', 'official-root',
                 'split', 'train-images', 'test-images', 'annotations', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--max-seconds', type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing stage output is preserved.')
    args.output.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + args.max_seconds
    plan = json.loads(args.plan.read_text())
    report = {'status': 'starting', 'phase': plan['phase'], 'plan_sha256': sha256(args.plan),
              'checkpoint_sha256': sha256(args.checkpoint), 'optimizer_updates': 0,
              'competition_submission_performed': False}
    try:
        for path, key in ((args.checkpoint, 'checkpoint_sha256'), (args.split, 'split_sha256'),
                          (args.annotations, 'annotation_sha256')):
            if sha256(path) != plan[key]:
                raise ValueError(f'Plan provenance differs: {key}')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        backbone, decoder, checkpoint = load_native_checkpoint(
            args.checkpoint, args.source_root, args.weights, device='cuda:0',
            split_path=args.split, official_root=args.official_root)
        shutil.copyfile(args.checkpoint, args.output / 'checkpoint.pt')
        shutil.copyfile(args.plan, args.output / 'plan.json')
        replay_paths = check_inputs(args.train_images, plan['replay_images'])
        _, replay = predict_phase(replay_paths, backbone, decoder, checkpoint,
                                  args.output / 'replay.csv', deadline)
        if replay['csv_sha256'] != plan['replay_csv_sha256']:
            raise ValueError('Fresh GPU session replay differs from original pilot CSV.')
        report['fresh_session_replay_exact'] = True
        atomic_json(args.output / 'report.json', report)
        if plan['phase'] == 'development':
            phases = [('development', args.train_images, plan['development_images'])]
        elif plan['phase'] == 'final':
            if plan.get('selection_lock', {}).get('checkpoint_sha256') != report['checkpoint_sha256']:
                raise ValueError('Final stage requires a matching development selection lock.')
            phases = [('holdout', args.train_images, plan['holdout_images']),
                      ('submission', args.test_images, plan['test_images'])]
        else:
            raise ValueError('Unknown inference phase.')
        report['results'] = {}
        for label, directory, expected in phases:
            if label == 'submission' and {p.name for p in directory.glob('*.jpeg')} != set(expected):
                raise ValueError('Test image inventory differs from registered 180-image list.')
            paths = check_inputs(directory, expected)
            predictions, result = predict_phase(paths, backbone, decoder, checkpoint,
                                                 args.output / (label + '.csv'), deadline)
            if label != 'submission':
                result['scores'] = score_phase(args.annotations, predictions, paths,
                                               args.output / (label + '_scores.json'))
            report['results'][label] = result
            atomic_json(args.output / 'report.json', report)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic() - start
        atomic_json(args.output / 'report.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
