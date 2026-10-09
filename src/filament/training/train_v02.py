"""S1 pilot: frozen SE-HOPE, existing instance head, native-grid whole-image eval."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from filament.data.coco_adapter import AnnotationSet
from filament.data.splits import select_filenames, validate_splits
from filament.data.tiles import Tile, load_native_image, native_grid, normalized_tile
from filament.evaluation.pq import evaluate
from filament.io.rle import read_submission, write_submission
from filament.models.instance_decoder import InstanceDecoder
from filament.models.loss import InstanceLoss
from filament.models.backbone_factory import create_backbone, backend_name, OFFICIAL
from filament.native_inference import NATIVE_PREPROCESS_VERSION, predict_native
from filament.training.checkpoint import VIEW, validate_checkpoint
from filament.training.native_data import NativeTrainingData, choose_subset
from filament.training.runtime import (Budget, FeatureCache, TimeBudgetExceeded, atomic_json,
                                       implementation_hashes, sha256, training_lock)


def validate_config(config):
    backend = backend_name(config)
    if backend == OFFICIAL and config['mode'] != 'free_gpu_pilot':
        raise ValueError('Official backend only supports an explicit CUDA pilot.')
    if config['mode'] not in {'free_gpu_pilot', 'cpu_contract_fixture'}:
        raise ValueError('Only bounded pilot/fixture modes are supported.')
    if not 0 < config['max_seconds'] <= 3300 or not 0 < config['train_seconds'] < config['max_seconds']:
        raise ValueError('Invalid pilot time bounds.')
    if not 1 <= config['steps'] <= 1024 or not 1 <= config['grad_accumulation'] <= 8:
        raise ValueError('Pilot update limits exceeded.')
    if not 1 <= config['original_block_updates'] <= config['steps']:
        raise ValueError('Invalid original sampling block length.')
    if not math.isfinite(config['learning_rate']) or not 0 < config['learning_rate'] <= .01:
        raise ValueError('Invalid learning rate.')
    if not 0 < config['evaluation_reserve_seconds'] < config['max_seconds']:
        raise ValueError('Whole-image evaluation needs a positive bounded reserve.')
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for name, value in config['thresholds'].items() if name != 'min_shared_pixels'):
        raise ValueError('Invalid probability or IoU threshold.')
    if config['thresholds']['min_shared_pixels'] < 1:
        raise ValueError('Overlap association requires shared foreground pixels.')
    if config['tile_size'] < 32 or config['tile_size'] % 32 or not 0 < config['stride'] <= config['tile_size']:
        raise ValueError('Invalid native tile geometry.')
    if not 1 <= config['count_train'] <= 16 or not 1 <= config['count_val'] <= 4:
        raise ValueError('Pilot image counts exceed the bounded development scope.')
    if config['mode'] == 'free_gpu_pilot' and config['expected_shape'] != [2048, 2048]:
        raise ValueError('Competition pilot must evaluate complete native originals.')
    if config['mode'] == 'cpu_contract_fixture' and (config['steps'] > 8 or max(config['expected_shape']) > 128):
        raise ValueError('Local contract fixture is restricted to tiny synthetic data.')


def device_preflight(backbone, decoder_config, device, tile_size, budget, *, pixels=None, inference_config=None):
    """FP32 backbone parity and finite 256->512 head backward before real training."""
    official = backbone.provenance.get('backend') == OFFICIAL
    report = {'device': str(device), 'srnl_dtype': 'float32', 'amp': False,
              'backend': backbone.provenance.get('backend', 'portable_v1'),
              'admission_contract': 'official_same_backend_runtime_v1' if official else 'portable_cpu_cuda_parity_v1',
              'cross_backend_equivalence_claimed': False}
    if official and (pixels is None or inference_config is None):
        raise ValueError('Official preflight requires actual training pixels and whole-grid settings.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; do not silently fall back to CPU.')
        free, total = torch.cuda.mem_get_info(device)
        if free < 5 * 2**30:
            raise MemoryError('Less than5 GiB free GPU memory before pilot.')
        torch.cuda.set_per_process_memory_fraction(.8, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report.update(gpu_name=torch.cuda.get_device_name(device), free_bytes_before=free,
                      total_bytes=total, allocator_fraction=.8)
        torch.cuda.reset_peak_memory_stats(device)
    sizes = sorted({min(256, tile_size), tile_size})
    measurements = []
    for size in sizes:
        budget.check()
        if official:
            y, x0 = ((n-size)//2 for n in pixels.shape)
            x = normalized_tile(pixels, Tile(0, y, x0, size, size))
        else:
            x = torch.linspace(-2, 2, 3 * size * size).reshape(1, 3, size, size)
        cpu_features = None
        if device.type == 'cuda' and size == sizes[0] and not official:
            backbone.cpu()
            with torch.no_grad():
                cpu_features = tuple(f.clone() for f in backbone(x))
        backbone.to(device)
        start = time.monotonic()
        with torch.no_grad():
            features = backbone(x.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_seconds = time.monotonic() - start
        if not all(torch.isfinite(f).all() for f in features):
            raise FloatingPointError('Nonfinite backbone features.')
        repeat_error = None
        if official:
            expected_shapes = [(1,c,size//s,size//s) for c,s in zip((64,128,256,512),(4,8,16,32))]
            if [tuple(f.shape) for f in features] != expected_shapes:
                raise ValueError('Official feature pyramid geometry differs.')
            with torch.no_grad():
                repeated = backbone(x.to(device))
            if not all(torch.equal(a,b) for a,b in zip(features,repeated)):
                raise RuntimeError('Same-backend repeated features differ.')
            repeat_error = 0.0
            del repeated
        if cpu_features is not None:
            errors = [float((a - b.cpu()).abs().max()) for a, b in zip(cpu_features, features)]
            report['cpu_cuda_max_abs_errors'] = errors
            report['cpu_cuda_tolerance'] = {'rtol': .001, 'atol': .0001}
            if not all(torch.allclose(a, b.cpu(), rtol=.001, atol=.0001) for a, b in zip(cpu_features, features)):
                raise RuntimeError(f'CPU/CUDA feature parity failed: {errors}')
        probe = InstanceDecoder(**decoder_config).to(device)
        target = torch.zeros(1, size, size, device=device)
        target[:, size // 3:size // 3 + max(2, size // 32), size // 4:3 * size // 4] = 1
        loss, _ = InstanceLoss(balanced_mask_bce=True)(probe(features), [target])
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters()):
            raise FloatingPointError('Device preflight loss/gradient failure.')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        measurements.append({'size': size, 'backbone_seconds': forward_seconds,
                             'same_backend_repeat_max_abs': repeat_error,
                             'forward_backward_seconds': time.monotonic() - start,
                             'probe_loss': float(loss.detach()),
                             'cuda_peak_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
        del probe, target, features, cpu_features, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['measurements'] = measurements
    if official:
        budget.check(reserve=1200)
        probe = InstanceDecoder(**decoder_config).to(device).eval()
        start = time.monotonic()
        with torch.no_grad():
            _, diagnostic = predict_native(pixels, lambda x: probe(backbone(x.to(device))), **inference_config)
        expected_tiles = len(native_grid(pixels.shape, tile_size, inference_config['stride']))
        if diagnostic['tile_count'] != expected_tiles or diagnostic['coverage_minimum'] != 1:
            raise RuntimeError('Preflight whole-image coverage failure.')
        report['untrained_whole_grid_runtime'] = dict(seconds=time.monotonic()-start,
            expected_tiles=expected_tiles, diagnostic=diagnostic, accuracy_measurement=False)
        del probe
        gc.collect()
        torch.cuda.empty_cache()
    report['passed'] = True
    return report


def run(args, config, budget):
    output = args.output
    dataset = AnnotationSet(args.annotations)
    manifest = json.loads(args.split.read_text())
    validate_splits(manifest, dataset, manifest.get('reviewed_duplicate_groups', []))
    if config.get('annotation_sha256') and dataset.sha256 != config['annotation_sha256']:
        raise ValueError('Annotation file SHA differs from pinned pilot data.')
    if config.get('split_sha256') and sha256(args.split) != config['split_sha256']:
        raise ValueError('Final split SHA differs from pinned pilot split.')
    names, val_names = choose_subset(manifest, fold=config['fold'], count_train=config['count_train'],
                                    count_val=config['count_val'], seed=config['seed'])
    for name in names + val_names:
        if dataset.shape(dataset.by_filename[name][0]) != tuple(config['expected_shape']):
            raise ValueError('Dataset dimensions differ from pilot config.')
    data = NativeTrainingData(dataset, manifest, args.images, names, fold=config['fold'],
                              tile_size=config['tile_size'], stride=config['stride'])
    device = torch.device('cuda:0' if config['mode'] == 'free_gpu_pilot' else 'cpu')
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    rng = random.Random(config['seed'])
    if psutil.virtual_memory().available < 3 * 2**30:
        raise MemoryError('Less than3 GiB available host RAM.')
    if psutil.disk_usage(output).free < 2 * 2**30:
        raise OSError('Less than2 GiB free output storage.')
    inference_config = dict(tile_size=config['tile_size'], stride=config['stride'], **config['thresholds'])
    backbone = create_backbone(config, args.source_root, args.weights, device=device,
                               official_root=getattr(args, 'official_root', None))
    preflight_pixels = load_native_image(args.images/names[0], tuple(config['expected_shape'])) if backend_name(config) == OFFICIAL else None
    preflight = device_preflight(backbone, config['decoder'], device, config['tile_size'], budget,
                                 pixels=preflight_pixels, inference_config=inference_config)
    preflight['training_probe_image'] = names[0] if preflight_pixels is not None else None
    atomic_json(output / 'device_preflight.json', preflight)
    print(json.dumps({'event': 'device_preflight_passed', **preflight}), flush=True)
    torch.manual_seed(config['seed'])
    decoder = InstanceDecoder(**config['decoder']).to(device)
    criterion = InstanceLoss(balanced_mask_bce=config['balanced_mask_bce'])
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config['learning_rate'], weight_decay=.01)
    provenance_key = json.dumps({'backbone': backbone.provenance,
                                 'preprocessing': NATIVE_PREPROCESS_VERSION}, sort_keys=True)
    cache = FeatureCache(provenance_key)
    config_sha = sha256(args.config)
    code_hashes = implementation_hashes()
    image_sha = {n: sha256(args.images / n) for n in names + val_names}
    atomic_json(output / 'run_inputs.json', {'config': config, 'config_sha256': config_sha,
                'split_sha256': sha256(args.split), 'annotation_sha256': dataset.sha256,
                'training_images': names, 'validation_images': val_names, 'image_sha256': image_sha,
                'implementation_sha256': code_hashes, 'backbone_provenance': backbone.provenance,
                'versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'numpy', 'Pillow', 'scipy', 'pycocotools', 'psutil')},
                'sampler': 'uniform originals in shuffled blocks, then uniform annotator; positive50percent/uniform50percent; hard-negative mining not enabled in S1'})

    def feature(image_key, tile, tensor):
        budget.check()
        with torch.no_grad():
            values = cache.get(image_key, (tile.y, tile.x, *tile.shape),
                               lambda: backbone(tensor.to(device)), provenance_key=provenance_key)
        return tuple(v.to(device) for v in values)

    history, sampled_records = [], {}
    optimizer_steps = 0

    def save_checkpoint():
        checkpoint = {'view': VIEW, 'preprocessing': NATIVE_PREPROCESS_VERSION,
                      'decoder_config': config['decoder'],
                      'decoder': {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                      'optimizer': optimizer.state_dict(), 'steps': optimizer_steps,
                      'training_config': config, 'config_sha256': config_sha,
                      'inference_config': inference_config,
                      'backbone_provenance': backbone.provenance,
                      'annotation_sha256': dataset.sha256, 'split_sha256': sha256(args.split),
                      'train_images': names, 'validation_images': val_names,
                      'implementation_sha256': code_hashes, 'image_sha256': image_sha,
                      'scope': config['mode'], 'production_ready': False}
        temporary = output / 'checkpoint.tmp.pt'
        torch.save(checkpoint, temporary)
        os.replace(temporary, output / 'checkpoint.pt')

    train_start = time.monotonic()
    order = []
    try:
        for step in range(config['steps']):
            # Reserve time for a full whole-image evaluation; platform/bootstrap
            # impose an independent hard timeout as well.
            if time.monotonic() - train_start >= config['train_seconds']:
                break
            budget.check(reserve=config['evaluation_reserve_seconds'])
            if step % config['original_block_updates'] == 0:
                if not order:
                    order = names.copy()
                    rng.shuffle(order)
                name = order.pop()
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.
            for _ in range(config['grad_accumulation']):
                budget.check(reserve=config['evaluation_reserve_seconds'])
                original, record, tile, targets, positive_request = data.sample(name, rng)
                features = feature(original['pixel_sha256'], tile, normalized_tile(original['pixels'], tile))
                target = torch.from_numpy(targets.masks).float().to(device)
                loss, _ = criterion(decoder(features), [target])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                (loss / config['grad_accumulation']).backward()
                step_loss += float(loss.detach()) / config['grad_accumulation']
                stats = sampled_records.setdefault(record, {'patches': 0, 'positive': 0, 'truncated_targets': 0})
                stats['patches'] += 1
                stats['positive'] += int(len(targets.masks) > 0)
                stats['truncated_targets'] += int(targets.truncated.sum())
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()):
                raise FloatingPointError('Missing or nonfinite decoder gradient.')
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            optimizer_steps += 1
            history.append(step_loss)
            if optimizer_steps % 16 == 0:
                save_checkpoint()
                progress = {'stage': 'training', 'steps': optimizer_steps, 'loss': step_loss,
                            'remaining_seconds': budget.remaining, 'cache_bytes': cache.bytes}
                atomic_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    except TimeBudgetExceeded:
        # Save the last completed update; discard a partial accumulated gradient.
        optimizer.zero_grad(set_to_none=True)
    save_checkpoint()
    if optimizer_steps == 0:
        raise RuntimeError('No optimizer updates completed; do not call the pilot trained.')
    decoder.eval()
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, code_hashes=code_hashes, split_sha256=sha256(args.split))
    reloaded = InstanceDecoder(**config['decoder']).to(device).eval()
    reloaded.load_state_dict(checkpoint['decoder'], strict=True)
    original = data.original(names[0])
    tile = original['tiles'][0]
    features = feature(original['pixel_sha256'], tile, normalized_tile(original['pixels'], tile))
    with torch.no_grad():
        before, after = decoder(features), reloaded(features)
    reload_error = max(float((before[k] - after[k]).abs().max()) for k in before)
    if reload_error != 0:
        raise AssertionError('Checkpoint readback predictions are not exact.')
    del decoder, optimizer, before, after, features, checkpoint
    decoder = reloaded
    data.cache.clear()
    cache.clear()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    evaluations = {}
    for label, selected in [('train', names), ('validation', val_names)]:
        predictions, diagnostics = {}, {}
        for name in selected:
            budget.check()
            pixels = load_native_image(args.images / name, tuple(config['expected_shape']))
            tiles = native_grid(pixels.shape, config['tile_size'], config['stride'])
            image_key = __import__('hashlib').sha256(pixels.tobytes()).hexdigest()
            cursor = 0

            def predictor(tensor):
                nonlocal cursor
                current = tiles[cursor]
                cursor += 1
                return decoder(feature(image_key, current, tensor))

            start = time.monotonic()
            rles, diagnostic = predict_native(pixels, predictor, **inference_config)
            if cursor != len(tiles):
                raise AssertionError('Whole-image inference skipped tiles.')
            predictions[Path(name).stem] = rles
            diagnostic['seconds'] = time.monotonic() - start
            diagnostics[name] = diagnostic
            atomic_json(output / 'progress.json', {'stage': f'{label}_evaluation', 'image': name,
                        'completed_images': len(predictions), 'total_images': len(selected),
                        'instances': len(rles), 'remaining_seconds': budget.remaining})
            print(json.dumps({'event': 'whole_image_evaluated', 'split': label, 'image': name,
                              'tiles': cursor, 'instances': len(rles), 'seconds': diagnostic['seconds']}), flush=True)
        csv = output / f'{label}_predictions.csv'
        write_submission(csv, predictions, predictions, tuple(config['expected_shape']))
        restored = read_submission(csv, predictions, tuple(config['expected_shape']))
        if restored != predictions:
            raise AssertionError('Prediction CSV readback differs.')
        scores = evaluate(dataset, restored, selected)
        atomic_json(output / f'{label}_scores.json', scores)
        atomic_json(output / f'{label}_diagnostics.json', diagnostics)
        evaluations[label] = {'official': scores['official'], 'one_to_one': scores['one_to_one_diagnostic'],
                              'images': len(selected), 'instances': sum(map(len, predictions.values())),
                              'csv_sha256': sha256(csv), 'seconds': sum(d['seconds'] for d in diagnostics.values())}
    report = {'status': 'completed_whole_image_pilot', 'scope': config['mode'], 'steps': optimizer_steps,
              'steps_requested': config['steps'], 'whole_image_oracle_crops': False,
              'train_images': names, 'validation_images': val_names,
              'evaluations': evaluations, 'training_loss_history': history,
              'sampled_annotator_records': sampled_records,
              'checkpoint_readback_max_abs_error': reload_error, 'checkpoint_sha256': sha256(output / 'checkpoint.pt'),
              'split_sha256': sha256(args.split), 'config_sha256': config_sha,
              'cache': {k: getattr(cache, k) for k in ('hits', 'misses', 'evictions', 'peak_bytes')},
              'device_preflight': preflight,
              'cuda_peak_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'wall_seconds': time.monotonic() - budget.start,
              'rss_bytes_at_end': psutil.Process().memory_info().rss,
              'production_ready': False, 'kaggle_submission_performed': False,
              'screening_only': True, 'holdout_used': False}
    atomic_json(output / 'report.json', report)
    atomic_json(output / 'progress.json', {'stage': 'complete', 'steps': optimizer_steps,
                'train_pq': evaluations['train']['official']['pq'],
                'validation_pq': evaluations['validation']['official']['pq']})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'annotations', 'images', 'split', 'source-root', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--official-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Existing run preserved; use a new run directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    budget = Budget(config['max_seconds'])
    with training_lock(args.output.parent / '.native_training.lock', args.output.name):
        def alarm(_signum, _frame):
            raise TimeBudgetExceeded('Hard wall-clock alarm reached.')
        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, config['max_seconds'])
        try:
            atomic_json(args.output / 'progress.json', {'stage': 'starting', 'mode': config['mode']})
            result = run(args, config, budget)
            print(json.dumps({k: v for k, v in result.items() if k not in
                              ('training_loss_history', 'sampled_annotator_records')}, indent=2), flush=True)
        except Exception as exc:
            atomic_json(args.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                        'wall_seconds': time.monotonic() - budget.start,
                        'checkpoint_exists': (args.output / 'checkpoint.pt').exists(),
                        'completed': False})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
