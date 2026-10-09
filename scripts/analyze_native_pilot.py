"""Post-hoc whole-image error slices; annotations never select inference crops."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from filament.data.coco_adapter import AnnotationSet
from filament.data.splits import select_filenames, validate_splits
from filament.data.tiles import native_grid
from filament.evaluation.pq import evaluate, overlaps, summarize
from filament.io.rle import checked_rle, read_submission
from filament.training.runtime import atomic_json, sha256


def fit_area_registration(dataset, manifest, split_sha256, fold=0):
    """Fit thirds on all annotator instances in development TRAIN only.

    This weighting mirrors evaluation's repeated annotator targets, not the
    uniform-original training sampler. No validation or holdout mask is read.
    """
    validate_splits(manifest, dataset, manifest.get('reviewed_duplicate_groups', []))
    if manifest.get('near_duplicate_review') != 'closed_bounded_registered_review_v02':
        raise ValueError('Area registration requires the finalized reviewed split.')
    names = sorted(select_filenames(manifest, 'train', fold))
    rows, records = [], []
    for name in names:
        for record in dataset.by_filename[name]:
            records.append(record)
            rows.extend([name, record, annotation['id'], int(coco_mask.area(rle))]
                        for annotation, rle in zip(dataset.annotations[record], dataset.rles(record)))
    if not rows:
        raise ValueError('No development training instances for area quantiles.')
    areas = np.array([r[3] for r in rows], dtype=np.int64)
    lower, upper = map(float, np.quantile(areas, [1/3, 2/3], method='linear'))
    return {'schema_version': 1, 'scope': 'development_train_only', 'fold': fold,
            'annotation_sha256': dataset.sha256, 'split_sha256': split_sha256,
            'weighting': 'each annotator instance once; repeated originals across annotators',
            'training_sampler_weighting_is_different': True,
            'quantile_method': 'linear', 'quantile_probabilities': [1/3, 2/3],
            'thresholds': {'small_upper_exclusive': lower, 'large_lower_inclusive': upper},
            'bin_rule': 'small: area<q1; medium: q1<=area<q2; large: area>=q2; ties stay together',
            'training_images': names, 'annotator_image_ids': records,
            'original_images': len(names), 'annotator_records': len(records), 'instances': len(rows),
            'training_instance_area_sha256': hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest(),
            'training_bin_counts': {'small': int((areas < lower).sum()),
                                    'medium': int(((areas >= lower) & (areas < upper)).sum()),
                                    'large': int((areas >= upper).sum())},
            'validation_masks_used': False, 'holdout_masks_used': False, 'test_data_used': False}


def validate_area_registration(registration, dataset, manifest, split_sha256, fold=0):
    expected = fit_area_registration(dataset, manifest, split_sha256, fold)
    if any(registration.get(key) != value for key, value in expected.items()):
        raise ValueError('Area registration differs from development training data or provenance.')
    return registration


def error_slices(dataset, predictions, filenames, *, registration=None, tile_size=512, stride=384):
    """Use strict one-to-one IoU>.5 matches; count each annotator target separately."""
    slices = {k: {'targets': 0, 'matched': 0, 'best_iou_sum': 0.} for k in
              ('all', 'area_lt256', 'area_256_to4095', 'area_ge4096',
               'bbox_aspect_ge4', 'bbox_short_side_le8', 'bbox_long_side_gt512',
               'crosses_native_tile_edge')}
    if registration is not None:
        for key in ('registered_small', 'registered_medium', 'registered_large'):
            slices[key] = {'targets': 0, 'matched': 0, 'best_iou_sum': 0.}
    for values in slices.values():
        values['matched_iou_sum'] = 0.
    stations = {}
    rows, background_fp, total_fp = [], 0, 0
    for name in filenames:
        shape = dataset.shape(dataset.by_filename[name][0])
        grid = native_grid(shape, tile_size, stride)
        x_edges = {e for t in grid for e in (t.x, t.x+t.width) if 0 < e < shape[1]}
        y_edges = {e for t in grid for e in (t.y, t.y+t.height) if 0 < e < shape[0]}
        station = Path(name).stem[-2:]
        station_data = stations.setdefault(station, {'images': [], 'matrices': []})
        station_data['images'].append(name)
        pred = [checked_rle(r, shape) for r in predictions[Path(name).stem]]
        for record in dataset.by_filename[name]:
            gt = dataset.rles(record)
            matrix = overlaps(gt, pred)
            station_data['matrices'].append(matrix)
            hits = matrix > .5
            matches = np.zeros(len(gt), dtype=bool)
            matched_iou = np.zeros(len(gt), dtype=float)
            matched_pred = np.zeros(len(pred), dtype=bool)
            if min(matrix.shape):
                a, b = linear_sum_assignment(-(hits * (min(matrix.shape) + 1. + matrix)))
                valid = hits[a, b]
                matches[a[valid]] = True
                matched_iou[a[valid]] = matrix[a[valid], b[valid]]
                matched_pred[b[valid]] = True
            best = matrix.max(axis=1) if len(pred) else np.zeros(len(gt))
            pred_best = matrix.max(axis=0) if len(gt) else np.zeros(len(pred))
            total_fp += int((~matched_pred).sum())
            background_fp += int(((pred_best == 0) & ~matched_pred).sum())
            for index, rle in enumerate(gt):
                area = int(coco_mask.area(rle))
                x, y, width, height = map(float, coco_mask.toBbox(rle))
                labels = ['all', 'area_lt256' if area < 256 else 'area_256_to4095' if area < 4096 else 'area_ge4096']
                if registration is not None:
                    cuts = registration['thresholds']
                    labels.append('registered_small' if area < cuts['small_upper_exclusive'] else
                                  'registered_medium' if area < cuts['large_lower_inclusive'] else 'registered_large')
                if max(width, height) / max(1, min(width, height)) >= 4:
                    labels.append('bbox_aspect_ge4')
                if min(width, height) <= 8:
                    labels.append('bbox_short_side_le8')
                if max(width, height) > 512:
                    labels.append('bbox_long_side_gt512')
                if any(x < e < x + width for e in x_edges) or any(y < e < y + height for e in y_edges):
                    labels.append('crosses_native_tile_edge')
                for label in labels:
                    slices[label]['targets'] += 1
                    slices[label]['matched'] += int(matches[index])
                    slices[label]['best_iou_sum'] += float(best[index])
                    slices[label]['matched_iou_sum'] += float(matched_iou[index])
                rows.append({'image': name, 'annotator_image_id': record,
                             'annotation_id': dataset.annotations[record][index]['id'],
                             'area': area, 'bbox': [x, y, width, height],
                             'station': station, 'slices': labels,
                             'matched': bool(matches[index]), 'best_prediction_iou': float(best[index]),
                             'matched_iou': float(matched_iou[index])})
    for values in slices.values():
        count = values['targets']
        values['recall'] = values['matched'] / count if count else None
        values['mean_best_iou'] = values['best_iou_sum'] / count if count else None
        values['sq'] = values['matched_iou_sum'] / values['matched'] if values['matched'] else None
    station_scores = {}
    for station, data in sorted(stations.items()):
        strict = summarize(data['matrices'], one_to_one=True)
        targets = strict['tp'] + strict['fn']
        station_scores[station] = {'original_images': len(data['images']),
                                  'annotator_records': len(data['matrices']),
                                  'official': summarize(data['matrices']), 'one_to_one': strict,
                                  'recall': strict['tp'] / targets if targets else None}
    return {'scope': 'post-hoc diagnostic, repeated targets across annotators; descriptive slices overlap',
            'geometry_note': 'Axis-aligned bbox aspect/extent is not skeleton length or physical filament width. Tile edges use actual image and run geometry.',
            'slice_metric_note': 'Target recall/SQ only; no area-slice PQ because unmatched predictions have no unambiguous target area.',
            'tile_geometry': {'tile_size': tile_size, 'stride': stride}, 'stations': station_scores,
            'slices': slices, 'unmatched_predictions': total_fp,
            'unmatched_predictions_with_zero_gt_overlap': background_fp,
            'targets': rows}


def analyze_run(run, annotations, *, registration_path=None, split_path=None):
    run = Path(run)
    if (registration_path is None) != (split_path is None):
        raise ValueError('Supply both the area registration and its split manifest.')
    report = json.loads((run / 'report.json').read_text())
    if report['status'] != 'completed_whole_image_pilot' or report['whole_image_oracle_crops'] or report['holdout_used']:
        raise ValueError('Require a completed whole-image development run without oracle crops or holdout use.')
    inputs = json.loads((run / 'run_inputs.json').read_text())
    dataset = AnnotationSet(annotations)
    checkpoint_sha = sha256(run / 'checkpoint.pt')
    if checkpoint_sha != report['checkpoint_sha256']:
        raise ValueError('Checkpoint digest differs from runner report.')
    if inputs['annotation_sha256'] != dataset.sha256:
        raise ValueError('Annotation digest differs from training/evaluation data.')
    if any(inputs[key] != report[key] for key in ('split_sha256', 'config_sha256')):
        raise ValueError('Run report provenance differs from input record.')
    if (inputs['training_images'] != report['train_images'] or
            inputs['validation_images'] != report['validation_images'] or
            set(report['train_images']) & set(report['validation_images'])):
        raise ValueError('Run image selection differs or overlaps across partitions.')
    registration = None
    if registration_path is not None:
        manifest = json.loads(Path(split_path).read_text())
        split_sha = sha256(split_path)
        if split_sha != inputs['split_sha256']:
            raise ValueError('Registered split differs from run split.')
        registration = validate_area_registration(json.loads(Path(registration_path).read_text()),
            dataset, manifest, split_sha, inputs['config']['fold'])
        for key, split in [('train_images', 'train'), ('validation_images', 'val')]:
            if not set(report[key]) <= set(select_filenames(manifest, split, inputs['config']['fold'])):
                raise ValueError('Reported evaluation names cross the registered split.')
    elif report['scope'] != 'cpu_contract_fixture':
        raise ValueError('Real solar error analysis requires preregistered training-only area bins.')
    result = {'run_report_sha256': sha256(run / 'report.json'), 'scope': report['scope'],
              'checkpoint_sha256': checkpoint_sha, 'annotation_sha256': dataset.sha256,
              'area_registration_sha256': sha256(registration_path) if registration_path else None,
              'analysis_source_sha256': sha256(Path(__file__)), 'splits': {},
              'production_ready': False, 'holdout_used': False}
    for label, key in [('train', 'train_images'), ('validation', 'validation_images')]:
        names = report[key]
        shape = dataset.shape(dataset.by_filename[names[0]][0])
        csv = run / f'{label}_predictions.csv'
        if sha256(csv) != report['evaluations'][label]['csv_sha256']:
            raise ValueError('Prediction file differs from runner report.')
        predictions = read_submission(csv, [Path(n).stem for n in names], shape)
        scores = evaluate(dataset, predictions, names)
        for actual, recorded in [('official', 'official'), ('one_to_one_diagnostic', 'one_to_one')]:
            if json.loads(json.dumps(scores[actual])) != report['evaluations'][label][recorded]:
                raise AssertionError('Independent PQ/count readback differs.')
        result['splits'][label] = {'official': scores['official'],
            **error_slices(dataset, predictions, names, registration=registration,
                           tile_size=inputs['config']['tile_size'], stride=inputs['config']['stride'])}
        diagnostics = json.loads((run / f'{label}_diagnostics.json').read_text())
        filters = {k: sum(tile[k] for d in diagnostics.values() for tile in d['per_tile'])
                   for k in ('queries', 'below_object', 'empty_mask', 'below_score', 'local_nms_removed', 'retained')}
        result['splits'][label]['filter_counts'] = filters
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--area-registration', type=Path)
    parser.add_argument('--split', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing analysis; choose a new output.')
    result = analyze_run(args.run, args.annotations, registration_path=args.area_registration, split_path=args.split)
    atomic_json(args.output, result)
    print(json.dumps({label: value['slices'] for label, value in result['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
