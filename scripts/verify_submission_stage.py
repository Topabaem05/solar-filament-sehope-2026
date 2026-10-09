"""Read-only local verification/re-scoring of completed remote inference stages."""
import argparse
import json
from pathlib import Path

import numpy as np
from filament.data.coco_adapter import AnnotationSet
from filament.data.splits import select_filenames
from filament.evaluation.pq import evaluate
from filament.io.rle import read_submission
from filament.training.runtime import sha256
from analyze_native_pilot import error_slices


def group_interval(scores, split, filenames):
    allowed = set(filenames)
    groups = {}
    record_group = {a: r['group_id'] for r in split['records'] if r['file_name'] in allowed
                    for a in r['annotator_image_ids']}
    for record in scores['per_record']:
        group = record_group[record['annotator_image_id']]
        values = groups.setdefault(group, np.zeros(2))
        values += [record['matched_iou_sum'], record['denominator']]
    rows = np.array(list(groups.values()))
    rng = np.random.default_rng(314159)
    samples = rows[rng.integers(0, len(rows), size=(2000, len(rows)))].sum(1)
    pq = samples[:, 0] / np.maximum(samples[:, 1], 1e-12)
    return {'method': '2000 percentile bootstrap resamples of registered temporal groups',
            'metric': 'public-reference-arithmetic PQ', 'groups': len(groups), 'seed': 314159,
            'level': 0.95, 'lower': float(np.quantile(pq, .025)), 'upper': float(np.quantile(pq, .975)),
            'limitation': 'Conditional on fixed model/split; does not include model-selection uncertainty or unmodeled observation correlation.'}


def verify(run, annotations, split_path, registration_path):
    run = Path(run)
    report = json.loads((run / 'report.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    if (report['status'] != 'completed' or not report['fresh_session_replay_exact']
            or report['optimizer_updates'] != 0):
        raise ValueError('Incomplete inference stage.')
    expected_phases = {'development': {'development'}, 'final': {'holdout', 'submission'}}
    if (plan['phase'] not in expected_phases or report['phase'] != plan['phase']
            or set(report.get('results', {})) != expected_phases[plan['phase']]):
        raise ValueError('Required inference results are missing or unexpected.')
    for path, expected in [(run / 'plan.json', report['plan_sha256']),
                           (run / 'checkpoint.pt', plan['checkpoint_sha256']),
                           (split_path, plan['split_sha256']), (annotations, plan['annotation_sha256']),
                           (run / 'replay.csv', plan['replay_csv_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Artifact digest differs: {Path(path).name}')
    split = json.loads(Path(split_path).read_text())
    dataset = AnnotationSet(annotations)
    registration = json.loads(Path(registration_path).read_text())
    results = {}
    for label, remote in report['results'].items():
        key = {'development': 'development_images', 'holdout': 'holdout_images', 'submission': 'test_images'}[label]
        names = sorted(plan[key])
        if label != 'submission':
            expected = select_filenames(split, 'val' if label == 'development' else 'holdout', 0)
            if set(names) != set(expected):
                raise ValueError('Full split inventory differs.')
        elif len(names) != 180:
            raise ValueError('Test inventory must contain exactly 180 images.')
        csv = run / (label + '.csv')
        if remote['annotation_input'] or not remote['full_image_coverage']:
            raise ValueError('Invalid inference contract.')
        if sha256(csv) != remote['csv_sha256']:
            raise ValueError('CSV differs from completed remote report.')
        predictions = read_submission(csv, [Path(n).stem for n in names])
        diagnostics = json.loads((run / (label + '.diagnostics.json')).read_text())
        if set(diagnostics) != set(names) or remote['images'] != len(names):
            raise ValueError('Diagnostic coverage differs.')
        for name, diag in diagnostics.items():
            if (diag['coverage_minimum'] < 1 or diag['tile_count'] != 25
                    or diag['shape'] != [2048, 2048]
                    or diag['output_instances'] != len(predictions[Path(name).stem])):
                raise ValueError('Invalid full native-grid diagnostic.')
        result = {'original_images': len(names), 'prediction_instances': sum(map(len, predictions.values())),
                  'zero_detection_images': sum(not masks for masks in predictions.values()),
                  'csv_sha256': sha256(csv), 'csv_bytes': csv.stat().st_size,
                  'all_images_accounted_for': True, 'rle_roundtrip_valid': True,
                  'no_exact_duplicate_instance_masks': True}
        if label != 'submission':
            scores = evaluate(dataset, predictions, names)
            # Normalize histogram keys without depending on JSON member order.
            normalize = lambda value: json.loads(json.dumps(value))
            if any(normalize(scores[key]) != remote['scores'][key]
                   for key in ('official', 'one_to_one_diagnostic')):
                raise ValueError('Independent local metric disagrees with remote score.')
            result.update(official=scores['official'], strict=scores['one_to_one_diagnostic'],
                          pq_interval=group_interval(scores, split, names),
                          errors=error_slices(dataset, predictions, names, registration=registration))
        elif csv.stat().st_size > 10_000_000:
            raise ValueError('CSV exceeds the organizer form 10MB limit.')
        results[label] = result
    return {'status': 'verified', 'phase': plan['phase'], 'stage_report_sha256': sha256(run / 'report.json'),
            'checkpoint_sha256': plan['checkpoint_sha256'], 'fresh_session_replay_exact': True,
            'inference_uses_annotations': False, 'results': results, 'kaggle_server_acceptance': 'not_yet_verified'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'annotations', 'split', 'registration', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.annotations, args.split, args.registration)
    if args.output.exists():
        raise FileExistsError('Existing verification report preserved.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    for label, record in result['results'].items():
        print(label, json.dumps({k: v for k, v in record.items() if k != 'errors'}, indent=2))
