"""Independently verify RLE disjointness, union preservation, ownership and PQ."""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from filament.data.coco_adapter import AnnotationSet
from filament.evaluation.pq import evaluate
from filament.io.rle import checked_rle, read_submission
from filament.training.runtime import sha256
from verify_submission_stage import group_interval


def assert_disjoint(counts, shape=(2048, 2048)):
    rles = [checked_rle(c, shape) for c in counts]
    if rles:
        pairwise = np.asarray(coco_mask.iou(rles, rles, [0] * len(rles)))
        np.fill_diagonal(pairwise, 0)
        if (pairwise > 0).any():
            raise ValueError('Overlapping instance pixels are forbidden.')


def verify_projection(original, projected, shape=(2048, 2048)):
    assert_disjoint(projected, shape)
    source = [checked_rle(c, shape) for c in original]
    target = [checked_rle(c, shape) for c in projected]
    union = None
    index = 0
    for rle in source:
        overlap = 0 if union is None else int(coco_mask.area(coco_mask.merge([rle, union], intersect=True)))
        remaining = int(coco_mask.area(rle)) - overlap
        if remaining:
            if index == len(target):
                raise ValueError('Missing residual foreground.')
            actual = target[index]
            area = int(coco_mask.area(actual))
            if (area != remaining or int(coco_mask.area(coco_mask.merge([rle, actual], intersect=True))) != area
                    or (union is not None and int(coco_mask.area(coco_mask.merge([union, actual], intersect=True))) != 0)):
                raise ValueError('Pixel ownership or foreground union differs.')
            index += 1
        union = rle if union is None else coco_mask.merge([union, rle])
    if index != len(target):
        raise ValueError('Unexpected extra instance.')


def verify(export_dir, development_run, final_run, development_verification, final_verification,
           lock_path, annotations, split_path):
    export_dir = Path(export_dir)
    report = json.loads((export_dir / 'export_report.json').read_text())
    lock = json.loads(Path(lock_path).read_text())
    base_dev = json.loads(Path(development_verification).read_text())
    base_final = json.loads(Path(final_verification).read_text())
    if (report['status'] != 'completed' or report['lock_sha256'] != sha256(lock_path)
            or report['script_sha256'] != sha256(Path(__file__).with_name('disjoint_export.py'))
            or base_dev['status'] != 'verified' or base_final['status'] != 'verified'):
        raise ValueError('Missing verified source or immutable export.')
    split = json.loads(Path(split_path).read_text())
    dataset = AnnotationSet(annotations)
    results = {}
    for label, run, base in [('development', development_run, base_dev), ('holdout', final_run, base_final), ('submission', final_run, base_final)]:
        run = Path(run)
        plan = json.loads((run / 'plan.json').read_text())
        if (sha256(run / 'report.json') != base['stage_report_sha256']
                or sha256(annotations) != plan['annotation_sha256'] or sha256(split_path) != plan['split_sha256']
                or plan['checkpoint_sha256'] != lock['checkpoint_sha256']):
            raise ValueError('Source provenance mismatch.')
        key = {'development': 'development_images', 'holdout': 'holdout_images', 'submission': 'test_images'}[label]
        names = sorted(plan[key])
        source_csv, csv = run / (label + '.csv'), export_dir / (label + '.csv')
        if (sha256(source_csv) != base['results'][label]['csv_sha256']
                or sha256(source_csv) != report['results'][label]['source_csv_sha256']
                or sha256(csv) != report['results'][label]['csv_sha256']):
            raise ValueError('CSV digest mismatch.')
        stems = [Path(n).stem for n in names]
        original, projected = read_submission(source_csv, stems), read_submission(csv, stems)
        for stem in stems:
            verify_projection(original[stem], projected[stem])
        result = {'original_images': len(names), 'prediction_instances': sum(map(len, projected.values())),
                  'zero_detection_images': sum(not c for c in projected.values()),
                  'csv_sha256': sha256(csv), 'csv_bytes': csv.stat().st_size,
                  'all_images_accounted_for': True, 'rle_roundtrip_valid': True,
                  'no_exact_duplicate_instance_masks': True, 'no_overlapping_instance_pixels': True,
                  'foreground_union_preserved': True, 'confidence_priority_ownership_verified': True}
        if label != 'submission':
            scores = evaluate(dataset, projected, names)
            result.update(official=scores['official'], strict=scores['one_to_one_diagnostic'],
                          pq_interval=group_interval(scores, split, names))
        if label == 'submission' and (len(names) != 180 or csv.stat().st_size > 10_000_000):
            raise ValueError('Invalid test count or file size.')
        results[label] = result
        print(json.dumps({'phase': label, 'verified': True, 'pq': result.get('official', {}).get('pq')}), flush=True)
    return {'status': 'verified', 'phase': 'final', 'checkpoint_sha256': lock['checkpoint_sha256'],
            'postprocessing': report['algorithm'], 'postprocessing_lock_sha256': sha256(lock_path),
            'export_report_sha256': sha256(export_dir / 'export_report.json'),
            'fresh_session_replay_exact': base_final['fresh_session_replay_exact'],
            'inference_uses_annotations': False, 'results': results,
            'holdout_caveat': 'Base holdout already observed; fixed format correction chosen for server constraint, without score tuning.',
            'kaggle_server_acceptance': 'not_yet_verified'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir', 'development-run', 'final-run', 'development-verification', 'final-verification',
                 'lock', 'annotations', 'split', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing verification preserved.')
    result = verify(args.export_dir, args.development_run, args.final_run,
                    args.development_verification, args.final_verification, args.lock, args.annotations, args.split)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
