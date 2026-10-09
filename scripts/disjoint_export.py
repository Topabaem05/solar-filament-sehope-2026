"""Enforce exclusive instance pixels without changing the model or foreground union.

Native CSV order is descending fused confidence (stable member-key tie break).
The first instance owns shared pixels. No labels, image locations, or scores from
an evaluation set enter this deterministic competition-format correction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from filament.io.rle import decode_mask, read_submission, write_submission
from filament.training.runtime import sha256


def exclusive_masks(counts_list, shape=(2048, 2048)):
    claimed = np.zeros(shape, dtype=bool)
    result = []
    changed = dropped = removed = 0
    for counts in counts_list:
        original = decode_mask(counts, shape).astype(bool)
        residual = original & ~claimed
        difference = int(np.count_nonzero(original & claimed))
        changed += difference > 0
        removed += difference
        claimed |= original
        if not residual.any():
            dropped += 1
            continue
        result.append(coco_mask.encode(np.asfortranarray(residual, dtype=np.uint8))['counts'].decode('ascii'))
    return result, {'input_instances': len(counts_list), 'output_instances': len(result),
                    'changed_instances': changed, 'empty_residual_instances_removed': dropped,
                    'duplicate_pixel_assignments_removed': removed,
                    'union_foreground_pixels': int(claimed.sum())}


def export(development_run, final_run, output, lock_path):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing export is preserved.')
    lock = json.loads(Path(lock_path).read_text())
    if lock['algorithm'] != 'confidence_order_first_owner_v1' or lock['parameters']:
        raise ValueError('Expected the fixed parameter-free format correction.')
    output.mkdir(parents=True)
    report = {'status': 'started', 'algorithm': lock['algorithm'],
              'lock_sha256': sha256(lock_path), 'script_sha256': sha256(__file__),
              'inference_uses_annotations': False, 'results': {}}
    for label, run in [('development', development_run), ('holdout', final_run), ('submission', final_run)]:
        run = Path(run)
        source = json.loads((run / 'report.json').read_text())
        plan = json.loads((run / 'plan.json').read_text())
        if (source['status'] != 'completed' or not source['fresh_session_replay_exact']
                or plan['checkpoint_sha256'] != lock['checkpoint_sha256']
                or sha256(run / 'plan.json') != source['plan_sha256']):
            raise ValueError('Incomplete or mismatched source stage.')
        csv = run / (label + '.csv')
        if sha256(csv) != source['results'][label]['csv_sha256']:
            raise ValueError('Source CSV differs from the completed stage.')
        key = {'development': 'development_images', 'holdout': 'holdout_images', 'submission': 'test_images'}[label]
        names = sorted(plan[key])
        original = read_submission(csv, [Path(n).stem for n in names])
        predictions, stats = {}, {}
        for name, masks in original.items():
            predictions[name], stats[name] = exclusive_masks(masks)
        target = output / (label + '.csv')
        write_submission(target, predictions, original)
        report['results'][label] = {'source_csv_sha256': sha256(csv), 'csv_sha256': sha256(target),
                                   'images': len(names), 'per_image': stats}
        (output / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'phase': label, 'images': len(names), 'output_instances': sum(map(len, predictions.values()))}), flush=True)
    report['status'] = 'completed'
    (output / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('development-run', 'final-run', 'output', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.development_run, args.final_run, args.output, args.lock)
