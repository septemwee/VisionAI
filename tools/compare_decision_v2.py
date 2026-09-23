"""Calibrate Decision v2, freeze it, then compare on held-out data.

One PatchCore inference per sample supplies both decisions. Never writes
recipes or model artifacts. Synthetic alpha > 0.1 defines provisional GT.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from services.patchcore_service import PatchCoreService
from services.synthetic_defect_service import generate_synthetic_validation
from services.validation_service import audit_splits, artifact_identity, EXTENSIONS
from services.verdict_service import evaluate_verdict
from utils.decision_v2 import analyze_map, calibrate, decide
from utils.image_utils import heatmap_overlay, unwrap_anomaly_map
from utils.paths import resolve_recipe_path
from utils.pixel_gate import qualifying_region_mask


def score_split(service, root, role, temporary):
    good_dir = root / role / 'good'
    synth_dir, mask_dir = temporary / role / 'images', temporary / role / 'masks'
    manifest = generate_synthetic_validation(good_dir, synth_dir, mask_dir=mask_dir)
    samples = []
    paths = [(p, 'good') for p in sorted(good_dir.rglob('*')) if p.suffix.lower() in EXTENSIONS]
    paths += [(synth_dir / item['file'], item['type']) for item in manifest]
    for index, (path, kind) in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        started = time.perf_counter()
        score, raw = service.predict_full(image)
        elapsed = (time.perf_counter() - started) * 1000
        amap = unwrap_anomaly_map(raw).copy()
        gt = None
        if kind != 'good':
            alpha = cv2.imread(str(mask_dir / path.name), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(alpha, (amap.shape[1], amap.shape[0]),
                            interpolation=cv2.INTER_LINEAR) > 25.5
        samples.append(dict(name=path.name, kind=kind, role=role, score=score,
                            amap=amap, gt=gt, image=image, inference_ms=elapsed))
        if index % 20 == 0 or index == len(paths):
            print(f'{role}: {index}/{len(paths)}', flush=True)
    return samples


def overlap(mask, gt):
    intersection = int(np.logical_and(mask, gt).sum())
    union = int(np.logical_or(mask, gt).sum())
    return dict(iou=intersection / union if union else 0.,
                precision=intersection / int(mask.sum()) if mask.any() else 0.,
                recall=intersection / int(gt.sum()) if gt.any() else 0.)


def evaluate(samples, recipe, settings, out):
    rows = []
    gate = recipe['pixel_gate']
    threshold = recipe['anomaly_threshold']
    for sample in samples:
        raw, gt = sample['amap'], sample['gt']
        old_fail, old_reason, _ = evaluate_verdict(sample['score'], raw, threshold, recipe)
        started = time.perf_counter()
        old_mask, _ = qualifying_region_mask(raw, gate['threshold_ratio'] * threshold * 100,
                                             gate['min_area_px'], gate['max_area_px'])
        old_ms = (time.perf_counter() - started) * 1000
        if old_mask is None:
            old_mask = np.zeros(raw.shape, bool)
        started = time.perf_counter()
        analysis = analyze_map(raw, settings)
        new_fail = decide(sample['score'], threshold, analysis)
        new_ms = (time.perf_counter() - started) * 1000
        regions = analysis['regions']
        row = dict(image_name=sample['name'], role=sample['role'],
                   ground_truth='GOOD' if gt is None else 'SYNTHETIC_DEFECT',
                   defect_type=sample['kind'], image_score=sample['score'],
                   image_threshold=threshold, **settings,
                   strong_peak=max((r['peak'] for r in regions), default=None),
                   region_mean=max((r['mean'] for r in regions), default=None),
                   region_area=max((r['area'] for r in regions), default=0),
                   old_verdict='FAIL' if old_fail else 'PASS',
                   new_verdict='FAIL' if new_fail else 'PASS',
                   old_local_fail=bool(old_mask.any()), new_local_fail=analysis['local_fail'],
                   old_reason=old_reason, inference_ms=sample['inference_ms'],
                   old_segmentation_ms=old_ms, new_segmentation_ms=new_ms,
                   regions=regions)
        for key in ('iou', 'precision', 'recall'):
            row['old_' + key] = None
            row['new_' + key] = None
        if gt is not None:
            row.update({'old_' + k: v for k, v in overlap(old_mask, gt).items()})
            row.update({'new_' + k: v for k, v in overlap(analysis['mask'], gt).items()})
            row['raw_peak_in_defect'] = float(raw[gt].max()) if gt.any() else None
        else:
            row['raw_peak_in_defect'] = None
        rows.append(row)
        # Deterministic first 12 test defects, not cherry-picked successes.
        preview_index = sum(r['role'] == 'test' and r['ground_truth'] != 'GOOD' for r in rows)
        if sample['role'] == 'test' and gt is not None and preview_index <= 12:
            panels = []
            for title, amap, mask in [('V1', raw, old_mask),
                                      ('V2', analysis['processed_map'], analysis['mask'])]:
                panel = heatmap_overlay(sample['image'], amap)
                h, w = panel.shape[:2]
                selected = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                truth = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                for binary, color in ((selected, (0, 255, 255)), (truth, (255, 255, 255))):
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(panel, contours, -1, color, 2)
                panel = cv2.resize(panel, (512, 512))
                cv2.putText(panel, title + ' | yellow=segment white=GT', (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                panels.append(panel)
            if not cv2.imwrite(str(out / f'preview_{preview_index:02d}.png'), np.hstack(panels)):
                raise OSError('Could not write preview')
    return rows


def summarize(rows):
    summary = {}
    for role in ('calibration', 'test'):
        for kind in ('good', 'synthetic', 'scratch', 'spot', 'chip', 'pin_discoloration'):
            group = [r for r in rows if r['role'] == role and (
                r['defect_type'] != 'good' if kind == 'synthetic' else r['defect_type'] == kind)]
            summary[role + '/' + kind] = dict(count=len(group))
            for version in ('old', 'new'):
                summary[role + '/' + kind][version] = dict(
                    failed=sum(r[version + '_verdict'] == 'FAIL' for r in group),
                    local_failed=sum(r[version + '_local_fail'] for r in group),
                    mean_iou=float(np.mean([r[version + '_iou'] for r in group])) if kind != 'good' else None,
                    mean_recall=float(np.mean([r[version + '_recall'] for r in group])) if kind != 'good' else None,
                    mean_precision=float(np.mean([r[version + '_precision'] for r in group])) if kind != 'good' else None,
                    median_segmentation_ms=float(np.median([r[version + '_segmentation_ms'] for r in group])))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', default='recipes/TJA1041_SO14/recipe.json')
    parser.add_argument('--output', default='reports/decision_v2_20260923')
    args = parser.parse_args()
    recipe_path = Path(args.recipe)
    before = recipe_path.read_bytes()
    recipe = json.loads(before)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = resolve_recipe_path(recipe['prepared_dataset_path'])
    audit = audit_splits(root)
    identity = artifact_identity(recipe)
    torch.set_num_threads(4)
    service = PatchCoreService()
    service.load_model(recipe['model']['path'])
    with TemporaryDirectory(prefix='visionai_v2_') as temporary:
        calibration = score_split(service, root, 'calibration', Path(temporary))
        settings, metrics = calibrate(
            [s['amap'] for s in calibration if s['kind'] == 'good'],
            [s['amap'] for s in calibration if s['kind'] != 'good'])
        (out / 'candidate.json').write_text(json.dumps(dict(
            settings=settings, calibration=metrics, model_identity=identity,
            dataset_audit=audit, image_threshold=recipe['anomaly_threshold'],
            activated=False), indent=2), encoding='utf-8')
        print('Frozen calibration: ' + json.dumps(settings), flush=True)
        test = score_split(service, root, 'test', Path(temporary))
        rows = evaluate(calibration + test, recipe, settings, out)
    if recipe_path.read_bytes() != before or audit_splits(root) != audit:
        raise RuntimeError('Recipe or dataset changed during comparison')
    if artifact_identity(recipe) != identity:
        raise RuntimeError('Model artifacts changed during comparison')
    report = dict(activated=False, evidence='synthetic_only', settings=settings,
                  calibration=metrics, summary=summarize(rows), rows=rows,
                  scope='prepared crops, excludes capture/YOLO/orientation/marking/Qt',
                  ground_truth='generator alpha > 0.1, resized to map resolution',
                  model_identity=identity, dataset_audit=audit)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, regions=json.dumps(row['regions'])))
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
