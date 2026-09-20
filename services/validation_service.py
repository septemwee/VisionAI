"""Artifact identity and independent dataset validation for Trainer."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from utils.paths import resolve_recipe_path

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def artifact_identity(recipe):
    digest = hashlib.sha256()
    root = resolve_recipe_path(recipe['model']['path'])
    for name in ('patchcore.pt', 'memory_bank.pt', 'metadata.json'):
        digest.update(name.encode())
        with (root / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    digest.update(json.dumps({key: recipe.get(key) for key in
        ('roi_default', 'roi_statistics', 'preprocessing_version')}, sort_keys=True).encode())
    return digest.hexdigest()


def audit_splits(root):
    """Reject empty splits and identical decoded images crossing split boundaries."""
    root = Path(root)
    seen = {}
    digest = hashlib.sha256()
    counts = {}
    for split in ('train', 'calibration', 'test'):
        paths = sorted(p for p in (root / split / 'good').rglob('*')
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f'{split}/good has no images. Prepare independent data for all three splits.')
        counts[split] = len(paths)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Unreadable image: {path}')
            key = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if key in seen and seen[key] != split:
                raise ValueError(f'Duplicate image crosses {seen[key]} and {split}: {path.name}')
            seen[key] = split
            digest.update(str(path.relative_to(root)).encode() + key.encode())
    return {'counts': counts, 'sha256': digest.hexdigest()}


def invalidate_validation(recipe):
    recipe.pop('calibration', None)
    recipe.pop('acceptance_report', None)
    recipe['validated'] = False
    steps = (recipe.get('assistant_state') or {}).get('steps', {})
    for step in ('S6', 'S7'):
        steps.pop(step, None)


def summarize_test(scores, threshold):
    import math
    values = [float(score) for _, score in scores]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Independent test scores are missing or invalid.')
    failed = sum(value > threshold for value in values)
    return {'count': len(values), 'above_threshold': failed,
            'rate': failed / len(values)}


def calibrate_pixel_gate(good_details, defect_details, image_threshold):
    """Choose a compact-anomaly gate from held-out maps without test leakage."""
    from services.patchcore_service import SCORE_SCALE
    from utils.pixel_gate import evaluate_pixel_gate
    good_maps = [np.asarray(entry[2]) for entry in good_details]
    defect_maps = [np.asarray(entry[2]) for entry in defect_details]
    if not good_maps or not defect_maps:
        raise ValueError("Good and synthetic anomaly maps are required.")
    max_good = int(np.floor(len(good_maps) * 0.01))
    candidates = []
    for ratio in (0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00):
        threshold = ratio * float(image_threshold) * SCORE_SCALE
        for minimum in (1, 4, 8, 12, 16, 24, 32, 48, 64):
            for maximum in (64, 128, 256, 512, 1024, 4096, 65536):
                if maximum < minimum:
                    continue
                good_hits = sum(evaluate_pixel_gate(amap, threshold, minimum, maximum)[0]
                                for amap in good_maps)
                if good_hits > max_good:
                    continue
                defect_hits = sum(evaluate_pixel_gate(amap, threshold, minimum, maximum)[0]
                                  for amap in defect_maps)
                candidates.append((defect_hits, -good_hits, ratio, minimum, maximum))
    if not candidates:
        raise ValueError("No pixel gate satisfies the held-out good-image limit.")
    detected, negative_good, ratio, minimum, maximum = max(candidates)
    return {
        "enabled": True, "threshold_ratio": ratio,
        "min_area_px": minimum, "max_area_px": maximum,
        "synthetic_detected": detected, "synthetic_total": len(defect_maps),
        "synthetic_detection_rate": detected / len(defect_maps),
        "expected_false_alarms": -negative_good,
    }


def summarize_pixel_test(details, image_threshold, gate):
    from services.patchcore_service import SCORE_SCALE
    from utils.pixel_gate import evaluate_pixel_gate
    threshold = gate["threshold_ratio"] * image_threshold * SCORE_SCALE
    hits = sum(evaluate_pixel_gate(entry[2], threshold, gate["min_area_px"], gate["max_area_px"])[0]
               for entry in details)
    return {"count": len(details), "triggered": hits, "rate": hits / len(details)}


def check_acceptance(recipe):
    report = recipe.get('acceptance_report') or {}
    if not report:
        raise ValueError('Run independent validation before finalizing this recipe.')
    if not (recipe.get('model') or {}).get('trained'):
        raise ValueError('Train the current dataset first.')
    if report.get('model_identity') != artifact_identity(recipe):
        raise ValueError('The model or crop settings changed. Validate again.')
    if report.get('dataset_audit') != audit_splits(resolve_recipe_path(recipe['prepared_dataset_path'])):
        raise ValueError('The dataset changed. Validate again.')
    threshold = recipe.get('anomaly_threshold')
    if report.get('threshold') != threshold or (recipe.get('calibration') or {}).get('proposed') != threshold:
        raise ValueError('The threshold changed. Validate again.')
    if report['good']['rate'] > 0.01:
        raise ValueError('Independent good-image false alarms exceed the 1% target.')
    if report['synthetic']['rate'] < 0.8:
        raise ValueError('Independent synthetic detection is below the provisional 80% target.')
    pixel = report.get("pixel_gate") or {}
    if not pixel or pixel["good"]["rate"] > 0.01 or pixel["synthetic"]["rate"] < 0.8:
        raise ValueError("Independent pixel-gate validation did not meet the provisional targets.")
    return report
