"""Experimental, stateless map decisions; not enabled for live recipes.

Heatmap consumers use processed_map; contours use mask. Both refer to the
same current-image representation. Thresholds are raw map distances.
"""
import math

import cv2
import numpy as np

from utils.image_utils import unwrap_anomaly_map


def process_map(anomaly_map, spatial_sigma=0.8):
    amap = unwrap_anomaly_map(anomaly_map)
    if (amap is None or amap.ndim != 2 or not amap.size
            or not np.isfinite(amap).all()):
        raise ValueError('Decision v2 requires a finite, nonempty 2D anomaly map')
    if not math.isfinite(spatial_sigma) or not 0 <= spatial_sigma <= 5:
        raise ValueError('Invalid spatial sigma')
    amap = np.asarray(amap, dtype=np.float32).copy()
    return (cv2.GaussianBlur(amap, (0, 0), spatial_sigma)
            if spatial_sigma else amap)


def _regions(amap, low, high):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (amap > low).astype(np.uint8), connectivity=8)
    strong = set(np.unique(labels[amap > high]).tolist()) - {0}
    regions = []
    for label in sorted(strong):
        x, y, width, height, area = (int(v) for v in stats[label])
        region_mask = labels[y:y + height, x:x + width] == label
        values = amap[y:y + height, x:x + width][region_mask]
        regions.append(dict(label=label, area=area, peak=float(values.max()),
                            mean=float(values.mean()), bbox=[x, y, width, height]))
    return labels, regions


def analyze_map(anomaly_map, settings):
    low = float(settings['pixel_low_threshold'])
    high = float(settings['pixel_high_threshold'])
    minimum = settings['min_region_area']
    if (not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high
            or isinstance(minimum, bool) or int(minimum) != minimum or minimum < 1):
        raise ValueError('Invalid Decision v2 thresholds or minimum area')
    amap = process_map(anomaly_map, settings['spatial_sigma'])
    labels, candidates = _regions(amap, low, high)
    regions = [region for region in candidates if region['area'] >= minimum]
    mask = np.isin(labels, [region['label'] for region in regions])
    return dict(processed_map=amap, mask=mask, regions=regions,
                local_fail=bool(regions))


def decide(score, image_threshold, analysis):
    if (not math.isfinite(score) or not math.isfinite(image_threshold)
            or image_threshold <= 0):
        raise ValueError('Invalid image score or threshold')
    return bool(score > image_threshold or analysis['local_fail'])


def calibrate(good_maps, synthetic_maps, spatial_sigma=0.8):
    """Select local thresholds using calibration maps ONLY, never test images.

    High candidates come from good-image peak statistics; low candidates
    from sampled good-map distributions. No dependence on image threshold.
    Maximize local synthetic recall under a zero calibration false-alarm
    constraint. This is provisional synthetic evidence, not production accuracy.
    """
    if not good_maps or not synthetic_maps:
        raise ValueError('Both good and synthetic calibration maps are required')
    good = [process_map(amap, spatial_sigma) for amap in good_maps]
    defects = [process_map(amap, spatial_sigma) for amap in synthetic_maps]
    peaks = np.array([amap.max() for amap in good], dtype=float)
    sample = np.concatenate([amap.ravel()[::max(1, amap.size // 4096)] for amap in good])
    highs = sorted(set(float(v) for v in np.quantile(peaks, [.80, .90, .95, 1.])))
    lows = sorted(set(float(v) for v in np.quantile(sample, [.90, .95, .99, .999])))
    best = None
    tried = 0
    for high in highs:
        for low in lows:
            if low > high:
                continue
            good_areas = [max((r['area'] for r in _regions(amap, low, high)[1]), default=0)
                          for amap in good]
            defect_areas = [max((r['area'] for r in _regions(amap, low, high)[1]), default=0)
                            for amap in defects]
            for minimum in sorted(set([1, 4, 16, 64, 256, max(good_areas) + 1])):
                tried += 1
                false_alarms = sum(area >= minimum for area in good_areas)
                if false_alarms:
                    continue
                hits = sum(area >= minimum for area in defect_areas)
                # On equal recall prefer higher low threshold (less spreading),
                # then higher high threshold and smaller minimum area.
                rank = (hits, low, high, -minimum)
                if best is None or rank > best[0]:
                    best = (rank, dict(pixel_low_threshold=low, pixel_high_threshold=high,
                                       min_region_area=minimum, spatial_sigma=spatial_sigma))
    if best is None:
        raise ValueError('No valid Decision v2 calibration candidate')
    return best[1], dict(good_false_alarms=0, good_total=len(good),
                         synthetic_detected=best[0][0], synthetic_total=len(defects),
                         candidates_evaluated=tried)
