"""Pixel-level secondary gate for localized PatchCore anomalies.

The image-level PatchCore score is a weighted patch maximum, so a small
defect only nudges that maximum slightly and can stay below the image
threshold even though a clearly abnormal, spatially compact region is
present on the anomaly map. This module implements the secondary decision
rule: fail the part when the map contains a connected region that is
anomalous and compact.

Regions larger than ``max_area_px`` are deliberately ignored: a map-wide
elevation is the image-level gate's job, and treating it here would reject
uniformly noisy good parts. The map this operates on is the sigma=4
smoothed 256x256 anomaly map PatchCoreService already produces, so one
pixel here is one pixel of the model input, not of the camera frame.
"""

import cv2
import numpy as np

from utils.image_utils import unwrap_anomaly_map


def largest_region(anomaly_map, pixel_threshold):
    """Return ``(area_px, mask)`` of the largest region above a threshold.

    Display-oriented helper. Unusable input (``None``, empty, or non-finite
    values) yields ``(0, None)`` rather than an error.
    """
    amap = unwrap_anomaly_map(anomaly_map)
    if amap is None or amap.size == 0 or not np.isfinite(amap).all():
        return 0, None

    mask = (amap > pixel_threshold).astype(np.uint8)
    if not mask.any():
        return 0, None

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    return area, mask.astype(bool)


def evaluate_pixel_gate(anomaly_map, pixel_threshold, min_area_px, max_area_px):
    """Apply the localized-anomaly rule to one anomaly map.

    Returns ``(triggered, matched_area_px, mask)``. The gate triggers when
    any connected region above ``pixel_threshold`` has an area between
    ``min_area_px`` and ``max_area_px`` inclusive (a compact anomaly); the
    largest qualifying region is reported. Regions outside the bounds —
    specks below ``min_area_px`` and map-wide elevations above
    ``max_area_px`` — never trigger.

    A map with non-finite values cannot be evaluated, so this raises
    ``ValueError`` instead of returning "not triggered": the verdict caller
    treats a gate that cannot be computed as a failure, never as a pass.
    """
    amap = unwrap_anomaly_map(anomaly_map)
    if amap is None:
        return False, 0, None

    if amap.size == 0:
        return False, 0, None

    if not np.isfinite(amap).all():
        raise ValueError("anomaly map contains non-finite values")

    mask = (amap > pixel_threshold).astype(np.uint8)
    if not mask.any():
        return False, 0, None

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    triggered = False
    matched_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area_px and area > matched_area:
            triggered = True
            matched_area = area

    return triggered, matched_area, mask.astype(bool)
