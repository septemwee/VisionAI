"""Shared PASS/FAIL verdict logic for live inspection and trainer review.

Both the live inspection worker (main.py) and the trainer review page
(services/inference_service.py) must reach the same verdict for the same
prediction: a single implementation here prevents the two paths from
drifting — e.g. one path gaining a safety rule the other lacks.

The verdict is fail-closed: a non-finite score, a missing anomaly map or a
gate that cannot be evaluated always yields FAIL, never PASS.
"""

import math
import numpy as np

from services.patchcore_service import SCORE_SCALE
from services.recipe_service import normalize_pixel_gate
from utils.pixel_gate import evaluate_pixel_gate, unwrap_anomaly_map


def evaluate_verdict(score, anomaly_map, threshold, recipe):
    """Return ``(is_defect, reason, gate_region_px)`` for one prediction.

    ``score`` is the 0-1 PatchCore fraction, ``anomaly_map`` the raw model
    map (any wrapping; ``None`` fails closed), ``threshold`` the normalized
    0-1 recipe threshold and ``recipe`` the full recipe dict supplying the
    pixel-gate configuration.
    """
    if not math.isfinite(score):
        # A NaN score compares False against any threshold, which would
        # silently pass the part; an unevaluable prediction must fail.
        return True, f"Invalid PatchCore score {score!r}", 0

    if not math.isfinite(threshold) or threshold <= 0:
        return True, "Invalid anomaly threshold", 0
    amap = unwrap_anomaly_map(anomaly_map)
    if amap is None or amap.ndim != 2 or not amap.size or not np.isfinite(amap).all():
        return True, "Inspection unavailable: invalid anomaly map", 0

    is_defect = score > threshold
    reason = f"Score: {score:.4f} (threshold: {threshold:.4f})"
    gate_region_px = 0

    if not is_defect:
        # The image-level score is a weighted patch maximum: a small defect
        # raises it only slightly and can stay below the limit. The pixel
        # gate adds a second rule on the anomaly map so a compact anomalous
        # region still fails the part.
        gate = normalize_pixel_gate(recipe)

        # Legacy recipes were calibrated for image scores only. A local rule
        # needs an explicit version and a matching calibration record.
        local_policy = recipe.get("verdict_policy") == "image_and_pixel_v1"
        if local_policy:
            calibration = recipe.get("calibration") or {}
            if (calibration.get("pixel_gate") != gate
                    or calibration.get("proposed") != threshold):
                return True, "Inspection unavailable: pixel gate requires calibration", 0
        if local_policy and gate["enabled"]:
            pixel_threshold = gate["threshold_ratio"] * threshold * SCORE_SCALE

            if anomaly_map is None:
                # PatchCore always returns a map; reaching here means the
                # prediction pipeline is degraded, so fail closed.
                return True, "Pixel gate unavailable (no anomaly map)", 0

            try:
                triggered, area, _ = evaluate_pixel_gate(
                    anomaly_map,
                    pixel_threshold,
                    gate["min_area_px"],
                    gate["max_area_px"],
                )
            except ValueError as error:
                return True, f"Pixel gate error: {error}", 0

            if triggered:
                return (
                    True,
                    f"Pixel region {area}px "
                    f"({gate['min_area_px']}-{gate['max_area_px']}px gate, "
                    f"thr {pixel_threshold:.1f})",
                    area,
                )

    return is_defect, reason, gate_region_px
