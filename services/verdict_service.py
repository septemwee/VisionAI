"""Shared PASS/FAIL verdict logic for live inspection and trainer review.

Both the live inspection worker (main.py) and the trainer review page
(services/inference_service.py) must reach the same verdict for the same
prediction: a single implementation here prevents the two paths from
drifting — e.g. one path gaining a safety rule the other lacks.

Invalid inspection data raises InspectionError and never receives a quality verdict.
"""

import math
import numpy as np

from services.patchcore_service import SCORE_SCALE
from services.recipe_service import normalize_pixel_gate
from utils.pixel_gate import evaluate_pixel_gate, unwrap_anomaly_map


class InspectionError(ValueError):
    """Inspection failed before a quality verdict was available."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def evaluate_verdict(score, anomaly_map, threshold, recipe):
    """Return ``(is_defect, reason, gate_region_px)`` for one prediction.

    ``score`` is the 0-1 PatchCore fraction, ``anomaly_map`` the raw model
    map (any wrapping; ``None`` fails closed), ``threshold`` the normalized
    0-1 recipe threshold and ``recipe`` the full recipe dict supplying the
    pixel-gate configuration.
    """
    try:
        valid_score = math.isfinite(score)
    except (TypeError, ValueError):
        valid_score = False
    if not valid_score:
        raise InspectionError("INVALID_SCORE", f"Invalid PatchCore score {score!r}")

    try:
        valid_threshold = math.isfinite(threshold) and threshold > 0
    except (TypeError, ValueError):
        valid_threshold = False
    if not valid_threshold:
        raise InspectionError("INVALID_RECIPE", "Invalid anomaly threshold")
    amap = unwrap_anomaly_map(anomaly_map)
    if amap is None or amap.ndim != 2 or not amap.size or not np.isfinite(amap).all():
        raise InspectionError("INVALID_ANOMALY_MAP", "Missing, empty, or non-finite anomaly map")

    local_policy = recipe.get("verdict_policy") == "image_and_pixel_v1"
    if local_policy:
        block = recipe.get("pixel_gate")
        if (not isinstance(block, dict)
                or not isinstance(block.get("enabled"), bool)
                or not all(key in block for key in ("threshold_ratio", "min_area_px", "max_area_px"))):
            raise InspectionError("INVALID_RECIPE", "Pixel gate configuration is missing")
        gate = normalize_pixel_gate(recipe)
        if gate != block:
            raise InspectionError("INVALID_RECIPE", "Pixel gate configuration is invalid")
        calibration = recipe.get("calibration") or {}
        if (calibration.get("pixel_gate") != gate
                or calibration.get("proposed") != threshold):
            raise InspectionError("INVALID_RECIPE", "Pixel gate requires matching calibration")

    is_defect = score > threshold
    reason = f"Score: {score:.4f} (threshold: {threshold:.4f})"
    gate_region_px = 0

    if not is_defect:
        # The image-level score is a weighted patch maximum: a small defect
        # raises it only slightly and can stay below the limit. The pixel
        # gate adds a second rule on the anomaly map so a compact anomalous
        # region still fails the part.
        if local_policy and gate["enabled"]:
            pixel_threshold = gate["threshold_ratio"] * threshold * SCORE_SCALE

            try:
                triggered, area, _ = evaluate_pixel_gate(
                    anomaly_map,
                    pixel_threshold,
                    gate["min_area_px"],
                    gate["max_area_px"],
                )
            except ValueError as error:
                raise InspectionError("DECISION_ERROR", f"Pixel gate error: {error}") from error

            if triggered:
                return (
                    True,
                    f"Pixel region {area}px "
                    f"({gate['min_area_px']}-{gate['max_area_px']}px gate, "
                    f"thr {pixel_threshold:.1f})",
                    area,
                )

    return is_defect, reason, gate_region_px
