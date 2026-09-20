import numpy as np
import pytest

from services.verdict_service import evaluate_verdict


def test_enabled_pixel_gate_can_fail_a_small_region():
    anomaly_map = np.zeros((1, 1, 32, 32), dtype=float)
    anomaly_map[0, 0, 10:14, 10:14] = 90.0
    defect, reason, area = evaluate_verdict(
        0.1,
        anomaly_map,
        0.6,
        {"verdict_policy": "image_and_pixel_v1",
         "calibration": {"proposed": 0.6, "pixel_gate": {"enabled": True, "threshold_ratio": 0.5,
                         "min_area_px": 4, "max_area_px": 100}},
         "pixel_gate": {"enabled": True, "threshold_ratio": 0.5,
                         "min_area_px": 4, "max_area_px": 100}},
    )
    assert defect
    assert area == 16
    assert reason.startswith("Pixel region")


def test_disabled_pixel_gate_does_not_change_image_score():
    anomaly_map = np.ones((32, 32), dtype=float) * 90.0
    defect, reason, area = evaluate_verdict(
        0.1, anomaly_map, 0.6,
        {"pixel_gate": {"enabled": False}},
    )
    assert not defect
    assert area == 0
    assert reason.startswith("Score:")


@pytest.mark.parametrize("amap", [None, np.empty((0, 0)), np.full((4, 4), np.nan), np.zeros((4,))])
def test_invalid_output_never_passes_even_with_gate_disabled(amap):
    defect, reason, _ = evaluate_verdict(0.1, amap, 0.6, {"pixel_gate": {"enabled": False}})
    assert defect
    assert "unavailable" in reason


def test_legacy_recipe_keeps_image_only_verdict():
    amap = np.zeros((32, 32))
    amap[10:14, 10:14] = 90
    recipe = {"pixel_gate": {"enabled": True, "threshold_ratio": 0.5,
                             "min_area_px": 4, "max_area_px": 100}}
    assert not evaluate_verdict(0.1, amap, 0.6, recipe)[0]
    recipe["verdict_policy"] = "image_and_pixel_v1"
    defect, reason, _ = evaluate_verdict(0.1, amap, 0.6, recipe)
    assert defect
    assert "requires calibration" in reason
