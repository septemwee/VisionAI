"""Tests for the localized-anomaly pixel gate, the shared verdict and the
pixel-gate recipe normalization."""

import numpy as np
import pytest

from services.recipe_service import normalize_pixel_gate
from services.verdict_service import evaluate_verdict
from utils.pixel_gate import evaluate_pixel_gate, largest_region


def _blob_map(value=80.0):
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[10:13, 14:17] = value  # 3x3 blob = 9 px
    return amap


def test_largest_region_finds_blob():
    area, mask = largest_region(_blob_map(), 40.0)
    assert area == 9
    assert mask is not None and int(mask.sum()) == 9


def test_largest_region_empty_below_threshold():
    area, mask = largest_region(_blob_map(value=30.0), 40.0)
    assert area == 0
    assert mask is None


def test_largest_region_unwraps_4d_map():
    area, _ = largest_region(_blob_map()[None, None], 40.0)
    assert area == 9


def test_largest_region_rejects_none_and_nonfinite():
    assert largest_region(None, 40.0)[0] == 0
    bad = _blob_map()
    bad[0, 0] = np.nan
    assert largest_region(bad, 40.0)[0] == 0


def test_gate_triggers_on_compact_region():
    triggered, area, _ = evaluate_pixel_gate(_blob_map(), 40.0, 5, 600)
    assert triggered
    assert area == 9


def test_gate_ignores_map_wide_elevation():
    amap = np.full((32, 32), 50.0, dtype=np.float32)
    triggered, _, _ = evaluate_pixel_gate(amap, 40.0, 5, 600)
    assert not triggered


def test_gate_ignores_speck_below_min_area():
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[5, 5] = 90.0
    triggered, _, _ = evaluate_pixel_gate(amap, 40.0, 5, 600)
    assert not triggered


def test_gate_never_triggers_with_inverted_bounds():
    triggered, _, _ = evaluate_pixel_gate(_blob_map(), 40.0, 600, 5)
    assert not triggered


def test_gate_raises_on_nonfinite_map():
    amap = _blob_map()
    amap[20, 20] = np.nan
    with pytest.raises(ValueError):
        evaluate_pixel_gate(amap, 40.0, 5, 600)


# ---------------------------------------------------------------------------
# Shared verdict (fail-closed rules)
# ---------------------------------------------------------------------------

def test_verdict_passes_clean_part():
    is_defect, reason, region = evaluate_verdict(0.10, np.zeros((32, 32)), 0.5, {})
    assert not is_defect
    assert reason.startswith("Score:")
    assert region == 0


def test_verdict_fails_on_high_score():
    is_defect, reason, region = evaluate_verdict(0.61, np.zeros((32, 32)), 0.60, {})
    assert is_defect
    assert reason.startswith("Score:")
    assert region == 0


def test_verdict_nan_score_fails_closed():
    is_defect, reason, _ = evaluate_verdict(float("nan"), None, 0.5, {})
    assert is_defect
    assert "Invalid PatchCore score" in reason


def test_verdict_nonfinite_map_fails_closed():
    amap = _blob_map()
    amap[20, 20] = np.inf
    is_defect, reason, _ = evaluate_verdict(0.10, amap, 0.5, {})
    assert is_defect
    assert "invalid anomaly map" in reason


def test_verdict_pixel_gate_triggers():
    # 4x4 blob (16 px >= min_area 12, value 80) sits above the gate
    # threshold 0.65*0.6*100 = 39.
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[10:14, 14:18] = 80.0
    recipe = {"verdict_policy": "image_and_pixel_v1"}
    gate = normalize_pixel_gate(recipe)
    recipe["calibration"] = {"proposed": 0.6, "pixel_gate": gate}
    is_defect, reason, region = evaluate_verdict(0.10, amap, 0.6, recipe)
    assert is_defect
    assert "Pixel region" in reason
    assert region == 16


def test_verdict_pixel_gate_disabled():
    recipe = {"pixel_gate": {"enabled": False}}
    is_defect, _, region = evaluate_verdict(0.10, _blob_map(), 0.6, recipe)
    assert not is_defect
    assert region == 0


def test_normalize_pixel_gate_defaults_for_legacy_recipe():
    gate = normalize_pixel_gate({"anomaly_threshold": 0.6})
    assert gate["enabled"] is True
    assert 0.0 < gate["threshold_ratio"] <= 1.0
    assert 1 <= gate["min_area_px"] < gate["max_area_px"]


def test_normalize_pixel_gate_keeps_valid_block():
    recipe = {
        "pixel_gate": {
            "enabled": False,
            "threshold_ratio": 0.5,
            "min_area_px": 8,
            "max_area_px": 99,
        }
    }
    gate = normalize_pixel_gate(recipe)
    assert gate == {
        "enabled": False,
        "threshold_ratio": 0.5,
        "min_area_px": 8,
        "max_area_px": 99,
    }


def test_normalize_pixel_gate_repairs_corrupt_block():
    recipe = {
        "pixel_gate": {
            "enabled": "yes",
            "threshold_ratio": -2,
            "min_area_px": 0,
            "max_area_px": 2,
        }
    }
    gate = normalize_pixel_gate(recipe)
    assert gate["enabled"] is True
    assert 0.0 < gate["threshold_ratio"] <= 1.0
    assert gate["min_area_px"] >= 1
    assert gate["max_area_px"] >= gate["min_area_px"]


def test_normalize_pixel_gate_clamps_ratio_above_one():
    recipe = {"pixel_gate": {"threshold_ratio": 4.0}}
    gate = normalize_pixel_gate(recipe)
    assert gate["threshold_ratio"] == 1.0
