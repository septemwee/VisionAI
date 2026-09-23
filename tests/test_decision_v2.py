import numpy as np
import pytest

from utils.decision_v2 import analyze_map, decide, calibrate


def config(**updates):
    return dict(pixel_low_threshold=4.0, pixel_high_threshold=8.0,
                min_region_area=3, spatial_sigma=0.0, **updates)


def test_hysteresis_grows_only_regions_connected_to_strong_core():
    amap = np.zeros((12, 12), np.float32)
    amap[1:4, 1:4] = 5
    amap[2, 2] = 9
    amap[7:10, 7:10] = 5
    result = analyze_map(amap, config())
    assert result['mask'].sum() == 9
    assert not result['mask'][8, 8]
    region = result['regions'][0]
    assert region['area'] == 9
    assert region['peak'] == 9
    assert region['mean'] == pytest.approx(49 / 9)
    assert region['bbox'] == [1, 1, 3, 3]
    assert decide(0.1, 0.6, result)


def test_small_core_can_qualify_after_growth_and_image_failure_needs_no_region():
    amap = np.zeros((10, 10), np.float32)
    amap[2:4, 2:4] = 6
    amap[2, 2] = 9
    assert analyze_map(amap, config())['local_fail']
    empty = analyze_map(np.zeros_like(amap), config())
    assert not decide(0.1, 0.6, empty)
    assert decide(0.7, 0.6, empty)


def test_map_is_current_image_only_and_input_is_not_mutated():
    first = np.full((10, 10), 10, dtype=np.float32)
    before = first.copy()
    analyze_map(first, config())
    second = analyze_map(np.zeros_like(first), config())
    assert not second['local_fail']
    np.testing.assert_array_equal(first, before)


@pytest.mark.parametrize('amap', [None, np.zeros((0, 0)), np.full((4, 4), np.nan)])
def test_invalid_maps_are_not_silently_passed(amap):
    with pytest.raises(ValueError):
        analyze_map(amap, config())


def test_invalid_threshold_order_is_rejected():
    settings = config()
    settings['pixel_low_threshold'] = 10
    with pytest.raises(ValueError):
        analyze_map(np.zeros((4, 4)), settings)


def test_calibration_uses_maps_and_rejects_good_regions():
    good = [np.ones((12, 12), np.float32), np.full((12, 12), 2, np.float32)]
    defect = np.ones((12, 12), np.float32)
    defect[3:7, 3:7] = 12
    settings, metrics = calibrate(good, [defect], spatial_sigma=0)
    assert metrics['good_false_alarms'] == 0
    assert metrics['synthetic_detected'] == 1
    assert analyze_map(defect, settings)['local_fail']
