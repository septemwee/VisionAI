"""Tests for the shared rotated-ROI crop helper and letterbox."""

import math

import numpy as np
import pytest

from utils.image_utils import (
    box_aligned_heatmap,
    crop_rotated_roi,
    heatmap_overlay,
    letterbox,
    letterbox_placement,
    rotated_crop_bounds,
)


def test_crop_contains_bright_block():
    image = np.zeros((100, 120, 3), dtype="uint8")
    image[30:70, 40:80] = 255
    roi = {"cx": 60.0, "cy": 50.0, "width": 40.0, "height": 40.0, "angle": 0.0}

    crop = crop_rotated_roi(image, roi, {})
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.max() == 255


def test_padding_expands_crop():
    image = np.zeros((100, 100, 3), dtype="uint8")
    roi = {"cx": 50.0, "cy": 50.0, "width": 20.0, "height": 20.0, "angle": 0.0}

    no_padding = crop_rotated_roi(image, roi, {"padding": 0})
    padded = crop_rotated_roi(image, roi, {"padding": 10})
    assert padded.shape[0] > no_padding.shape[0]
    assert padded.shape[1] > no_padding.shape[1]


def test_angle_offset_rotates_content():
    image = np.zeros((100, 100, 3), dtype="uint8")
    image[10, 50] = 255  # top-center marker
    roi = {"cx": 50.0, "cy": 50.0, "width": 80.0, "height": 80.0, "angle": 0.0}

    crop = crop_rotated_roi(image, roi, {"angle_offset": 180.0, "padding": 10})
    assert crop.shape == (100, 100, 3)
    # A 180-degree rotation about the ROI center moves (x=50, y=10) to
    # (50, 90); with the 10px padding the crop starts at (0, 0), so the
    # marker lands at row 90, col 50.
    assert crop[90, 50, 0] == 255


def test_letterbox_keeps_aspect_and_pads():
    image = np.full((50, 100, 3), 200, dtype="uint8")
    canvas = letterbox(image, 64, 64)

    assert canvas.shape == (64, 64, 3)
    assert canvas[16, 32, 0] == 200  # centered 2:1 content
    assert canvas[0, 32, 0] == 0  # letterbox bars are black


def test_heatmap_overlay_fixed_scale_is_deterministic():
    image = np.zeros((16, 16, 3), dtype="uint8")
    amap = np.full((16, 16), 50.0)

    first = heatmap_overlay(image, amap)
    second = heatmap_overlay(image, amap)

    assert first is not None and second is not None
    assert np.array_equal(first, second)


def test_letterbox_placement_matches_letterbox_math():
    placement = letterbox_placement(100, 50, 64, 64)
    assert placement["scale"] == pytest.approx(0.64)
    assert placement["width"] == 64 and placement["height"] == 32
    assert placement["x_offset"] == 0 and placement["y_offset"] == 16


def test_rotated_crop_bounds_match_crop_rotated_roi():
    image = np.zeros((200, 200, 3), dtype="uint8")
    roi = {"cx": 60.0, "cy": 50.0, "width": 40.0, "height": 40.0, "angle": 0.0}
    bounds = rotated_crop_bounds(image.shape, roi, {})
    crop = crop_rotated_roi(image, roi, {})
    assert bounds == (30, 20, 90, 80)
    assert crop.shape[:2] == (bounds[3] - bounds[1], bounds[2] - bounds[0])


def test_heatmap_overlay_mask_limits_blend():
    image = np.zeros((32, 32, 3), dtype="uint8")
    image[:] = 40
    amap = np.full((32, 32), 50.0)

    unmasked = heatmap_overlay(image, amap)
    fully = heatmap_overlay(image, amap, mask=np.ones((32, 32)))
    none_mask = heatmap_overlay(image, amap, mask=np.zeros((32, 32)))

    assert not np.array_equal(unmasked, image)
    assert np.array_equal(fully, unmasked)
    assert np.array_equal(none_mask, image)


def test_box_aligned_heatmap_maps_peak_back_to_frame():
    frame = np.zeros((400, 400, 3), dtype="uint8")
    marker = (206, 196)
    frame[marker[1] - 2:marker[1] + 3, marker[0] - 2:marker[0] + 3] = 255

    angle = math.radians(30.0)
    roi = {"cx": 200.0, "cy": 200.0, "width": 120.0, "height": 80.0, "angle": angle}
    crop = crop_rotated_roi(frame, roi, {})
    crop_h, crop_w = crop.shape[:2]

    intensity = crop.astype(float).sum(axis=2)
    by, bx = np.unravel_index(np.argmax(intensity), intensity.shape)
    placement = letterbox_placement(crop_w, crop_h, 256, 256)

    amap = np.zeros((256, 256), dtype=np.float32)
    mx = int(placement["x_offset"] + bx * placement["scale"])
    my = int(placement["y_offset"] + by * placement["scale"])
    amap[my - 2:my + 3, mx - 2:mx + 3] = 90.0

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    center = np.array([roi["cx"], roi["cy"]])
    half = np.array([roi["width"] / 2.0, roi["height"] / 2.0])
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    local = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                      [half[0], half[1]], [-half[0], half[1]]])
    points = center + local @ rot

    result = box_aligned_heatmap(frame, roi, {}, crop, amap, points, 256, 256)
    assert result is not None

    pad = max(8, int(0.1 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))))
    dx1 = max(0, int(points[:, 0].min()) - pad)
    dy1 = max(0, int(points[:, 1].min()) - pad)
    red = result[:, :, 2].astype(float)
    frame_red = frame[dy1:dy1 + red.shape[0], dx1:dx1 + red.shape[1], 2].astype(float)
    added = np.clip(red - frame_red, 0.0, None)
    ys, xs = np.nonzero(added > 0.5 * added.max())
    weight = added[ys, xs]
    frame_peak = (
        float((xs * weight).sum() / weight.sum()) + dx1,
        float((ys * weight).sum() / weight.sum()) + dy1,
    )
    assert abs(frame_peak[0] - marker[0]) <= 4
    assert abs(frame_peak[1] - marker[1]) <= 4