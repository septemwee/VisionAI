import cv2
import numpy as np
import pytest

from services.capture_service import CaptureService, scaled_capture_rect


def test_supplied_screenshot_geometry():
    assert scaled_capture_rect((1920, 1032), (2160, 1620), 51.4, (-107, 27)) == (298, 126, 1110, 833)


def test_center_and_zoom():
    assert scaled_capture_rect((1000, 800), (800, 600), 50) == (300, 250, 400, 300)
    assert scaled_capture_rect((1000, 800), (800, 600), 100) == (100, 100, 800, 600)


def test_rejects_frame_outside_window():
    with pytest.raises(ValueError):
        scaled_capture_rect((1000, 800), (2160, 1620), 100)
    with pytest.raises(ValueError):
        scaled_capture_rect((1000, 800), (800, 600), 50, (1000, 0))


def test_saved_dimensions_match_border(tmp_path):
    service = CaptureService()
    service.set_folder(tmp_path)
    rect = scaled_capture_rect((1920, 1032), (2160, 1620), 51.4, (-107, 27))
    service.set_area(rect, (1920, 1032))
    path = service.save(np.zeros((1032, 1920, 3), dtype=np.uint8))
    assert cv2.imread(str(path)).shape[:2] == (833, 1110)
