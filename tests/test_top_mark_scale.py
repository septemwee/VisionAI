"""Regression for top-mark crops smaller than the stored template."""

import cv2

from services.top_mark_service import TopMarkService
from utils.paths import RECIPES_DIR


def test_smaller_crop_of_saved_template_can_be_matched():
    template_path = RECIPES_DIR / "TJA1041_SO14" / "top_mark_template.jpg"
    template = cv2.imread(str(template_path))
    assert template is not None
    crop = cv2.resize(template, (534, 347), interpolation=cv2.INTER_AREA)

    service = TopMarkService()
    angle, score, gap = service.detect_orientation(crop, template_path)
    assert angle == 0
    assert score > -999.0
    assert gap > 0

    upside_down = cv2.rotate(crop, cv2.ROTATE_180)
    angle, score, gap = service.detect_orientation(upside_down, template_path)
    assert angle == 180
    assert score > -999.0
    assert gap > 0
