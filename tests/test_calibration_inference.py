import cv2
import numpy as np
import pytest

from services.patchcore_service import PatchCoreService


def test_calibration_scores_every_image_using_live_inference(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    for path, value in ((tmp_path / "a.png", 10), (tmp_path / "nested" / "a.png", 80)):
        assert cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
    service = PatchCoreService()
    calls = []
    def predict(image):
        calls.append(image.shape)
        return float(image.mean()) / 100, np.zeros((8, 8))
    monkeypatch.setattr(service, "predict_full", predict)
    result = service.evaluate_folder(tmp_path)
    assert [score for _, score in result] == [0.1, 0.8]
    assert len({name for name, _ in result}) == 2
    assert len(calls) == 2


def test_calibration_rejects_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="No calibration"):
        PatchCoreService().evaluate_folder(tmp_path)


def test_calibration_rejects_invalid_prediction(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    service = PatchCoreService()
    monkeypatch.setattr(service, "predict_full", lambda _: (0.1, None))
    with pytest.raises(ValueError, match="Invalid model output"):
        service.evaluate_folder(tmp_path)
