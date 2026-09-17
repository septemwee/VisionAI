"""Tests for the PatchCoreService evaluation contract (stubbed engine)."""

import pytest

import numpy as np

from services import patchcore_service as pcs


class _StubPrediction:
    def __init__(self, name, raw_score):
        self.pred_score = [raw_score]
        self.image_path = [f"some/dir/{name}"]


class _StubEngine:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, model, dataset):
        return self._predictions


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(pcs, "PredictDataset", lambda **kwargs: object())
    svc = pcs.PatchCoreService()
    svc.model = object()
    return svc


def test_evaluate_folder_returns_pairs(service):
    service.engine = _StubEngine(
        [
            _StubPrediction("a.png", 54.0),
            _StubPrediction("b.png", 55.5),
        ]
    )

    scored = service.evaluate_folder("unused")

    assert scored == [("a.png", pytest.approx(0.54)), ("b.png", pytest.approx(0.555))]


def test_evaluate_folder_empty_returns_list(service):
    service.engine = _StubEngine([])
    assert service.evaluate_folder("unused") == []


def test_anomaly_map_smoothing_dampens_and_resets():
    import numpy as np

    service = pcs.PatchCoreService()
    first = np.full((8, 8), 10.0)
    smoothed = service._smooth_anomaly_map(first)
    assert np.allclose(smoothed, first)

    spike = np.full((8, 8), 10.0)
    spike[3, 3] = 90.0
    smoothed = service._smooth_anomaly_map(spike)
    assert smoothed[3, 3] == pytest.approx(0.4 * 90.0 + 0.6 * 10.0)

    other_shape = np.zeros((4, 4))
    assert np.allclose(service._smooth_anomaly_map(other_shape), other_shape)
