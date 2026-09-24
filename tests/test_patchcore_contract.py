"""Tests for the PatchCoreService folder-evaluation contract."""

import cv2
import pytest
from types import SimpleNamespace

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


def test_evaluate_folder_returns_pairs(service, monkeypatch, tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "b.png"), np.ones((8, 8, 3), np.uint8))
    scores = iter((0.54, 0.555))
    monkeypatch.setattr(
        service,
        "predict_full",
        lambda image: (next(scores), np.zeros((8, 8), np.float32)),
    )

    scored = service.evaluate_folder(tmp_path)

    assert scored == [("a.png", pytest.approx(0.54)), ("b.png", pytest.approx(0.555))]


def test_evaluate_folder_empty_is_rejected(service, tmp_path):
    with pytest.raises(ValueError, match="No calibration images"):
        service.evaluate_folder(tmp_path)


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


def test_load_model_uses_num_neighbors_recorded_with_artifact(tmp_path, monkeypatch):
    """A compact bank must not silently reopen with PatchCore's k=9 default."""
    model_dir = tmp_path / "candidate"
    model_dir.mkdir()
    for name in ("patchcore.pt", "memory_bank.pt"):
        (model_dir / name).write_bytes(b"stub")
    (model_dir / "metadata.json").write_text(
        '{"memory_bank_shape": [12, 3], "num_neighbors": 1}', encoding="utf-8"
    )

    raw = SimpleNamespace(memory_bank=None, num_neighbors=9)
    fake_model = SimpleNamespace(
        model=raw,
        load_state_dict=lambda *_args, **_kwargs: SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        eval=lambda: None,
    )
    class _FakePatchcore:
        @staticmethod
        def configure_pre_processor(**_kwargs):
            return object()

        def __new__(cls, **_kwargs):
            return fake_model

    monkeypatch.setattr(pcs, "_import_anomalib", lambda: None)
    monkeypatch.setattr(pcs, "Patchcore", _FakePatchcore)
    load_calls = []

    def load(path, **kwargs):
        load_calls.append((path.name, kwargs))
        return {"state": "stub"} if path.name == "patchcore.pt" else np.zeros((12, 3))

    monkeypatch.setattr(pcs.torch, "load", load)

    service = pcs.PatchCoreService()
    service.load_model(model_dir)

    assert service.model.model.num_neighbors == 1
    assert all(kwargs.get("mmap") is True for _name, kwargs in load_calls)


@pytest.mark.parametrize('content', [
    '{bad json', '[]',
    '{"image_size": [256, 256]}',
    '{"num_neighbors": 0}',
    '{"num_neighbors": "bad"}',
    '{"memory_bank_shape": "bad"}',
])
def test_invalid_model_metadata_is_rejected_before_inference(tmp_path, content):
    from services.patchcore_service import read_model_metadata
    (tmp_path / 'metadata.json').write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        read_model_metadata(tmp_path)


def test_legacy_model_metadata_without_image_size_is_accepted(tmp_path):
    from services.patchcore_service import read_model_metadata
    (tmp_path / 'metadata.json').write_text(
        '{"memory_bank_shape": [512, 1536]}', encoding='utf-8')
    assert read_model_metadata(tmp_path)['memory_bank_shape'] == [512, 1536]


def test_cached_model_cannot_hide_missing_artifact(tmp_path, monkeypatch):
    (tmp_path / 'metadata.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(pcs, '_import_anomalib', lambda: None)
    service = pcs.PatchCoreService()
    service._model_cache[str(tmp_path.resolve())] = object()
    with pytest.raises(FileNotFoundError):
        service.load_model(tmp_path)
