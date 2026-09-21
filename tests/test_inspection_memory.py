"""Behaviour tests for inspection scene memory and clean window capture."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from main import InspectionWorker


class _FakeWindow:
    """Minimal pygetwindow-like window for tick() tests."""

    left, top, width, height = 10, 20, 640, 480
    _hWnd = None


@pytest.fixture
def worker(temp_recipes_dir):
    """An InspectionWorker with captured signal emissions and no real models."""
    instance = InspectionWorker(detection_model=None)
    frames = []
    statuses = []
    instance.overlay_update.connect(lambda frame, rect, boxes, hwnd: frames.append((frame, rect, boxes)))
    instance.status_update.connect(statuses.append)
    return instance, frames, statuses


def _dirty_scene_memory(instance):
    instance.last_boxes = [{"result": "FAIL", "score": 0.9, "stale": True}]
    instance.last_detect_time = 123.0
    instance.last_points = np.zeros((4, 2))
    instance.orientation_state = (None, True, 180, "template.jpg")
    instance.orientation_tick = 1


def test_reset_scene_memory_clears_everything(worker):
    instance, _frames, _statuses = worker
    _dirty_scene_memory(instance)

    instance.reset_scene_memory()

    assert instance.last_boxes == []
    assert instance.last_detect_time == 0.0
    assert instance.last_points is None
    assert instance.orientation_state is None
    assert instance.orientation_tick == 0


def test_no_source_tick_resets_scene_memory(worker, monkeypatch):
    instance, frames, statuses = worker
    _dirty_scene_memory(instance)
    monkeypatch.setattr(instance, "find_window", lambda: None)

    instance.tick()

    assert statuses[-1]["inspection_result"] == "NO SOURCE"
    assert instance.state_manager.get_state().value == "WAITING_SOURCE"
    assert instance.last_boxes == []
    assert instance.last_detect_time == 0.0
    assert instance.last_points is None
    assert instance.orientation_state is None
    assert frames[-1][0] is None  # overlay receives the hide signal


def test_minimized_source_hides_overlay(worker, monkeypatch):
    """A minimized source must be treated like a missing one: the overlay
    hides and scene memory resets instead of drawing stale boxes."""

    class _MinimizedWindow(_FakeWindow):
        isMinimized = True

    instance, frames, statuses = worker
    monkeypatch.setattr(instance, "find_window", lambda: _MinimizedWindow())

    instance.tick()

    assert statuses[-1]["inspection_result"] == "NO SOURCE"
    assert frames[-1][0] is None


def test_capture_error_hides_overlay(worker, monkeypatch):
    """A failing capture (e.g. the source window closing) must hide the
    overlay instead of leaving stale boxes on screen."""
    instance, frames, statuses = worker
    monkeypatch.setattr(instance, "find_window", lambda: _FakeWindow())

    def _failing_grab(window):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(instance, "grab_frame", _failing_grab)

    instance.tick()

    assert statuses[-1]["inspection_result"] == "NO SOURCE"
    assert frames[-1][0] is None


def test_fresh_detection_is_reinspected_not_carried(worker, monkeypatch):
    """A fresh detection must be inspected from the current frame, even when
    the detection memory still holds an old FAIL verdict for the same count."""
    instance, frames, statuses = worker
    _dirty_scene_memory(instance)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fresh_box = {
        "points": np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float),
        "conf": 0.99,
        "label": "R",
    }

    monkeypatch.setattr(instance, "find_window", lambda: _FakeWindow())
    monkeypatch.setattr(instance, "grab_frame", lambda window: (frame, (10, 20, 640, 480)))
    monkeypatch.setattr(instance, "detect", lambda captured: (None, 30.0))
    monkeypatch.setattr(instance, "parse_boxes", lambda result: [dict(fresh_box)])
    monkeypatch.setattr(
        instance,
        "inspect",
        lambda captured, points: ("PASS", 0.1, "Score: 0.1000", False, None),
    )

    instance.current_recipe = {
        "recipe_name": "R",
        "anomaly_threshold": 0.5,
        "top_mark_template": "template.jpg",
    }

    instance.tick()

    payload = statuses[-1]
    assert payload["inspection_result"] == "PASS"
    assert payload["score"] == pytest.approx(0.1)
    assert frames[-1][2][0]["result"] == "PASS"
    # A preliminary DETECTED emission causes yellow/green flicker each cycle.
    assert len(frames) == 1


def test_lost_part_resets_orientation_cache(worker, monkeypatch):
    """When the part disappears, a new part placed at the same spot must be
    orientation-checked fresh instead of reusing the cached verdict."""
    instance, _frames, _statuses = worker
    instance.current_recipe = {
        "recipe_name": "R",
        "anomaly_threshold": 0.5,
    }
    instance.orientation_state = (None, True, 180, "template.jpg")
    instance.orientation_tick = 1

    empty_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    monkeypatch.setattr(instance, "find_window", lambda: _FakeWindow())
    monkeypatch.setattr(instance, "grab_frame", lambda window: (empty_frame, (10, 20, 640, 480)))
    monkeypatch.setattr(instance, "detect", lambda captured: (None, 30.0))
    monkeypatch.setattr(instance, "parse_boxes", lambda result: [])

    instance.tick()

    assert _statuses[-1]["inspection_result"] == "NOT FOUND"
    assert instance.orientation_state is None
    assert instance.orientation_tick == 0
    assert instance.last_points is None


def test_grab_window_content_rejects_missing_hwnd():
    from utils.window_capture import grab_window_content

    frame, rect = grab_window_content(None)
    assert frame is None
    assert rect is None


@pytest.mark.skipif(
    not hasattr(__import__("ctypes"), "windll"), reason="Windows-only capture path"
)
def test_grab_window_content_rejects_invalid_hwnd():
    from utils.window_capture import grab_window_content

    frame, rect = grab_window_content(0)
    assert frame is None
    assert rect is None


# ---------------------------------------------------------------------------
# Fail-closed inspection behavior (review fixes)
# ---------------------------------------------------------------------------


def _inspection_frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def _inspection_points():
    return np.array([[20, 20], [180, 20], [180, 180], [20, 180]], dtype=float)


def test_missing_template_fails_closed(worker):
    instance, _frames, _statuses = worker
    instance.current_recipe = {"recipe_name": "R"}

    is_upside_down, angle, error = instance.check_orientation(_inspection_frame())

    assert not is_upside_down
    assert error  # must report an error, never silently skip the check


def test_orientation_verdict_ignores_match_strength(worker, monkeypatch):
    """The top-mark match-strength gate was removed: weak scores no longer
    fail the inspection; the verdict follows the detected angle alone."""
    instance, _frames, _statuses = worker
    instance.current_recipe = {"recipe_name": "R", "top_mark_template": "t.jpg"}
    monkeypatch.setattr(
        instance.top_mark_service, "detect_orientation", lambda roi, path: (0, 0.10, 0.0)
    )

    is_upside_down, angle, error = instance.check_orientation(_inspection_frame())
    assert not error and not is_upside_down and angle == 0


def test_load_failure_clears_model(worker, monkeypatch):
    instance, _frames, _statuses = worker
    instance.patchcore_service.model = object()  # previous recipe's model
    monkeypatch.setattr(
        instance.patchcore_service,
        "load_model",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )
    instance.pending_recipe = {"recipe_name": "R", "model": {"path": "missing"}}

    instance._load_pending_recipe()

    assert instance.patchcore_service.model is None


def test_calibrated_recipe_rejects_invalid_threshold(worker, monkeypatch):
    instance, _frames, _statuses = worker
    instance.current_recipe = {
        "recipe_name": "R",
        "anomaly_threshold": "",
        "calibration": {"p99": 0.4},
        "top_mark_template": "t.jpg",
        "roi_default": {},
    }
    instance.target_width = 64
    instance.target_height = 64
    monkeypatch.setattr(
        instance.top_mark_service, "detect_orientation", lambda roi, path: (0, 0.90, 0.50)
    )

    def _must_not_score(image):
        raise AssertionError("scoring must not run on an invalid threshold")

    monkeypatch.setattr(instance.patchcore_service, "predict_full", _must_not_score)

    result, score, reason, _upside, _heatmap = instance.inspect(
        _inspection_frame(), _inspection_points()
    )

    assert result == "FAIL"
    assert "Invalid anomaly threshold" in reason


def test_package_roi_matches_crop_rotated_convention(worker):
    instance, _frames, _statuses = worker

    straight = instance._package_roi(_inspection_points())
    assert abs(straight["angle"]) < 1e-6
    assert abs(straight["width"] - 160) < 1e-3
    assert abs(straight["height"] - 160) < 1e-3

    theta = np.deg2rad(30.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    center = np.array([100.0, 100.0])
    half = 60.0
    corners = np.array(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ]
    )
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = (corners @ rotation.T) + center
    roi = instance._package_roi(rotated)

    assert abs(np.degrees(roi["angle"]) - 30.0) < 1e-6
    assert abs(roi["width"] - 120) < 1e-3
    assert abs(roi["height"] - 120) < 1e-3
    assert abs(roi["cx"] - 100.0) < 1e-6


def test_normalize_threshold_semantics():
    from services.recipe_service import (
        normalize_anomaly_threshold,
        stored_threshold_usable,
    )

    assert normalize_anomaly_threshold(0.4333) == pytest.approx(0.4333)
    assert normalize_anomaly_threshold(60.0) == pytest.approx(0.60)
    # Exactly 1.0 can never be exceeded -> unusable -> safe default.
    assert normalize_anomaly_threshold(1.0) == pytest.approx(0.60)
    assert normalize_anomaly_threshold(100.0) == pytest.approx(0.60)
    assert normalize_anomaly_threshold("") == pytest.approx(0.60)

    assert not stored_threshold_usable("")
    assert not stored_threshold_usable(None)
    assert not stored_threshold_usable(1.0)
    assert not stored_threshold_usable(100.0)
    assert stored_threshold_usable(0.4333)
    assert stored_threshold_usable(60.0)


def test_recipe_write_is_atomic(temp_recipes_dir):
    from services.recipe_service import RecipeService

    service = RecipeService()
    service.save_recipe("Atomic", {"recipe_name": "Atomic"})

    recipe_file = temp_recipes_dir / "Atomic" / "recipe.json"
    assert recipe_file.exists()
    assert not list(temp_recipes_dir.glob("**/*.tmp"))
