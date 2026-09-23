"""Slow consumers must skip backlog and reject unrelated results."""
from types import SimpleNamespace
from threading import Event, Thread

import numpy as np

from services.latest_frame import LatestFrame


def test_slow_consumer_gets_latest_without_backlog():
    slot = LatestFrame()
    entered, release = Event(), Event()
    observed = []
    slot.put("first")

    def consume():
        observed.append(slot.read()[1])
        entered.set()
        assert release.wait(2)
        observed.append(slot.read()[1])

    consumer = Thread(target=consume)
    consumer.start()
    assert entered.wait(2)
    for index in range(100):
        slot.put(index)
    release.set()
    consumer.join(2)
    assert not consumer.is_alive()
    assert observed == ["first", 99]


def make_controller():
    from main import InspectionApp
    statuses, overlays = [], []
    widget = SimpleNamespace(capture_mode=False, recipe_request_id=7,
                             update_status=lambda **kw: statuses.append(kw))
    worker = SimpleNamespace(live_frames=LatestFrame(), completed_frames=LatestFrame(),
                             pipeline_epoch=3)
    controller = SimpleNamespace(
        worker=worker, status_widget=widget, overlay=SimpleNamespace(roi_mode=False),
        _display_versions=None, _display_source=None, _status_result_version=0,
        _apply_overlay=lambda *args: overlays.append(args),
        _apply_status=lambda payload: statuses.append(payload),
    )
    points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    packet = dict(epoch=3, source_id=5, window=SimpleNamespace(_hWnd=9),
                  frame=np.zeros((30, 30, 3), np.uint8), rect=(0, 0, 30, 30),
                  boxes=[dict(points=points, conf=.99, label="R")], fps=20,
                  captured_at=0, error="")
    completed = dict(packet, boxes=[dict(packet["boxes"][0], result="PASS", score=.1)],
                     payload=dict(recipe_request_id=7, inspection_result="PASS"))
    worker.live_frames.put(packet)
    worker.completed_frames.put(completed)
    return InspectionApp, controller, packet, completed, statuses, overlays


def test_detection_updates_do_not_reset_completed_result_colour():
    app, controller, packet, _, statuses, overlays = make_controller()
    app._poll_live(controller)
    controller.worker.live_frames.put(dict(packet))
    app._poll_live(controller)
    assert [entry[2][0]["result"] for entry in overlays] == ["PASS"]
    assert sum(item["inspection_result"] == "PASS" for item in statuses) == 1


def test_overlay_uses_completed_frame_geometry_with_its_segment():
    """The visible box must be from the exact capture PatchCore inspected.

    A newer YOLO frame can be geometrically different while PatchCore is
    still running. Re-projecting the old verdict onto that newer box makes
    the segment appear detached from the package.
    """
    app, controller, packet, completed, _statuses, overlays = make_controller()
    completed_frame = np.full((30, 30, 3), 77, np.uint8)
    completed_points = np.array([[4, 3], [24, 3], [24, 23], [4, 23]], dtype=float)
    completed = dict(
        completed,
        frame=completed_frame,
        boxes=[dict(completed["boxes"][0], points=completed_points,
                    segments=[[[6, 6], [12, 6], [12, 12], [6, 12]]])],
    )
    controller.worker.completed_frames.put(completed)

    app._poll_live(controller)

    displayed_frame, _rect, displayed_boxes, _hwnd = overlays[-1]
    assert displayed_frame is completed_frame
    assert np.array_equal(displayed_boxes[0]["points"], completed_points)
    assert displayed_boxes[0]["segments"] == completed["boxes"][0]["segments"]


def test_old_source_result_cannot_follow_reappearing_source():
    app, controller, packet, _, statuses, overlays = make_controller()
    controller.worker.live_frames.put(dict(packet, source_id=6))
    app._poll_live(controller)
    assert all(item["inspection_result"] != "PASS" for item in statuses)
    assert overlays[-1][0] is None


def test_old_recipe_result_is_rejected():
    app, controller, _, completed, statuses, overlays = make_controller()
    controller.worker.completed_frames.put(dict(completed, epoch=2))
    app._poll_live(controller)
    assert all(item["inspection_result"] != "PASS" for item in statuses)
    assert overlays[-1][0] is None


def test_recipe_change_hides_old_completed_overlay_while_loading():
    app, controller, packet, _completed, _statuses, overlays = make_controller()
    app._poll_live(controller)
    assert overlays[-1][0] is packet["frame"]

    controller.status_widget.recipe_request_id = 8
    controller.worker.live_frames.put(dict(packet))
    app._poll_live(controller)

    assert overlays[-1][0] is None


def test_live_detection_continues_while_inspection_is_blocked(monkeypatch):
    import main
    started, release, captured_more, inspected_again = Event(), Event(), Event(), Event()
    captures, inspected = [], []
    window = SimpleNamespace(_hWnd=9, isMinimized=False)
    points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    monkeypatch.setattr(main.mss, "mss", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(main.InspectionWorker, "find_window", lambda self: window)
    monkeypatch.setattr(main.InspectionWorker, "detect", lambda self, frame: (None, 20))
    monkeypatch.setattr(main.InspectionWorker, "parse_boxes", lambda self, result: [
        dict(points=points.copy(), conf=.99, label="R")])

    def capture(self, window):
        captures.append(len(captures) + 1)
        if len(captures) >= 5:
            captured_more.set()
        return np.full((30, 30, 3), len(captures), np.uint8), (0, 0, 30, 30)

    def inspect(self, frame, points, object_key=None, render_heatmap=False):
        inspected.append(int(frame[0, 0, 0]))
        if len(inspected) == 1:
            started.set()
            assert release.wait(3)
        else:
            inspected_again.set()
        return "PASS", .1, "", False, None

    monkeypatch.setattr(main.InspectionWorker, "grab_frame", capture)
    monkeypatch.setattr(main.InspectionWorker, "inspect", inspect)
    worker = main.InspectionWorker(object())
    worker.current_recipe = dict(recipe_name="R")
    worker.start()
    try:
        assert started.wait(3)
        assert captured_more.wait(3), "detection blocked behind inspection"
        release.set()
        assert inspected_again.wait(3)
        assert inspected[1] >= 4, "consumer replayed a stale frame backlog"
    finally:
        release.set()
        worker.stop()
        assert worker.wait(3000)
