from main import InspectionWorker
from PySide6.QtCore import Qt
from ui.inspection.overlay import OverlayWindow
from ui.inspection.status_widget import StatusWidget


def test_capture_worker_does_not_find_source_or_inspect(monkeypatch):
    worker = InspectionWorker()
    worker.set_capture_only(True)
    def unexpected():
        raise AssertionError("Capture must not require an inspection window")
    monkeypatch.setattr(worker, "find_window", unexpected)
    monkeypatch.setattr(worker, "_load_pending_recipe", unexpected)
    worker.tick()


def test_capture_frame_without_program(qapp, monkeypatch):
    monkeypatch.setattr("ui.inspection.status_widget.gw.getActiveWindow", lambda: None)
    overlay = OverlayWindow()
    widget = StatusWidget(overlay)
    try:
        flags = overlay.windowFlags()
        widget.capture_width.setValue(200)
        widget.capture_height.setValue(100)
        widget.capture_zoom.setValue(100)
        widget.pages.setCurrentIndex(1)
        qapp.processEvents()
        assert widget._screen_region[2:] == (200, 100)
        assert not overlay.isVisible()
        assert not widget.capture_take_btn.isEnabled()
        assert not hasattr(widget, "capture_source")
        widget.pages.setCurrentIndex(0)
        assert bool(overlay.windowFlags() & Qt.WindowStaysOnTopHint) == bool(flags & Qt.WindowStaysOnTopHint)
        assert not overlay.isVisible()
    finally:
        widget.close()
        overlay.close()


def test_frame_tracks_only_supported_active_windows(qapp, monkeypatch):
    from types import SimpleNamespace
    active = SimpleNamespace(title="LCmicro", _hWnd=12345, isMinimized=False,
                             left=0, top=0, width=20000, height=20000)
    monkeypatch.setattr("ui.inspection.status_widget.gw.getActiveWindow", lambda: active)
    overlay = OverlayWindow()
    monkeypatch.setattr(overlay, "sync_above_window", lambda _: True)
    widget = StatusWidget(overlay)
    try:
        widget.capture_width.setValue(200)
        widget.capture_height.setValue(100)
        widget.capture_zoom.setValue(100)
        widget.pages.setCurrentIndex(1)
        assert overlay.isVisible()
        active.title = "PowerPoint"
        widget._sync_capture_target()
        assert overlay.isVisible()
        active.title = "Google Chrome"
        widget._sync_capture_target()
        assert not overlay.isVisible()
        assert not widget.capture_take_btn.isEnabled()
    finally:
        widget.close()
        overlay.close()
