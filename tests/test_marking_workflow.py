import json

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from services.marking_service import MarkingService, save_regions
from ui.inspection.overlay import OverlayWindow
from ui.inspection.status_widget import StatusWidget


def drag(overlay, start, end):
    QTest.mousePress(overlay, Qt.LeftButton, pos=QPoint(*start))
    QTest.mouseMove(overlay, QPoint(*end))
    QTest.mouseRelease(overlay, Qt.LeftButton, pos=QPoint(*end))


def test_select_draw_save_two_templates(qapp, temp_recipes_dir):
    path = temp_recipes_dir / "recipe.json"
    path.write_text(json.dumps({"recipe_name": "test"}))
    overlay = OverlayWindow()
    overlay.resize(800, 600)
    overlay.set_frame(np.random.default_rng(4).integers(0, 255, (600, 800, 3), dtype=np.uint8))
    widget = StatusWidget(overlay)
    widget.current_recipe = {"recipe_name": "test", "_recipe_path": str(path)}
    updates = []
    widget.recipe_changed.connect(updates.append)
    overlay.show()
    widget.on_roi_clicked()
    assert not overlay.roi_buttons["laser_mark"].isEnabled()
    drag(overlay, (100, 180), (400, 400))
    top = overlay.roi_regions["top_mark"]
    QTest.mouseClick(overlay.roi_buttons["laser_mark"], Qt.LeftButton)
    assert overlay.roi_type == "laser_mark"
    assert overlay.roi_regions["top_mark"] == top
    drag(overlay, (150, 220), (240, 270))
    drag(overlay, (160, 230), (250, 280))
    assert overlay.roi_regions["laser_mark"] == (160, 230, 90, 50)
    assert overlay.roi_regions["top_mark"] == top
    QTest.keyClick(overlay, Qt.Key_Return)
    assert not overlay.roi_mode
    saved = json.loads(path.read_text())
    assert saved["laser_mark_roi"]["x"] == pytest.approx(0.2)
    assert (temp_recipes_dir / "top_mark_template.jpg").exists()
    assert (temp_recipes_dir / "laser_mark_template.jpg").exists()
    assert len(updates) == 1
    original = (temp_recipes_dir / "top_mark_template.jpg").read_bytes()
    widget.on_roi_clicked()
    QTest.mouseClick(overlay.roi_buttons["laser_mark"], Qt.LeftButton)
    tx, ty, tw, th = overlay.roi_regions["top_mark"]
    drag(overlay, (tx+10,ty+10), (tx+50,ty+40))
    QTest.keyClick(overlay, Qt.Key_Return)
    assert (temp_recipes_dir / "top_mark_template.jpg").read_bytes() == original
    before = path.read_bytes()
    widget.on_roi_clicked()
    QTest.keyClick(overlay, Qt.Key_Escape)
    assert path.read_bytes() == before
    assert widget.combo_pkg.isEnabled()
    widget.close()
    overlay.close()


def test_invalid_regions_leave_files_unchanged(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text('{"recipe_name":"test"}')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_regions(path, np.zeros((100,100,3),np.uint8),
                     {"top_mark": (10,10,50,50), "laser_mark": (0,0,20,20)},
                     {"top_mark","laser_mark"})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.jpg"))


@pytest.mark.parametrize("scale", [0.5, 0.73, 0.87, 1.0, 1.17, 1.5])
def test_saved_reference_matches_after_resize(tmp_path, scale):
    top = np.full((300, 480, 3), 45, np.uint8)
    cv2.rectangle(top,(12,12),(467,287),(130,130,130),4)
    cv2.putText(top,"IC PACKAGE",(35,75),cv2.FONT_HERSHEY_SIMPLEX,1,(180,180,180),2)
    cv2.putText(top,"LOT 2026",(80,185),cv2.FONT_HERSHEY_SIMPLEX,1.2,(150,150,150),2)
    path = tmp_path / "recipe.json"
    path.write_text('{"recipe_name":"test"}')
    recipe = save_regions(path,top,{"top_mark":(0,0,480,300),
                                   "laser_mark":(70,145,240,50)},
                          {"top_mark","laser_mark"})
    resized = cv2.resize(top,None,fx=scale,fy=scale)
    result = MarkingService().check(resized,recipe)
    assert result["status"] == "MATCH", result
    assert result["score"] >= 0.8


def test_match_position_and_mismatch(tmp_path):
    rng = np.random.default_rng(9)
    top = rng.integers(30, 180, (160, 220), dtype=np.uint8)
    laser = rng.integers(0, 255, (25, 50), dtype=np.uint8)
    top[60:85,80:130] = laser
    cv2.imwrite(str(tmp_path / "top.png"), top)
    cv2.imwrite(str(tmp_path / "laser.png"), laser)
    recipe = {
        "top_mark_template": str(tmp_path / "top.png"),
        "laser_mark_template": str(tmp_path / "laser.png"),
        "laser_mark_roi": {"x":80/220, "y":60/160, "w":50/220, "h":25/160},
    }
    service = MarkingService()
    def check(image):
        frame = np.zeros((250,330), np.uint8)
        frame[30:190,40:260] = image
        return service.check(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),recipe)
    assert check(top)["status"] == "MATCH"
    wrong = top.copy()
    wrong[60:85,80:130] = 100
    assert check(wrong)["status"] == "MISMATCH"
    wrong[100:125,140:190] = laser
    assert check(wrong)["status"] == "POSITION ERROR"
    assert check(np.zeros_like(top))["status"] == "FAIL"
    assert service.check(None, {})["status"] == "DISABLED"


@pytest.mark.parametrize("mark_status, expected", [
    ("MISMATCH", "FAIL"), ("POSITION ERROR", "FAIL"), ("UNKNOWN", "FAIL"), ("FAIL", "FAIL")])
def test_marking_stops_patchcore_and_updates_summary(qapp, temp_recipes_dir, monkeypatch,
                                                    mark_status, expected):
    from main import InspectionWorker
    from services.state_manager import ProgramState
    worker = InspectionWorker(None)
    worker.current_recipe = {"recipe_name": "test", "anomaly_threshold": 0.5}
    frame = np.zeros((300, 400, 3), np.uint8)
    points = np.array([[50,50],[250,50],[250,200],[50,200]],np.float32)
    monkeypatch.setattr(worker, "cached_orientation", lambda *args: (False,0,""))
    monkeypatch.setattr(worker.marking_service, "check", lambda *args:
                        {"status":mark_status, "score":0.3, "reason":"Mark test"})
    def must_not_run(*args):
        pytest.fail("PatchCore must not run when marking blocks inspection")
    monkeypatch.setattr(worker.patchcore_service,"predict_full",must_not_run)
    window = type("Window", (), {"isMinimized":False, "_hWnd":None})()
    monkeypatch.setattr(worker,"find_window",lambda:window)
    monkeypatch.setattr(worker,"grab_frame",lambda _: (frame,(0,0,400,300)))
    monkeypatch.setattr(worker,"detect",lambda _: (None,10))
    monkeypatch.setattr(worker,"parse_boxes",lambda _: [
        {"points":points,"conf":0.99,"label":"test"}])
    statuses = []
    worker.status_update.connect(statuses.append)
    worker.tick()
    assert statuses[-1]["inspection_result"] == expected
    assert statuses[-1]["marking"]["status"] == mark_status
    overlay = OverlayWindow()
    widget = StatusWidget(overlay)
    widget.update_status(**statuses[-1])
    assert mark_status in widget.marking_status.text()
    assert "IC #1" in widget.marking_detail.text()
    widget.close()
    overlay.close()
