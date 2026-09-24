from main import InspectionWorker
from services.state_manager import ProgramState
from ui.inspection.overlay import OverlayWindow
from ui.inspection.status_widget import StatusWidget


def test_first_selection_ready_and_current_verdict(qapp, monkeypatch):
    overlay = OverlayWindow()
    widget = StatusWidget(overlay)
    worker = InspectionWorker()
    widget.recipe_request_changed.connect(worker.apply_recipe)
    worker.status_update.connect(lambda payload: widget.update_status(**payload))
    monkeypatch.setattr(worker.patchcore_service, "load_model", lambda path: None)
    widget.combo_pkg.addItem("Probe", {"recipe_name": "Probe", "model": {"path": "mock"}})
    widget.combo_pkg.setCurrentIndex(widget.combo_pkg.count() - 1)
    assert widget.result.text() == "LOADING MODEL"
    worker._load_pending_recipe()
    assert widget.result.text() == "READY"
    assert not widget.model_loading
    for verdict in ("PASS", "FAIL"):
        worker.status_update.emit(dict(
            system_state=ProgramState.READY, inspection_result=verdict,
            count=1, fps=3, recipe_request_id=widget.recipe_request_id))
        assert widget.result.text() == verdict
        assert "Loading" not in widget.message.text()
    worker.status_update.emit(dict(
        system_state=ProgramState.ERROR, inspection_result="ERROR",
        count=1, fps=0, fail_reason="INVALID_ANOMALY_MAP: missing map",
        recipe_request_id=widget.recipe_request_id))
    assert widget.result.text() == "ERROR"
    assert "INVALID_ANOMALY_MAP" in widget.message.text()
    assert widget.orientation.text() == "NOT CHECKED"
    widget.update_status(ProgramState.READY, "SELECT PACKAGE", 0, 0,
                         model_loading=True, recipe_request_id=widget.recipe_request_id - 1)
    assert widget.result.text() == "ERROR"
    widget.close()
    overlay.close()
