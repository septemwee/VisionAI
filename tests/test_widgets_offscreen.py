"""Offscreen widget tests for the trainer UI pieces."""

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from services import recipe_service as recipe_service_module
from services import training_assistant as ta
from ui.trainer.pages.dataset_page import DatasetPage
from ui.trainer.pages.export_page import ExportPage
from ui.trainer.pages.review_page import ReviewPage
from ui.trainer.pages.training_page import TrainingPage
from ui.trainer.step_status_bar import StepStatusBar


class _StubParent:
    def __init__(self, recipe=None):
        self.current_recipe_data = recipe
        self.refresh_calls = 0

    def refresh_step_bar(self):
        self.refresh_calls += 1

    def update_context(self, recipe):
        self.current_recipe_data = recipe


def _ok_recipe():
    return {
        "recipe_name": "R",
        "package_family": "SO",
        "package_type": "SO14",
        "pin_count": 14,
        "model": {"trained": True, "path": "x"},
        "anomaly_threshold": 0.66,
        "calibration": {
            "proposed": 0.66,
            "p99": 0.55,
            "margin": 1.2,
            "expected_false_alarms": 0,
            "good_summary": {
                "min": 0.5,
                "median": 0.52,
                "p95": 0.54,
                "p99": 0.55,
                "max": 0.56,
                "count": 10,
            },
        },
        "roi_statistics": {"saved_count": 12, "rejected_count": 1},
        "version": "1.0",
    }


def test_scroll_page_helper(qapp):
    from PySide6.QtWidgets import QFrame, QScrollArea

    from ui.trainer.scroll_page import scrollable_page

    page = DatasetPage()
    scroll = scrollable_page(page)

    assert scroll.widgetResizable() is True
    assert scroll.frameShape() == QFrame.NoFrame
    assert scroll.widget() is page


def test_trainer_window_pages_scrollable(qapp, monkeypatch):
    from PySide6.QtWidgets import QScrollArea, QWidget

    from ui.trainer import trainer_window as tw

    class _StubPage(QWidget):
        """Page stand-in exposing the methods TrainerWindow calls."""

        def refresh_calibration_panel(self):
            pass

        def refresh_recommendations(self):
            pass

        def refresh_summary(self):
            pass

    # Stub the pages so the window builds without loading YOLO weights.
    for name in (
        "RecipePage",
        "DatasetPage",
        "ROIPage",
        "RecommendationPage",
        "TrainingPage",
        "ReviewPage",
        "ExportPage",
    ):
        monkeypatch.setattr(tw, name, _StubPage)

    window = tw.TrainerWindow()

    assert window.pages.count() == 7
    for index in range(window.pages.count()):
        entry = window.pages.widget(index)
        assert isinstance(entry, QScrollArea), f"page {index} is not scrollable"
        assert entry.widgetResizable() is True
        assert isinstance(entry.widget(), _StubPage)


def test_step_status_bar_next_button(qapp):
    bar = StepStatusBar()

    bar.update_statuses(ta.compute_step_status({"recipe_name": "R"}))
    assert bar.next_button.text().startswith("Next: S1")
    assert bar.next_button.isEnabled()

    complete = {
        step_id: {"status": ta.STATUS_OK, "detail": ""}
        for step_id in ta.STEP_IDS
    }
    bar.update_statuses(complete)
    assert bar.next_button.text() == "All Steps OK"
    assert not bar.next_button.isEnabled()


def test_training_page_panel_states(qapp):
    page = TrainingPage()

    page.refresh_calibration_panel()
    assert not page.btn_calibrate.isEnabled()

    page.parent_window = _StubParent(_ok_recipe())
    page.refresh_calibration_panel()
    assert page.btn_calibrate.isEnabled()


def test_training_page_proposal_and_apply(qapp, temp_recipes_dir, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    page = TrainingPage()
    recipe = _ok_recipe()
    page.parent_window = _StubParent(recipe)

    page._scores = [("a.png", 0.54), ("b.png", 0.56)]
    page._recompute_proposal()

    assert page._proposal is not None
    assert page.btn_apply_calibration.isEnabled()
    assert "Proposed threshold" in page.lbl_proposed.text()
    assert "2 images" in page.lbl_stats.text()

    page.apply_calibration()

    proposed = page._proposal[0]
    assert recipe["anomaly_threshold"] == pytest.approx(proposed)
    assert recipe["calibration"]["evaluated_at"]

    saved = recipe_service_module.RecipeService().load_recipe("R")
    assert saved["anomaly_threshold"] == pytest.approx(proposed)
    assert saved["calibration"]["proposed"] == pytest.approx(proposed)


def test_training_page_blocks_saturated_proposal(qapp, temp_recipes_dir, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    page = TrainingPage()
    recipe = _ok_recipe()
    page.parent_window = _StubParent(recipe)

    # P99 * margin saturates at exactly 1.0, which live inspection refuses
    # for calibrated recipes; the apply path must be blocked.
    page._scores = [("a.png", 0.95), ("b.png", 0.96)]
    page._recompute_proposal()

    assert page._proposal is not None
    assert not page.btn_apply_calibration.isEnabled()

    page.apply_calibration()

    assert recipe["anomaly_threshold"] == pytest.approx(0.66)
    assert recipe["calibration"]["proposed"] == pytest.approx(0.66)


def test_export_page_summary_and_finalize(qapp, temp_recipes_dir, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    page = ExportPage()
    recipe = _ok_recipe()
    page.parent_window = _StubParent(recipe)

    page.refresh_summary()
    text = page.summary_view.toPlainText()
    assert "Recipe : R" in text
    assert "trained" in text
    assert page.btn_finalize.isEnabled()

    page.finalize()
    assert recipe["validated"] is True
    assert recipe["version"] == "1.1"

    saved = recipe_service_module.RecipeService().load_recipe("R")
    assert saved["validated"] is True


def test_dataset_page_analysis_display(qapp):
    page = DatasetPage()

    geometry = {
        "images_analyzed": 3,
        "images_with_detection": 2,
        "size_stats": {"median_width": 101.0, "median_height": 51.0},
        "median_aspect_ratio": 2.0,
        "dominant_class": "SO14",
        "registry_match": {
            "package_family": "SOIC",
            "package_type": "SO14",
            "pin_count": 14,
            "match_score": 1.0,
        },
    }

    page._display_geometry(geometry)
    assert "SOIC" in page.family_label.text()
    assert "SO14" in page.type_label.text()
    assert "14" in page.pin_label.text()
    assert "2.00" in page.aspect_label.text()
    assert "101" in page.size_label.text()

    page._display_quality({}, geometry)
    assert "2/3" in page.coverage_label.text()
    assert "None" in page.quality_label.text()

    page._display_geometry({})
    assert page.family_label.text().endswith("-")
    assert page.pin_label.text().endswith("-")


def test_review_page_apply_threshold(qapp, temp_recipes_dir, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    page = ReviewPage()
    recipe = {"recipe_name": "R"}
    page.parent_window = _StubParent(recipe)

    page.threshold_slider.setValue(66)
    page.apply_threshold_to_recipe()

    assert recipe["anomaly_threshold"] == pytest.approx(0.66)
    saved = recipe_service_module.RecipeService().load_recipe("R")
    assert saved["anomaly_threshold"] == pytest.approx(0.66)


def test_recommendation_page_metadata_path(qapp, temp_recipes_dir, monkeypatch):
    from ui.trainer.pages.recommendation_page import RecommendationPage

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(
        "ui.trainer.pages.recommendation_page.ROIService", None
    )

    service = recipe_service_module.RecipeService()
    service.create_recipe(
        "X",
        {
            "recipe_name": "X",
            "package_family": "SO",
            "package_type": "SO14",
            "pin_count": 14,
            "prepared_dataset_path": "",
        },
    )

    page = RecommendationPage()
    recipe = {
        "recipe_name": "NEW",
        "package_family": "SO",
        "package_type": "SO14",
        "pin_count": 14,
    }
    page.parent_window = _StubParent(recipe)

    page.refresh_recommendations()

    deadline = time.time() + 10
    while page._result is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert page._result is not None
    assert page._result["decision"] == "review"
    assert page._result["chosen"] == "X"
    assert len(page._cards) == 1

    page.use_model("X")
    assert page.selection_label.text() == "Reuse : X"
    assert page.parent_window.current_recipe_data["recipe_name"] == "X"

    new_recipe = {
        "recipe_name": "NEW",
        "package_family": "SO",
        "package_type": "SO14",
        "pin_count": 14,
    }
    page.parent_window.current_recipe_data = new_recipe
    page.train_new_model()
    saved = service.load_recipe("NEW")
    assert ta.get_assistant_state(saved)["chosen_path"] == "train_new"
