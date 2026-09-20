from services.training_assistant import STATUS_ACTION_NEEDED, STATUS_OK, STEP_IDS
from ui.trainer.step_status_bar import StepStatusBar
from ui.trainer.trainer_window import STEP_TITLES, navigation_state
from ui.trainer.pages.training_page import TrainingPage
from ui.trainer.pages.recipe_page import RECIPE_FIELD_ORDER, RecipePage
from ui.theme import TRAINER_STYLESHEET, load_ui_fonts
from ui.theme import caption_label


def _statuses():
    return {
        step_id: {
            "status": STATUS_OK if index < 2 else STATUS_ACTION_NEEDED,
            "detail": f"Detail for {step_id}",
        }
        for index, step_id in enumerate(STEP_IDS)
    }


def test_wizard_titles_cover_every_visible_page():
    assert STEP_TITLES == [
        "Recipe Setup",
        "Dataset",
        "ROI Preparation",
        "Model Strategy",
        "Train & Validate",
        "Model Review",
        "Production Ready",
    ]


def test_stepper_exposes_current_and_completion_state(qapp):
    stepper = StepStatusBar()
    stepper.update_statuses(_statuses())
    stepper.set_current_step("S3")

    assert stepper.badges["S1"].property("completed") is True
    assert stepper.badges["S3"].property("current") is True
    assert stepper.badges["S3"].accessibleName() == "Step 3, ROI, current"
    assert stepper.badges["S4"].property("current") is False


def test_stepper_uses_plain_language_next_action(qapp):
    stepper = StepStatusBar()
    stepper.update_statuses(_statuses())

    assert stepper.next_button.text() == "Required next: ROI"


def test_navigation_state_explains_destination_and_boundaries():
    first = navigation_state(0)
    assert first == {
        "previous_enabled": False,
        "next_enabled": True,
        "next_text": "Continue to Dataset",
    }

    last = navigation_state(len(STEP_TITLES) - 1)
    assert last["previous_enabled"] is True
    assert last["next_enabled"] is False
    assert last["next_text"] == "Setup complete"


def test_training_page_uses_task_focused_actions(qapp):
    page = TrainingPage()
    assert page.btn_train.text() == "Train model"
    assert page.btn_calibrate.text() == "Validate model"
    assert page.btn_apply_calibration.text() == "Use recommended threshold"
    assert "Training activity" in page.txt_log.placeholderText()


def test_trainer_font_loader_is_safe_to_call_repeatedly(qapp):
    first = load_ui_fonts()
    second = load_ui_fonts()
    assert first >= 1
    assert second == first


def test_helper_copy_wraps_on_compact_windows(qapp):
    assert caption_label("Helpful guidance").wordWrap() is True


def test_recipe_setup_fits_compact_content_area(qapp, temp_recipes_dir):
    page = RecipePage()
    page.setStyleSheet(TRAINER_STYLESHEET)
    page.ensurePolished()
    assert page.minimumSizeHint().height() <= 330
    assert page.notes.maximumHeight() <= 64


def test_recipe_fields_follow_natural_reading_order():
    assert RECIPE_FIELD_ORDER == [
        "Recipe Name",
        "Package Family",
        "Package Type",
        "Package Size",
        "Package Version",
        "Type Name",
        "Pin Count",
        "Notes",
    ]
