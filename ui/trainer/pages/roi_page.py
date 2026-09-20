"""ROI verification and training-dataset generation page of the trainer."""

import math
import shutil
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from services.dataset_service import IMAGE_EXTENSIONS
from services.roi_service import ROIService
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    SPACE_XS,
    card_frame,
    make_button,
    section_label,
    title_label,
)
from ui.workers.auto_tune_worker import AutoTuneWorker
from ui.workers.crop_worker import CropWorker
from utils.image_utils import (
    DEFAULT_ANGLE_OFFSET,
    DEFAULT_HEIGHT_SCALE,
    DEFAULT_PADDING,
    DEFAULT_WIDTH_SCALE,
    letterbox,
    crop_yolo_obb,
)
from utils.paths import RECIPES_DIR, relativize_recipe_path

# Fallback QA tolerances applied to generated crops, relative to the average
# size (auto-tune stores per-recipe values in recipe["crop_qa"]).
WIDTH_TOLERANCE = 0.25
HEIGHT_TOLERANCE = 0.25
RATIO_TOLERANCE = 0.2

# Cap on in-memory flagged crops offered for manual keep.
MAX_FLAGGED_IN_MEMORY = 60


class ROIPage(QWidget):
    """Verifies detected ROIs and builds the normal-image training set."""

    def __init__(self):
        super().__init__()

        self.roi_service = ROIService()
        self.recipe_service = RecipeService()

        self.dataset_path = ""
        self.image_files = []
        self.current_index = 0
        self.current_image_path = None
        self.parent_window = None
        self.roi_data = None
        self.roi_overrides = {}

        self.autotune_worker = None
        self.crop_worker = None
        self._good_dir = None
        self._flagged_crops = []
        self._target_width = 640
        self._target_height = 640

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        root.setSpacing(SPACE_M)

        root.addWidget(title_label("ROI Dataset Preparation"))

        # ----- Image navigation -----
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(SPACE_S)

        self.btn_previous = make_button("Previous image", "secondary")
        self.btn_next = make_button("Next image", "secondary")
        self.image_index_label = QLabel("0 / 0")
        self.image_index_label.setStyleSheet(
            "border:none;background:transparent;font-weight:600;"
        )

        self.btn_previous.clicked.connect(self.previous_image)
        self.btn_next.clicked.connect(self.next_image)

        nav_layout.addWidget(self.btn_previous)
        nav_layout.addWidget(self.image_index_label)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        root.addLayout(nav_layout)

        # ----- Previews -----
        image_layout = QHBoxLayout()
        image_layout.setSpacing(SPACE_M)

        self.original_preview = QLabel("Original Image")
        self.original_preview.setAlignment(Qt.AlignCenter)
        self.original_preview.setObjectName("preview")
        self.original_preview.setMinimumSize(260, 200)

        self.roi_preview = QLabel("ROI Preview")
        self.roi_preview.setAlignment(Qt.AlignCenter)
        self.roi_preview.setObjectName("preview")
        self.roi_preview.setMinimumSize(260, 200)

        image_layout.addWidget(self.original_preview)
        image_layout.addWidget(self.roi_preview)
        root.addLayout(image_layout)

        # ----- ROI controls -----
        control_card = card_frame()
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        control_layout.setSpacing(SPACE_S)

        control_layout.addWidget(section_label("ROI Settings"))

        self.width_scale_spin = self._create_double_spin(0.5, 3.0, 0.1, DEFAULT_WIDTH_SCALE)
        self.height_scale_spin = self._create_double_spin(0.5, 3.0, 0.1, DEFAULT_HEIGHT_SCALE)
        self.padding_spin = self._create_int_spin(0, 100, DEFAULT_PADDING)
        self.angle_offset_spin = self._create_double_spin(-180, 180, 1, DEFAULT_ANGLE_OFFSET)

        for spin in (
            self.width_scale_spin,
            self.height_scale_spin,
            self.padding_spin,
            self.angle_offset_spin,
        ):
            spin.valueChanged.connect(self.refresh_preview)
            spin.setEnabled(False)
            spin.setToolTip("Trainer crops use the same YOLO OBB geometry as live Inspection.")

        fields = QFormLayout()
        fields.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        fields.setHorizontalSpacing(SPACE_M)
        fields.setVerticalSpacing(SPACE_XS)
        fields.addRow("Width Scale", self.width_scale_spin)
        fields.addRow("Height Scale", self.height_scale_spin)
        fields.addRow("Padding (px)", self.padding_spin)
        fields.addRow("Angle Offset (°)", self.angle_offset_spin)
        control_layout.addLayout(fields)

        detection_row = QHBoxLayout()
        detection_row.setSpacing(SPACE_M)

        self.confidence_label = QLabel("Confidence : -")
        self.size_label = QLabel("Size : -")
        self.confidence_label.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        self.size_label.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        detection_row.addWidget(self.confidence_label)
        detection_row.addWidget(self.size_label)
        detection_row.addStretch()
        control_layout.addLayout(detection_row)

        root.addWidget(control_card)

        # ----- Actions -----
        actions_card = card_frame()
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        actions_layout.setSpacing(SPACE_S)

        actions_layout.addWidget(section_label("Actions"))

        actions_layout.addWidget(
            QLabel(
                "Auto-tune derives the ROI defaults and QA tolerances from "
                "the detected package sizes."
            )
        )
        self.btn_autotune = make_button("Auto-tune Defaults", "secondary")
        self.btn_autotune.clicked.connect(self.start_auto_tune)
        self.btn_autotune.setEnabled(False)
        actions_layout.addWidget(self.btn_autotune)

        self.btn_save_override = make_button("Save Override For This Image", "secondary")
        self.btn_save_override.clicked.connect(self.save_override)
        self.btn_save_override.setEnabled(False)
        actions_layout.addWidget(self.btn_save_override)

        self.btn_save_default = make_button("Save As Default", "secondary")
        self.btn_save_default.clicked.connect(self.save_default)
        self.btn_save_default.setEnabled(False)
        actions_layout.addWidget(self.btn_save_default)

        self.btn_generate = make_button("Generate Training Dataset")
        self.btn_generate.clicked.connect(self.generate_training_dataset)
        actions_layout.addWidget(self.btn_generate)

        root.addWidget(actions_card)

        # ----- Generation progress -----
        self.generate_progress = QProgressBar()
        self.generate_progress.setValue(0)
        self.generate_progress.hide()
        root.addWidget(self.generate_progress)

        # ----- Flagged crops (crop QA) -----
        flagged_card = card_frame()
        flagged_layout = QVBoxLayout(flagged_card)
        flagged_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        flagged_layout.setSpacing(SPACE_S)

        flagged_layout.addWidget(section_label("Flagged Crops"))

        self.flagged_list = QListWidget()
        self.flagged_list.setMaximumHeight(130)

        self.btn_keep_flagged = make_button("Keep Selected Flagged Crop(s)", "secondary")
        self.btn_keep_flagged.clicked.connect(self.keep_selected_flagged)

        flagged_layout.addWidget(self.flagged_list)
        flagged_layout.addWidget(self.btn_keep_flagged)
        root.addWidget(flagged_card)

        self.setLayout(root)

    @staticmethod
    def _create_double_spin(minimum, maximum, step, default):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(default)
        return spin

    @staticmethod
    def _create_int_spin(minimum, maximum, default):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        return spin

    # ------------------------------------------------------------------
    # Dataset browsing
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_path):
        if not dataset_path:
            self.image_files = []
            self.current_index = 0
            return

        self.dataset_path = dataset_path

        self.image_files = []
        for extension in IMAGE_EXTENSIONS:
            self.image_files.extend(Path(dataset_path).glob(extension))
        self.image_files.sort()

        print(f"Loaded dataset: {dataset_path} ({len(self.image_files)} images)")

        # Load persisted overrides and ROI defaults (fixes memory-only
        # overrides: saved overrides now drive the preview again).
        recipe = self._current_recipe()
        if recipe:
            self.roi_overrides = dict(recipe.get("roi_overrides") or {})
            stored_default = recipe.get("roi_default")
            if isinstance(stored_default, dict) and stored_default:
                self.apply_settings(stored_default)

        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.image_files:
            return

        self.current_image_path = str(self.image_files[self.current_index])
        self.load_override(self.current_image_path)
        self.load_image(self.current_image_path)

        self.image_index_label.setText(
            f"{self.current_index + 1} / {len(self.image_files)}"
        )

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.original_preview.setPixmap(
            pixmap.scaled(self.original_preview.size(), Qt.KeepAspectRatio)
        )

        roi = self.roi_service.detect_roi(image_path)
        if roi is None:
            return

        self.roi_data = roi
        self.show_roi_preview(image_path, roi)

        self.confidence_label.setText(f"Confidence : {roi['confidence']:.3f}")
        self.size_label.setText(f"Size : {roi['width']:.0f} x {roi['height']:.0f}")

    # ------------------------------------------------------------------
    # ROI preview
    # ------------------------------------------------------------------

    def show_roi_preview(self, image_path, roi):
        image = cv2.imread(image_path)
        if image is None:
            return

        if not roi.get("points"):
            return
        preview = crop_yolo_obb(image, roi["points"])

        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)

        self.roi_preview.setPixmap(
            pixmap.scaled(self.roi_preview.size(), Qt.KeepAspectRatio)
        )

    def refresh_preview(self):
        if self.current_image_path and self.roi_data:
            self.show_roi_preview(self.current_image_path, self.roi_data)

    # ------------------------------------------------------------------
    # ROI settings persistence
    # ------------------------------------------------------------------

    def current_settings(self):
        """Return the current spin-box values as a settings dict."""
        return {
            "width_scale": self.width_scale_spin.value(),
            "height_scale": self.height_scale_spin.value(),
            "padding": self.padding_spin.value(),
            "angle_offset": self.angle_offset_spin.value(),
        }

    def apply_settings(self, settings):
        self.width_scale_spin.setValue(settings.get("width_scale", DEFAULT_WIDTH_SCALE))
        self.height_scale_spin.setValue(
            settings.get("height_scale", DEFAULT_HEIGHT_SCALE)
        )
        self.padding_spin.setValue(settings.get("padding", DEFAULT_PADDING))
        self.angle_offset_spin.setValue(
            settings.get("angle_offset", DEFAULT_ANGLE_OFFSET)
        )

    def save_override(self):
        """Store per-image ROI settings in the recipe."""
        if not self.current_image_path:
            return

        recipe = self._current_recipe()
        if recipe is None:
            return

        image_name = Path(self.current_image_path).name
        self.roi_overrides[image_name] = self.current_settings()
        recipe["roi_overrides"] = self.roi_overrides

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
        print(f"Override saved for {image_name}")

    def load_override(self, image_path):
        """Apply the stored override for an image, or the recipe default."""
        image_name = Path(image_path).name
        settings = self.roi_overrides.get(image_name)

        if settings is None:
            recipe = self._current_recipe()
            stored_default = (recipe or {}).get("roi_default")
            if isinstance(stored_default, dict) and stored_default:
                self.apply_settings(stored_default)
            else:
                self.apply_settings({})
            return

        self.apply_settings(settings)

    def save_default(self):
        """Store the current settings as the recipe-wide default."""
        recipe = self._current_recipe()
        if recipe is None:
            return

        recipe["roi_default"] = self.current_settings()
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        if self.parent_window:
            self.parent_window.current_recipe_data = recipe

    def _current_recipe(self):
        if self.parent_window is None:
            return None
        return self.parent_window.current_recipe_data

    # ------------------------------------------------------------------
    # Auto-tune (percentile-based defaults from detected package sizes)
    # ------------------------------------------------------------------

    def start_auto_tune(self):
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "Load a dataset first.")
            return

        if self.is_autotune_running():
            return

        self.btn_autotune.setEnabled(False)
        self.autotune_worker = AutoTuneWorker(
            [str(path) for path in self.image_files]
        )
        self.autotune_worker.finished_signal.connect(self.on_autotune_finished)
        self.autotune_worker.error_signal.connect(self.on_autotune_error)
        self.autotune_worker.start()

    def on_autotune_finished(self, plan):
        self.btn_autotune.setEnabled(True)

        if not plan:
            QMessageBox.information(
                self,
                "Auto-tune",
                "Auto-tune needs at least 3 detected packages in the "
                "dataset. Verify the detection model and image quality.",
            )
            return

        self.width_scale_spin.setValue(plan["width_scale"])
        self.height_scale_spin.setValue(plan["height_scale"])
        self.padding_spin.setValue(plan["padding"])
        self.angle_offset_spin.setValue(plan["angle_offset"])

        recipe = self._current_recipe()
        if recipe:
            recipe["roi_default"] = self.current_settings()
            recipe["crop_qa"] = {
                "width_tolerance": plan["width_tolerance"],
                "height_tolerance": plan["height_tolerance"],
                "ratio_tolerance": plan["ratio_tolerance"],
            }
            self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

            if self.parent_window and hasattr(
                self.parent_window, "refresh_step_bar"
            ):
                self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Auto-tune",
            "Defaults updated from detected package sizes "
            f"(padding {plan['padding']}px, QA tolerances "
            f"w {plan['width_tolerance']:.2f} / "
            f"h {plan['height_tolerance']:.2f} / "
            f"ratio {plan['ratio_tolerance']:.2f}).",
        )

    def on_autotune_error(self, error_message):
        self.btn_autotune.setEnabled(True)
        QMessageBox.critical(self, "Auto-tune Error", error_message)

    def is_autotune_running(self):
        """Return True while the auto-tune worker thread is active."""
        return self.autotune_worker is not None and self.autotune_worker.isRunning()

    # ------------------------------------------------------------------
    # Training dataset generation
    # ------------------------------------------------------------------

    def generate_training_dataset(self):
        """Crop, QA-classify and standardise every image into a training set."""
        recipe = self._current_recipe()
        if recipe is None:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        if self.crop_worker is not None and self.crop_worker.isRunning():
            return

        default_config = recipe.get("roi_default") or {}
        overrides = recipe.get("roi_overrides") or {}

        qa = recipe.get("crop_qa") or {}
        tolerances = {
            "width_tolerance": qa.get("width_tolerance", WIDTH_TOLERANCE),
            "height_tolerance": qa.get("height_tolerance", HEIGHT_TOLERANCE),
            "ratio_tolerance": qa.get("ratio_tolerance", RATIO_TOLERANCE),
        }

        dataset_root = RECIPES_DIR / recipe["recipe_name"] / "patchcore_dataset"

        # recipe_name is stored data; refuse any path that escapes RECIPES_DIR
        # so a crafted name can never delete or create directories outside it.
        recipes_root = RECIPES_DIR.resolve()
        resolved_root = dataset_root.resolve()
        if recipes_root not in resolved_root.parents:
            print(f"[ROI] Refusing unsafe dataset path: {resolved_root}")
            QMessageBox.warning(
                self,
                "Warning",
                "The recipe name resolves to an unsafe dataset path. "
                "Generation aborted.",
            )
            return

        output_dir = dataset_root / "train" / "good"

        # Wipe previous output so stale crops can never contaminate retraining.
        if dataset_root.exists():
            shutil.rmtree(dataset_root, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.btn_generate.setEnabled(False)
        self.generate_progress.setValue(0)
        self.generate_progress.show()

        self.crop_worker = CropWorker(
            [str(path) for path in self.image_files],
            default_config,
            overrides,
            tolerances,
            output_dir,
            max_flagged_crops=MAX_FLAGGED_IN_MEMORY,
        )
        self.crop_worker.progress.connect(self.generate_progress.setValue)
        self.crop_worker.finished_signal.connect(self.on_generate_finished)
        self.crop_worker.error_signal.connect(self.on_generate_error)
        self.crop_worker.finished.connect(self.on_generate_thread_finished)
        self.crop_worker.start()

    def on_generate_finished(self, result):
        self.generate_progress.hide()
        self.btn_generate.setEnabled(True)

        if not result.get("valid"):
            QMessageBox.warning(
                self, "Warning", "No valid ROIs were found in the dataset."
            )
            return

        recipe = self._current_recipe()
        if recipe is None:
            return

        saved_count = result["saved_count"]
        flagged = result["flagged"]
        avg_width = result["avg_width"]
        avg_height = result["avg_height"]
        target_width = result["target_width"]
        target_height = result["target_height"]
        bounds = result["bounds"]
        split_counts = result.get("split_counts", {"train": saved_count})

        self._good_dir = Path(result["output_dir"])
        self._target_width = target_width
        self._target_height = target_height
        self._flagged_crops = flagged[:MAX_FLAGGED_IN_MEMORY]
        if len(flagged) > MAX_FLAGGED_IN_MEMORY:
            print(
                f"[FLAG] {len(flagged) - MAX_FLAGGED_IN_MEMORY} additional "
                "flagged crops not held in memory"
            )
        self._rebuild_flagged_list()

        print(f"Saved: {saved_count}, flagged: {len(flagged)}")

        recipe["roi_statistics"] = {
            "avg_width": avg_width,
            "avg_height": avg_height,
            "min_width": bounds["min_width"],
            "max_width": bounds["max_width"],
            "min_height": bounds["min_height"],
            "max_height": bounds["max_height"],
            "avg_area": avg_width * avg_height,
            "avg_ratio": avg_width / avg_height,
            "target_width": target_width,
            "target_height": target_height,
            "saved_count": saved_count,
            "rejected_count": len(flagged),
            "split_counts": split_counts,
            "split_manifest": relativize_recipe_path(result.get("split_manifest")),
        }
        recipe["dataset_prepared"] = True
        recipe["preprocessing_version"] = "inspection_obb_v1"
        from services.validation_service import invalidate_validation
        invalidate_validation(recipe)
        if recipe.get('model'):
            recipe['model']['trained'] = False
        recipe["prepared_dataset_path"] = relativize_recipe_path(
            RECIPES_DIR / recipe["recipe_name"] / "patchcore_dataset"
        )

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        if self.parent_window and hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Success",
            f"Training dataset generated. Saved {saved_count}, "
            f"flagged {len(flagged)} (review them in the Flagged Crops list).",
        )

    def on_generate_error(self, error_message):
        self.generate_progress.hide()
        self.btn_generate.setEnabled(True)
        QMessageBox.critical(self, "Generation Error", error_message)

    def on_generate_thread_finished(self):
        if self.crop_worker is not None:
            self.crop_worker.deleteLater()
            self.crop_worker = None

    # ------------------------------------------------------------------
    # Flagged crop review
    # ------------------------------------------------------------------

    def _rebuild_flagged_list(self):
        self.flagged_list.clear()
        shown_any = False
        for index, entry in enumerate(self._flagged_crops):
            if entry is None:
                continue
            item = QListWidgetItem(
                f"{entry['image_name']} — {'; '.join(entry['reasons'])}"
            )
            item.setData(Qt.UserRole, index)
            self.flagged_list.addItem(item)
            shown_any = True

        if not shown_any:
            self.flagged_list.addItem("No flagged crops.")

    def keep_selected_flagged(self):
        """Save the selected flagged crops into the training set anyway."""
        if self._good_dir is None:
            QMessageBox.warning(
                self, "Warning", "Generate the training dataset first."
            )
            return

        selected = self.flagged_list.selectedItems()
        if not selected:
            return

        recipe = self._current_recipe()
        kept = 0

        for list_item in selected:
            index = list_item.data(Qt.UserRole)
            if index is None:
                continue
            entry = self._flagged_crops[index]
            if entry is None:
                continue

            standardized = letterbox(
                entry["crop"], self._target_width, self._target_height
            )
            cv2.imwrite(str(self._good_dir / entry["image_name"]), standardized)
            self._flagged_crops[index] = None
            kept += 1

        if kept:
            print(f"[FLAG] kept {kept} flagged crop(s) for training")
            if recipe:
                stats = recipe.get("roi_statistics") or {}
                stats["saved_count"] = int(stats.get("saved_count", 0)) + kept
                stats["rejected_count"] = max(
                    int(stats.get("rejected_count", 0)) - kept, 0
                )
                recipe["roi_statistics"] = stats
                # Manually accepted crops enter only the train split.  Any
                # prior model/validation must not be reused with this changed
                # dataset.
                from services.validation_service import invalidate_validation
                invalidate_validation(recipe)
                if recipe.get("model"):
                    recipe["model"]["trained"] = False
                self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        self._rebuild_flagged_list()
