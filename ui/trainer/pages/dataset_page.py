"""Dataset import page of the trainer: stats, analysis and quality screening."""

import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from services.dataset_service import IMAGE_EXTENSIONS, DatasetService
from services.recipe_service import RecipeService
from services.training_assistant import utc_now_iso
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    SPACE_XS,
    caption_label,
    card_frame,
    make_button,
    section_label,
    title_label,
)
from ui.workers.screening_worker import ScreeningWorker
from ui.workers.stats_worker import StatsWorker

FLAG_LABELS = {
    "corrupt": "unreadable",
    "blurry": "blurry",
    "brightness": "brightness outlier",
    "size": "size outlier",
}


class DatasetPage(QWidget):
    """Dataset folder selection, computed analysis and quality screening.

    Dataset size never gates anything — every displayed number is
    informational. Quality flags move files to ``quarantine/`` only after
    explicit user confirmation; nothing is ever deleted.
    """

    def __init__(self):
        super().__init__()

        self.dataset_service = DatasetService()
        self.recipe_service = RecipeService()

        self.analysis_result = None
        self.dataset_path = ""
        self.first_image_path = None
        self.parent_window = None
        self.worker = None
        self.stats_worker = None
        self.stats_generation = 0
        self.retired_workers = []

        # Tracks which flow currently owns the shared progress bar so a
        # finishing worker never hides a bar another flow is using.
        self.progress_owner = None
        self.analysis_generation = 0

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        root.setSpacing(SPACE_M)

        root.addWidget(title_label("Dataset Import"))
        root.addWidget(
            caption_label(
                "Point the trainer at a folder of good package images. "
                "Any dataset size is accepted — the numbers below are "
                "informational."
            )
        )

        # ----- Dataset folder card -----
        folder_card = card_frame()
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        folder_layout.setSpacing(SPACE_S)

        folder_layout.addWidget(section_label("Dataset Folder"))

        self.dataset_path_label = QLabel("No Dataset Selected")
        self.dataset_path_label.setWordWrap(True)
        folder_layout.addWidget(self.dataset_path_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(SPACE_S)

        self.btn_browse = make_button("Browse Dataset")
        self.btn_browse.clicked.connect(self.select_dataset)
        actions_row.addWidget(self.btn_browse)

        self.btn_analyze = make_button("Analyze Dataset", "secondary")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        actions_row.addWidget(self.btn_analyze)
        actions_row.addStretch()

        folder_layout.addLayout(actions_row)
        root.addWidget(folder_card)

        # ----- Summary card -----
        summary_card = card_frame()
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        summary_layout.setSpacing(SPACE_S)

        summary_layout.addWidget(section_label("Dataset Summary"))

        self.image_count_label = QLabel("Images : 0")
        self.resolution_label = QLabel("Resolution : -")
        self.corrupted_label = QLabel("Corrupted : 0")
        self.coverage_label = QLabel("Coverage : not analyzed yet")

        summary_fields = QFormLayout()
        summary_fields.setHorizontalSpacing(SPACE_M)
        summary_fields.setVerticalSpacing(SPACE_XS)
        summary_fields.addRow("Images", self.image_count_label)
        summary_fields.addRow("Resolution", self.resolution_label)
        summary_fields.addRow("Corrupted", self.corrupted_label)
        summary_fields.addRow("Detection coverage", self.coverage_label)
        summary_layout.addLayout(summary_fields)

        root.addWidget(summary_card)

        # ----- Analysis card -----
        analysis_card = card_frame()
        analysis_layout = QVBoxLayout(analysis_card)
        analysis_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        analysis_layout.setSpacing(SPACE_S)

        analysis_layout.addWidget(section_label("Physical Analysis"))
        analysis_layout.addWidget(
            caption_label(
                "Computed by the YOLO detection model over the dataset; "
                "package identity is matched against the model registry."
            )
        )

        self.family_label = QLabel("-")
        self.type_label = QLabel("-")
        self.pin_label = QLabel("-")
        self.size_label = QLabel("-")
        self.aspect_label = QLabel("-")
        self.quality_label = QLabel("-")

        analysis_fields = QFormLayout()
        analysis_fields.setHorizontalSpacing(SPACE_M)
        analysis_fields.setVerticalSpacing(SPACE_XS)
        analysis_fields.addRow("Package Family", self.family_label)
        analysis_fields.addRow("Package Type", self.type_label)
        analysis_fields.addRow("Pin Count", self.pin_label)
        analysis_fields.addRow("Median Package Size", self.size_label)
        analysis_fields.addRow("Aspect Ratio", self.aspect_label)
        analysis_fields.addRow("Quality Flags", self.quality_label)
        analysis_layout.addLayout(analysis_fields)

        root.addWidget(analysis_card)

        # ----- Progress + hint -----
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.hide()
        root.addWidget(self.progress)

        root.addStretch()

        root.addWidget(
            caption_label(
                "Tip: 300+ good images are recommended for best calibration."
            )
        )

        self.setLayout(root)

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------

    def image_paths(self, folder):
        """Return the sorted image file paths inside ``folder``."""
        paths = []
        if folder:
            root = Path(folder)
            if root.exists():
                for extension in IMAGE_EXTENSIONS:
                    paths.extend(root.glob(extension))
        return sorted(paths)

    def select_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if not folder:
            return

        self.dataset_path = folder

        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if recipe:
            recipe["dataset_path"] = folder
            self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
            print(f"Dataset path saved to recipe: {recipe['recipe_name']}")

        self.dataset_path_label.setText(folder)
        self.update_statistics(folder)

    def update_statistics(self, folder):
        """Start the dataset statistics pass in a background worker.

        Scanning opens every image once, which is slow for large
        datasets, so it must never run on the GUI thread. The summary
        card shows ellipses and the progress bar fills while loading.
        """
        self._retire_stats_worker()

        self.stats_generation += 1
        generation = self.stats_generation

        # Reset the display for the newly selected folder.
        self.analysis_result = None
        self.image_count_label.setText("Images : …")
        self.resolution_label.setText("Resolution : …")
        self.corrupted_label.setText("Corrupted : …")
        self.coverage_label.setText("Coverage : not analyzed yet")
        self._display_geometry({})
        self.quality_label.setText("-")
        self.btn_analyze.setEnabled(False)

        self.progress_owner = "stats"
        self.progress.setValue(0)
        self.progress.setFormat("Loading dataset… %p%")
        self.progress.show()

        self.stats_worker = StatsWorker(folder)
        self.stats_worker.progress.connect(self.progress.setValue)
        self.stats_worker.finished_signal.connect(
            lambda result, gen=generation: self.on_stats_finished(result, gen)
        )
        self.stats_worker.error_signal.connect(
            lambda message, gen=generation: self.on_stats_error(message, gen)
        )
        self.stats_worker.start()

    def _retire_stats_worker(self):
        """Detach a still-running stats scan so a new selection can start.

        The old worker keeps running detached: its signals are
        disconnected from the page so late results cannot touch the UI,
        and the reference is kept (with deleteLater on finish) so Qt
        never destroys a running thread.
        """
        worker = self.stats_worker
        if worker is None:
            return
        try:
            worker.disconnect(self)
        except TypeError:
            pass  # No connections to this page.
        worker.finished.connect(self._on_retired_worker_finished)
        self.retired_workers.append(worker)
        self.stats_worker = None

    def _on_retired_worker_finished(self):
        worker = self.sender()
        if worker in self.retired_workers:
            self.retired_workers.remove(worker)
        worker.deleteLater()

    def on_stats_finished(self, result, generation):
        if generation != self.stats_generation:
            return  # A newer folder selection superseded this scan.

        if self.progress_owner == "stats":
            self.progress.hide()
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.progress_owner = None
        self.analysis_result = result

        self.image_count_label.setText(f"Images : {result['image_count']}")
        self.resolution_label.setText(f"Resolution : {result['resolution']}")
        self.corrupted_label.setText(f"Corrupted : {result['corrupted']}")

        image_files = self.image_paths(self.dataset_path)
        self.first_image_path = str(image_files[0]) if image_files else None
        self.btn_analyze.setEnabled(bool(image_files))

        if image_files:
            print(f"First image: {self.first_image_path}")

    def on_stats_error(self, error_message, generation):
        if generation != self.stats_generation:
            return  # A newer folder selection superseded this scan.

        if self.progress_owner == "stats":
            self.progress.hide()
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.progress_owner = None
        QMessageBox.critical(self, "Dataset Error", error_message)

    # ------------------------------------------------------------------
    # Dataset analysis (quality screening + YOLO geometry)
    # ------------------------------------------------------------------

    def start_analysis(self):
        paths = self.image_paths(self.dataset_path)
        if not paths:
            QMessageBox.warning(self, "Warning", "Select a dataset folder first.")
            return

        if self.is_analysis_running():
            return

        self.btn_analyze.setEnabled(False)
        self.btn_browse.setEnabled(False)

        self.analysis_generation += 1
        generation = self.analysis_generation

        self.progress_owner = "analysis"
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.show()

        self.worker = ScreeningWorker(paths)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_signal.connect(
            lambda payload, gen=generation: self.on_analysis_finished(payload, gen)
        )
        self.worker.error_signal.connect(
            lambda message, gen=generation: self.on_analysis_error(message, gen)
        )
        self.worker.start()

    def on_analysis_finished(self, payload, generation):
        if generation != self.analysis_generation:
            return  # The user moved on; results no longer apply.

        self.btn_analyze.setEnabled(True)
        self.btn_browse.setEnabled(True)
        if self.progress_owner == "analysis":
            self.progress.hide()
            self.progress.setValue(0)
            self.progress_owner = None
        self.analysis_result = payload

        geometry = payload.get("geometry") or {}
        quality = payload.get("quality") or {}

        self._display_geometry(geometry)
        self._display_quality(quality, geometry)

        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if recipe:
            flagged = [
                entry
                for entry in quality.get("results", [])
                if entry.get("flags")
            ]
            recipe["dataset_analysis"] = {
                "geometry": geometry,
                "flagged_count": len(flagged),
                "analyzed_at": utc_now_iso(),
            }
            self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

            if hasattr(self.parent_window, "refresh_step_bar"):
                self.parent_window.refresh_step_bar()

        self._propose_quarantine(quality)

    def on_analysis_error(self, error_message, generation):
        if generation != self.analysis_generation:
            return  # The user moved on; results no longer apply.

        self.btn_analyze.setEnabled(True)
        self.btn_browse.setEnabled(True)
        if self.progress_owner == "analysis":
            self.progress.hide()
            self.progress.setValue(0)
            self.progress_owner = None
        QMessageBox.critical(self, "Analysis Error", error_message)

    def is_analysis_running(self):
        """Return True while the screening worker thread is active."""
        return self.worker is not None and self.worker.isRunning()

    # ------------------------------------------------------------------
    # Analysis display
    # ------------------------------------------------------------------

    def _display_geometry(self, geometry):
        """Fill the Physical Analysis card from computed results."""
        match = geometry.get("registry_match") or {}
        dominant = geometry.get("dominant_class")

        family = match.get("package_family") or dominant or "-"
        package_type = match.get("package_type") or "-"
        pin_count = match.get("pin_count")
        pin_text = str(pin_count) if pin_count else "-"

        size_stats = geometry.get("size_stats") or {}
        if size_stats:
            size_text = (
                f"{size_stats['median_width']:.0f} x "
                f"{size_stats['median_height']:.0f} px"
            )
        else:
            size_text = "-"

        aspect = geometry.get("median_aspect_ratio")
        aspect_text = f"{aspect:.2f}" if aspect else "-"

        self.family_label.setText(f"Package Family : {family}")
        self.type_label.setText(f"Package Type : {package_type}")
        self.pin_label.setText(f"Pin Count : {pin_text}")
        self.size_label.setText(f"Median Package Size : {size_text}")
        self.aspect_label.setText(f"Aspect Ratio : {aspect_text}")

    def _display_quality(self, quality, geometry):
        """Show detection coverage and quality-flag counts."""
        analyzed = geometry.get("images_analyzed", 0)
        with_detection = geometry.get("images_with_detection", 0)
        self.coverage_label.setText(
            f"Coverage : {with_detection}/{analyzed} images with detected package"
        )

        flagged = [
            entry for entry in quality.get("results", []) if entry.get("flags")
        ]
        if flagged:
            self.quality_label.setText(
                f"{len(flagged)} image(s) flagged for review"
            )
        else:
            self.quality_label.setText("None")

    def _propose_quarantine(self, quality):
        """Offer to move flagged images to quarantine (never deletes)."""
        flagged = [
            entry for entry in quality.get("results", []) if entry.get("flags")
        ]
        if not flagged:
            return

        lines = []
        for entry in flagged[:15]:
            labels = ", ".join(
                FLAG_LABELS.get(flag, flag) for flag in entry["flags"]
            )
            lines.append(f"{entry['name']} — {labels}")
        if len(flagged) > 15:
            lines.append(f"... and {len(flagged) - 15} more")

        answer = QMessageBox.question(
            self,
            "Flagged Images",
            "The following images were flagged:\n\n"
            + "\n".join(lines)
            + "\n\nMove them to the dataset's quarantine folder?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        quarantine_dir = Path(self.dataset_path) / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for entry in flagged:
            source = Path(entry["path"])
            if not source.exists():
                continue
            destination = quarantine_dir / source.name
            try:
                shutil.move(str(source), str(destination))
                moved += 1
            except OSError as error:
                print(f"[QUARANTINE] failed to move {source.name}: {error}")

        print(f"[QUARANTINE] moved {moved} image(s) to {quarantine_dir}")
        QMessageBox.information(
            self,
            "Quarantine",
            f"Moved {moved} image(s) to {quarantine_dir}.",
        )
