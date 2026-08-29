"""Entry point for the IC inspection overlay application.

The application captures a live source window, locates the IC package with a
YOLO oriented-bounding-box detector, verifies its orientation by matching the
top-mark template and scores surface anomalies with a PatchCore model.
"""

import sys
import time

import cv2
import mss
import numpy as np
import pygetwindow as gw
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from services.patchcore_service import PatchCoreService
from services.recipe_service import RecipeService
from services.state_manager import ProgramState, StateManager
from services.top_mark_service import TopMarkService
from ui.inspection.overlay import OverlayWindow
from ui.inspection.status_widget import StatusWidget
from utils.image_utils import crop_from_obb, letterbox
from utils.resource_path import resource_path

TARGET_WINDOW_TITLE = "PowerPoint"
DETECTION_CONFIDENCE = 0.95
KEEP_DETECTION_SECONDS = 1.5
OBB_SMOOTHING_ALPHA = 0.8
DEFAULT_ANOMALY_THRESHOLD = 0.60
PROCESS_INTERVAL_MS = 700
FALLBACK_TARGET_WIDTH = 1000
FALLBACK_TARGET_HEIGHT = 700


class InspectionApp:
    """Runs the inspection pipeline on a timer and drives the overlay UI."""

    def __init__(self, app, detection_model):
        self.detection_model = detection_model

        self.state_manager = StateManager()
        self.recipe_service = RecipeService()
        self.patchcore_service = PatchCoreService()
        self.top_mark_service = TopMarkService()

        self.overlay = OverlayWindow()
        self.status_widget = StatusWidget(self.overlay)
        self.status_widget.recipe_changed.connect(self.on_recipe_changed)

        self.sct = mss.mss()

        self.current_recipe = None
        self.target_width = FALLBACK_TARGET_WIDTH
        self.target_height = FALLBACK_TARGET_HEIGHT
        self.last_boxes = []
        self.last_detect_time = 0.0
        self.last_points = None

        self.state_manager.set_state(ProgramState.READY)

        geometry = app.primaryScreen().availableGeometry()
        self.status_widget.move(
            geometry.width() - self.status_widget.width() - 20,
            20,
        )

        self.overlay.show()
        self.status_widget.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(PROCESS_INTERVAL_MS)

    # ------------------------------------------------------------------
    # Recipe handling
    # ------------------------------------------------------------------

    def on_recipe_changed(self, recipe_data):
        """Load the recipe's PatchCore model and target ROI size."""
        self.current_recipe = recipe_data

        stats = recipe_data.get("roi_statistics", {})
        self.target_width = int(
            stats.get("target_width", stats.get("avg_width", FALLBACK_TARGET_WIDTH))
        )
        self.target_height = int(
            stats.get("target_height", stats.get("avg_height", FALLBACK_TARGET_HEIGHT))
        )

        print(f"[RECIPE] {recipe_data.get('recipe_name', '-')}")

        model_path = recipe_data.get("model", {}).get("path")
        if not model_path:
            print("[PATCHCORE] Recipe has no trained model")
            self.patchcore_service.model = None
            return

        try:
            self.patchcore_service.load_model(model_path)
        except Exception as error:  # Keep the UI running on a broken model.
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Model load error: {error}")

    # ------------------------------------------------------------------
    # Window capture
    # ------------------------------------------------------------------

    def find_window(self):
        """Return the source window, or ``None`` when it is not open."""
        try:
            for window in gw.getAllWindows():
                if window.title and TARGET_WINDOW_TITLE.lower() in window.title.lower():
                    return window
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Find window error: {error}")

        return None

    def grab_frame(self, window):
        """Capture the source window and return it as a BGR image."""
        monitor = {
            "left": int(window.left),
            "top": int(window.top),
            "width": int(window.width),
            "height": int(window.height),
        }

        screenshot = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        self.overlay.set_frame(frame)
        return frame

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame):
        """Run YOLO detection and return ``(result, fps)``."""
        self.state_manager.set_state(ProgramState.DETECTING)

        start_time = time.time()
        results = self.detection_model.predict(
            frame, conf=DETECTION_CONFIDENCE, verbose=False
        )
        elapsed = max(time.time() - start_time, 0.001)

        self.state_manager.set_state(ProgramState.READY)
        return results[0], 1.0 / elapsed

    def parse_boxes(self, result):
        """Convert OBB detections into the box dicts used by the overlay."""
        boxes = []

        if result.obb is None:
            return boxes

        points_list = result.obb.xyxyxyxy.cpu().numpy()
        conf_list = result.obb.conf.cpu().numpy()
        cls_list = result.obb.cls.cpu().numpy()

        label = (
            self.current_recipe["recipe_name"]
            if self.current_recipe is not None
            else "SELECT PACKAGE"
        )

        for points, conf, _cls_id in zip(points_list, conf_list, cls_list):
            boxes.append({"points": points, "conf": float(conf), "label": label})

        return boxes

    def apply_detection_memory(self, boxes):
        """Keep showing the last detection briefly after the target is lost."""
        if boxes:
            self.last_boxes = boxes
            self.last_detect_time = time.time()
            return boxes

        if time.time() - self.last_detect_time < KEEP_DETECTION_SECONDS:
            return self.last_boxes

        return boxes

    def smooth_points(self, points):
        """Exponentially smooth the bounding box across frames."""
        if self.last_points is None:
            self.last_points = points.copy()
            return points

        smoothed = (
            OBB_SMOOTHING_ALPHA * self.last_points
            + (1.0 - OBB_SMOOTHING_ALPHA) * points
        )
        self.last_points = smoothed.copy()
        return smoothed

    # ------------------------------------------------------------------
    # Inspection pipeline
    # ------------------------------------------------------------------

    def process(self):
        """Run one capture-detect-inspect cycle."""
        window = self.find_window()

        if window is None:
            self.state_manager.set_state(ProgramState.WAITING_SOURCE)
            self.overlay.hide()
            self.last_boxes = []
            self.status_widget.update_status(
                system_state=self.state_manager.get_state(),
                inspection_result="-",
                count=0,
                fps=0,
            )
            return

        try:
            if int(window.width) <= 0 or int(window.height) <= 0:
                return

            self.overlay.setGeometry(
                int(window.left),
                int(window.top),
                int(window.width),
                int(window.height),
            )
            frame = self.grab_frame(window)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Capture error: {error}")
            return

        try:
            result, fps = self.detect(frame)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Detection error: {error}")
            return

        try:
            boxes = self.parse_boxes(result)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"OBB parsing error: {error}")
            boxes = []

        boxes = self.apply_detection_memory(boxes)

        count = len(boxes)
        score = 0.0
        fail_reason = ""
        is_upside_down = False

        if count == 0:
            inspection_result = "NOT FOUND"
            self.last_points = None
        elif self.current_recipe is None:
            inspection_result = "SELECT PACKAGE"
        else:
            inspection_result, score, fail_reason, is_upside_down = self.inspect(
                frame, boxes[0]["points"]
            )

        self.status_widget.update_status(
            system_state=self.state_manager.get_state(),
            inspection_result=inspection_result,
            count=count,
            fps=fps,
            score=score,
            fail_reason=fail_reason,
            is_upside_down=is_upside_down,
        )

        self.overlay.show()
        for box in boxes:
            box["score"] = score
            box["result"] = inspection_result
        self.overlay.update_boxes(boxes)

    def inspect(self, frame, points):
        """Inspect one detected package.

        Returns ``(inspection_result, score, fail_reason, is_upside_down)``.
        """
        points = self.smooth_points(points)
        roi = crop_from_obb(frame, points)

        if roi.size == 0:
            return "NOT FOUND", 0.0, "", False

        is_upside_down, angle, orientation_error = self.check_orientation(roi)

        if orientation_error:
            return "FAIL", 0.0, orientation_error, False

        if is_upside_down:
            return "FAIL", 0.0, f"Orientation error ({angle}°)", True

        normalized_roi = letterbox(roi, int(self.target_width), int(self.target_height))
        score = self.patchcore_service.predict(normalized_roi)

        threshold = float(
            self.current_recipe.get("anomaly_threshold", DEFAULT_ANOMALY_THRESHOLD)
            or DEFAULT_ANOMALY_THRESHOLD
        )
        is_defect = score > threshold

        result = "FAIL" if is_defect else "PASS"
        reason = f"Score: {score:.4f} (threshold: {threshold:.4f})"

        print(f"[INSPECTION] {result} score={score:.4f} threshold={threshold:.4f}")

        return result, score, reason, False

    def check_orientation(self, roi):
        """Match the top-mark template and detect an upside-down package.

        Returns ``(is_upside_down, angle, error_message)``. A failed check
        fails closed and reports the error, so a broken template never
        passes a part.
        """
        template_path = self.current_recipe.get("top_mark_template")
        if not template_path:
            return False, 0, ""

        try:
            angle, match_score, score_gap = self.top_mark_service.detect_orientation(
                roi, template_path
            )
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Orientation check error: {error}")
            return False, 0, f"Orientation check failed: {error}"

        print(
            f"[TOPMARK] angle={angle} score={match_score:.4f} gap={score_gap:.4f}"
        )
        return angle != 0, angle, ""


def main():
    app = QApplication(sys.argv)

    print("Loading model...")
    detection_model = YOLO(str(resource_path("models/detection/best.pt")))
    print("Model loaded")

    app.inspection_controller = InspectionApp(app, detection_model)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
