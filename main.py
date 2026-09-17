"""Entry point for the IC inspection overlay application.

The application captures a live source window, locates the IC package with a
YOLO oriented-bounding-box detector, verifies its orientation by matching the
top-mark template and scores surface anomalies with a PatchCore model.

The capture/inspection pipeline runs on a dedicated worker thread so the Qt
GUI thread never blocks on model inference, window capture or file IO.
"""

import math
import sys
import time

import cv2
import mss
import numpy as np
import pygetwindow as gw
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox
from ultralytics import YOLO

from services.patchcore_service import PatchCoreService
from services.recipe_service import (
    RecipeService,
    normalize_anomaly_threshold,
    stored_threshold_usable,
)
from services.state_manager import ProgramState, StateManager
from services.top_mark_service import TopMarkService
from services.marking_service import MarkingService
from services.verdict_service import evaluate_verdict
from ui.inspection.overlay import OverlayWindow
from ui.inspection.status_widget import StatusWidget
from utils.image_utils import crop_yolo_obb, heatmap_overlay, letterbox, letterbox_placement, order_obb_points
from utils.pilot_log import start_session_log
from utils.resource_path import resource_path
from utils.window_capture import grab_window_content

TARGET_WINDOW_TITLE = "PowerPoint"
DETECTION_CONFIDENCE = 0.95
KEEP_DETECTION_SECONDS = 1.5
OBB_SMOOTHING_ALPHA = 0.8
PROCESS_INTERVAL_MS = 300
ORIENTATION_RECHECK_TICKS = 3
ORIENTATION_POSITION_TOLERANCE = 2.0
MAX_INSPECTIONS_PER_FRAME = 8
FALLBACK_TARGET_WIDTH = 1000
FALLBACK_TARGET_HEIGHT = 700

_NO_PENDING_RECIPE = object()


class InspectionWorker(QThread):
    """Runs capture → detect → inspect away from the GUI thread.

    Every result is delivered to the GUI thread through signals; no QWidget
    is ever touched from this thread. A slow cycle simply delays the next
    tick instead of starving the UI event loop.
    """

    overlay_update = Signal(object, object, object, object)
    status_update = Signal(dict)

    def __init__(self, detection_model=None, detection_model_path=None):
        super().__init__()
        self.detection_model = detection_model
        self.detection_model_path = detection_model_path

        self.state_manager = StateManager()
        self.recipe_service = RecipeService()
        self.patchcore_service = PatchCoreService()
        self.top_mark_service = TopMarkService()
        self.marking_service = MarkingService()

        self.pending_recipe = _NO_PENDING_RECIPE
        self.current_recipe = None
        self.target_width = FALLBACK_TARGET_WIDTH
        self.target_height = FALLBACK_TARGET_HEIGHT
        self.last_boxes = []
        self.last_detect_time = 0.0
        self.last_points = None
        self.orientation_state = None
        self.orientation_tick = 0
        self.orientation_dirty = False

        self.sct = None
        self.last_capture_from_screen = False
        self.capture_only = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def apply_recipe(self, recipe_data):
        """Queue a recipe switch; the model is loaded on this thread."""
        self.pending_recipe = recipe_data

    def set_capture_only(self, enabled):
        self.capture_only = bool(enabled)

    def stop(self):
        self.requestInterruption()

    def reset_scene_memory(self):
        """Forget every per-scene memory after a source or target change.

        The detection memory, the box smoother and the cached orientation all
        describe the PREVIOUS scene. Keeping any of them after the source
        window disappeared would let a stale verdict or a stale crop survive
        into the next frames, so everything is dropped together.
        """
        self.last_boxes = []
        self.last_detect_time = 0.0
        self.last_points = None
        self.orientation_state = None
        self.orientation_tick = 0
        self.orientation_dirty = False

    def run(self):
        """Drive the pipeline in a fixed-interval loop on this thread.

        A plain loop (instead of a QTimer inside the thread) keeps every
        tick on the worker: a QTimer connected to a QThread-subclass method
        would resolve the receiver's affinity to the creating (GUI) thread
        and silently queue the pipeline back onto the UI.
        """
        self.sct = mss.mss()
        if self.detection_model is None and self.detection_model_path:
            try:
                print("Loading detection model...")
                self.detection_model = YOLO(str(self.detection_model_path))
                print("Detection model loaded")
            except Exception as error:
                self.state_manager.set_state(ProgramState.ERROR, str(error))
                self.status_update.emit({
                    "system_state": self.state_manager.get_state(),
                    "inspection_result": "NO SOURCE",
                    "count": 0,
                    "fps": 0,
                    "error_message": f"Detection model load failed: {error}",
                })
                return
        self.state_manager.set_state(ProgramState.READY)

        interval = PROCESS_INTERVAL_MS / 1000.0
        next_deadline = time.time() + interval

        while not self.isInterruptionRequested():
            now = time.time()
            if now >= next_deadline:
                try:
                    self.tick()
                except Exception as error:  # A tick must never kill the worker.
                    print(f"Pipeline tick error: {error}")
                    self.state_manager.set_state(ProgramState.ERROR, str(error))
                next_deadline = time.time() + interval
            else:
                time.sleep(min(0.02, max(next_deadline - now, 0.001)))

    # ------------------------------------------------------------------
    # Recipe handling
    # ------------------------------------------------------------------

    def _load_pending_recipe(self):
        if self.pending_recipe is _NO_PENDING_RECIPE:
            return

        recipe_data, self.pending_recipe = self.pending_recipe, _NO_PENDING_RECIPE
        self.current_recipe = recipe_data
        self.orientation_state = None

        if recipe_data is None:
            print("[RECIPE] Cleared - no recipe selected")
            self.patchcore_service.model = None
            self.target_width = FALLBACK_TARGET_WIDTH
            self.target_height = FALLBACK_TARGET_HEIGHT
            return

        stats = recipe_data.get("roi_statistics", {})
        self.target_width = int(
            stats.get("target_width", stats.get("avg_width", FALLBACK_TARGET_WIDTH))
        )
        self.target_height = int(
            stats.get("target_height", stats.get("avg_height", FALLBACK_TARGET_HEIGHT))
        )

        print(f"[RECIPE] {recipe_data.get('recipe_name', '-')}")

        self.status_update.emit({
            "system_state": self.state_manager.get_state(),
            "inspection_result": "SELECT PACKAGE",
            "count": 0,
            "fps": 0,
            "fail_reason": "Loading recipe model...",
        })

        model_path = recipe_data.get("model", {}).get("path")
        if not model_path:
            print("[PATCHCORE] Recipe has no trained model")
            self.patchcore_service.model = None
            return

        try:
            self.patchcore_service.load_model(model_path)
        except Exception as error:
            # Fail closed: never keep scoring with the PREVIOUS recipe's
            # model under the new recipe's name.
            self.patchcore_service.model = None
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
        """Capture the source window content and return ``(frame, rect)``.

        Prefers ``PrintWindow`` so the capture contains only the source
        window's own content: overlapping windows — the app's overlay, the
        status panel, popups — are excluded, and the pipeline can never
        re-inspect its own drawn output. Falls back to an mss region grab
        when window-content capture is unavailable.
        """
        frame, rect = grab_window_content(getattr(window, "_hWnd", None))
        if frame is not None:
            self.last_capture_from_screen = False
            return frame, rect

        left, top = int(window.left), int(window.top)
        width, height = int(window.width), int(window.height)
        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        screenshot = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        # The screen-region grab composites everything drawn on top of the
        # source window — including this app's own boxes — into the frame.
        self.last_capture_from_screen = True
        return frame, (left, top, width, height)

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
        """Keep showing the last detections briefly after the target is lost.

        Fresh detections replace the memory. During the keep-alive window the
        stored detections are returned flagged as ``stale`` so the caller can
        hold their last verdicts without re-inspecting them. After the window
        the memory is cleared.
        """
        if boxes:
            self.last_boxes = boxes
            self.last_detect_time = time.time()
            return boxes

        if time.time() - self.last_detect_time < KEEP_DETECTION_SECONDS:
            for box in self.last_boxes:
                box["stale"] = True
            return self.last_boxes

        self.last_boxes = []
        return []

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

    def tick(self):
        """Run one capture-detect-inspect cycle on the worker thread."""
        self._load_pending_recipe()

        window = self.find_window()

        # A minimized source cannot be captured reliably; treat it like a
        # missing source so the overlay and verdicts reset instead of
        # drawing stale boxes over other windows.
        if window is None or getattr(window, "isMinimized", False):
            self.state_manager.set_state(ProgramState.WAITING_SOURCE)
            self.overlay_update.emit(None, None, [], None)
            self.reset_scene_memory()
            self.status_update.emit(
                {
                    "system_state": self.state_manager.get_state(),
                    "inspection_result": "NO SOURCE",
                    "count": 0,
                    "fps": 0,
                }
            )
            return

        try:
            frame, rect = self.grab_frame(window)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Capture error: {error}")
            self.status_update.emit(
                {
                    "system_state": self.state_manager.get_state(),
                    "inspection_result": "NO SOURCE",
                    "count": 0,
                    "fps": 0,
                    "error_message": str(error),
                }
            )
            self.overlay_update.emit(None, None, [], None)
            return

        # A source has been found. Publish this immediately so the operator
        # does not wait for the first YOLO warm-up to see the correct state.
        if self.current_recipe is None:
            self.status_update.emit({
                "system_state": self.state_manager.get_state(),
                "inspection_result": "SELECT PACKAGE",
                "count": 0,
                "fps": 0,
            })

        if self.capture_only:
            self.overlay_update.emit(frame, rect, [], getattr(window, "_hWnd", None))
            return

        try:
            result, fps = self.detect(frame)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Detection error: {error}")
            self.status_update.emit(
                {
                    "system_state": self.state_manager.get_state(),
                    "inspection_result": "NO SOURCE",
                    "count": 0,
                    "fps": 0,
                    "error_message": str(error),
                }
            )
            return

        try:
            boxes = self.parse_boxes(result)
        except Exception as error:
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"OBB parsing error: {error}")
            boxes = []

        boxes = self.apply_detection_memory(boxes)
        holding = any(box.get("stale") for box in boxes)

        count = len(boxes)
        score = 0.0
        fail_reason = ""
        is_upside_down = False
        heatmap = None
        marking = None

        if self.current_recipe is None:
            inspection_result = "SELECT PACKAGE"
            score = 0.0
            fail_reason = "Please Select package"
            for box in boxes:
                box["result"] = "SELECT PACKAGE"

        elif count == 0:
            inspection_result = "NOT FOUND"
            # The part is gone; a new part placed later must be checked fresh.
            self.last_points = None
            self.orientation_state = None
            self.orientation_tick = 0

        elif holding:
            # The detection dropped out; whatever verdict is displayed comes
            # from before the gap. The next fresh detection must re-derive
            # everything, including the cached orientation, because the part
            # may have been swapped while the box was stale.
            self.orientation_dirty = True
            verdict_box = max(boxes, key=lambda box: (
                {"FAIL": 3, "UNKNOWN": 2, "NOT INSPECTED": 2, "PASS": 1}.get(box.get("result"), 0),
                box.get("score", 0.0)))
            inspection_result = verdict_box.get("result", "NOT FOUND")
            if inspection_result in ("UNKNOWN", "NOT INSPECTED"):
                inspection_result = "FAIL"
            score = verdict_box.get("score", 0.0)
            fail_reason = verdict_box.get("fail_reason", "")
            is_upside_down = verdict_box.get("is_upside_down", False)
            heatmap = verdict_box.get("heatmap")
            marking = verdict_box.get("marking")

        else:
            if self.orientation_dirty:
                self.orientation_state = None
                self.orientation_tick = 0
                self.orientation_dirty = False

            for box in sorted(boxes, key=lambda item: item["conf"], reverse=True)[:MAX_INSPECTIONS_PER_FRAME]:
                self._last_roi_shape = None
                self._last_anomaly_map = None
                self._last_marking = {"status": "NOT CHECKED", "score": None}
                try:
                    result, item_score, item_reason, upside_down, item_heatmap = self.inspect(
                        frame, box["points"]
                    )
                except Exception as error:
                    self.state_manager.set_state(ProgramState.ERROR, str(error))
                    print(f"Inspection error: {error}")
                    result, item_score, item_reason, upside_down, item_heatmap = (
                        "FAIL", 0.0, f"Inspection error: {error}", False, None
                    )

                box["score"] = item_score
                box["result"] = result
                box["fail_reason"] = item_reason
                box["is_upside_down"] = upside_down
                box["marking"] = dict(self._last_marking, object=next(
                    i for i, candidate in enumerate(boxes, 1) if candidate is box))
                box["heatmap"] = None if self.last_capture_from_screen else item_heatmap
                box["segments"] = self._segment_contours(
                    box["points"],
                    getattr(self, "_last_roi_shape", None),
                    getattr(self, "_last_anomaly_map", None),
                    self.target_width,
                    self.target_height,
                )

            for box in sorted(boxes, key=lambda item: item["conf"], reverse=True)[MAX_INSPECTIONS_PER_FRAME:]:
                box["result"] = "NOT INSPECTED"
                box["score"] = 0.0
                box["fail_reason"] = "Inspection capacity exceeded"

            inspected = [box for box in boxes if "score" in box]
            worst = max(inspected, key=lambda box: box["score"], default=None)
            failed = next((box for box in inspected if box["result"] == "FAIL"), None)
            unknown = next((box for box in inspected if box["result"] in ("UNKNOWN", "NOT INSPECTED")), None)
            summary = failed or unknown or worst
            if failed:
                inspection_result = "FAIL"
            elif unknown:
                inspection_result = "FAIL"
            else:
                inspection_result = "PASS" if inspected else "NOT FOUND"
            if summary is not None:
                score = summary["score"]
                fail_reason = summary.get("fail_reason", "")
                is_upside_down = summary.get("is_upside_down", False)
                heatmap = summary.get("heatmap")
                marking = summary.get("marking")

        self.overlay_update.emit(frame, rect, boxes, getattr(window, "_hWnd", None))
        self.status_update.emit(
            {
                "system_state": self.state_manager.get_state(),
                "inspection_result": inspection_result,
                "count": count,
                "fps": fps,
                "score": score,
                "fail_reason": fail_reason,
                "is_upside_down": is_upside_down,
                "heatmap": heatmap,
                "marking": marking,
                "error_message": self.state_manager.get_message(),
            }
        )

    @staticmethod
    def _segment_contours(points, crop_shape, anomaly_map, target_width, target_height):
        """Map filtered PatchCore regions back onto one correctly ordered OBB."""
        if crop_shape is None:
            return []
        if anomaly_map is None:
            return []
        amap = np.asarray(anomaly_map)
        while amap.ndim > 2:
            amap = amap[0]
        crop_height, crop_width = crop_shape[:2]
        placement = letterbox_placement(
            crop_width, crop_height, target_width, target_height
        )
        x1 = placement["x_offset"]
        y1 = placement["y_offset"]
        x2 = x1 + placement["width"]
        y2 = y1 + placement["height"]
        if x2 <= x1 or y2 <= y1:
            return []

        # Remove the black letterbox area before thresholding, otherwise its
        # artificial border can become a false anomaly contour.
        map_target = cv2.resize(amap, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        map_crop = cv2.resize(map_target[y1:y2, x1:x2], (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
        finite = map_crop[np.isfinite(map_crop)]
        if finite.size == 0 or float(finite.max()) <= float(finite.min()):
            return []
        threshold = max(float(np.percentile(finite, 98)), float(finite.mean() + 2.0 * finite.std()))
        mask = (map_crop >= threshold).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        source = np.array([[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]], dtype=np.float32)
        target = order_obb_points(points)
        matrix = cv2.getPerspectiveTransform(source, target)
        segments = []
        for contour in contours:
            if cv2.contourArea(contour) < max(6.0, crop_width * crop_height * 0.0002):
                continue
            mapped = cv2.perspectiveTransform(contour.astype(np.float32), matrix)
            segments.append(mapped.reshape(-1, 2).tolist())
        return segments

    def cached_orientation(self, roi, points):
        """Reuse the last orientation result while the package is stationary.

        The template check reruns when the smoothed box moves beyond a small
        tolerance, every few ticks as a fallback, or after the template path
        changes, so a changed scene is still caught quickly.
        """
        self.orientation_tick += 1

        template_path = (
            self.current_recipe.get("top_mark_template")
            if self.current_recipe is not None
            else None
        )

        if self.orientation_state is not None:
            cached_points, is_upside_down, angle, cached_path = self.orientation_state
            moved = not np.allclose(
                points, cached_points, atol=ORIENTATION_POSITION_TOLERANCE
            )
            due = self.orientation_tick >= ORIENTATION_RECHECK_TICKS
            if cached_path == template_path and not moved and not due:
                return is_upside_down, angle, ""

        is_upside_down, angle, orientation_error = self.check_orientation(roi)
        if orientation_error:
            return is_upside_down, angle, orientation_error

        self.orientation_state = (points.copy(), is_upside_down, angle, template_path)
        self.orientation_tick = 0
        return is_upside_down, angle, ""

    @staticmethod
    def _package_roi(points):
        """Convert smoothed OBB corner points into a rotated-ROI dict.

        The dict matches the ``crop_rotated_roi`` input convention used by
        the trainer's crop pipeline (``cx``, ``cy``, ``width``, ``height``,
        ``angle`` in radians, width >= height, angle folded into
        ``(-45, 45]``), so live inspection crops the package the same way the
        training/calibration crops were produced.
        """
        (center, size, theta) = cv2.minAreaRect(points.astype(np.float32))
        width, height = float(size[0]), float(size[1])

        if width < height:
            width, height = height, width
            theta += 90.0

        theta = (theta + 45.0) % 90.0 - 45.0

        return {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "width": width,
            "height": height,
            "angle": math.radians(theta),
        }

    def inspect(self, frame, points):
        """Inspect one detected package.

        Returns ``(inspection_result, score, fail_reason, is_upside_down,
        heatmap)`` with the PatchCore anomaly heatmap as a BGR image or
        ``None`` when no heatmap was produced.
        """
        points = points.copy()

        raw_threshold = self.current_recipe.get("anomaly_threshold")
        if self.current_recipe.get("calibration") and not stored_threshold_usable(
            raw_threshold
        ):
            # A calibrated recipe with a corrupted threshold must never fall
            # back to the default gate: that silently loosens a measured
            # limit. Fail closed and tell the operator to recalibrate.
            return (
                "FAIL",
                0.0,
                f"Invalid anomaly threshold {raw_threshold!r} — recalibrate",
                False,
                None,
            )

        roi = crop_yolo_obb(
            frame,
            points
        )

        self._last_roi_shape = roi.shape

        if roi.size == 0:
            return "NOT FOUND", 0.0, "", False, None

        is_upside_down, angle, orientation_error = self.cached_orientation(roi, points)

        if orientation_error:
            return "FAIL", 0.0, orientation_error, False, None

        if is_upside_down:
            return "FAIL", 0.0, f"Orientation error ({angle}°)", True, None

        laser_status, laser_reason = self.check_laser_mark(roi)
        if laser_status == "FAIL":
            return "FAIL", 0.0, laser_reason, False, None
        if laser_status == "UNKNOWN":
            return "FAIL", 0.0, laser_reason, False, None

        normalized_roi = letterbox(roi, int(self.target_width), int(self.target_height))
        score, anomaly_map = self.patchcore_service.predict_full(normalized_roi)
        self._last_anomaly_map = anomaly_map

        threshold = normalize_anomaly_threshold(raw_threshold)
        is_defect, reason, gate_region_px = evaluate_verdict(
            score, anomaly_map, threshold, self.current_recipe
        )

        # Display the heatmap over the normalized inspection crop.
        display_map = self.patchcore_service.display_anomaly_map(anomaly_map)

        try:
            heatmap = heatmap_overlay(
                normalized_roi,
                anomaly_map,
                opacity=0.45
            )

        except Exception as error:
            print(f"[HEATMAP] box-aligned display failed: {error}")
            heatmap = heatmap_overlay(normalized_roi, display_map)

        result = "FAIL" if is_defect else "PASS"

        gate_info = f" gate_region={gate_region_px}px" if gate_region_px else ""
        print(
            f"[INSPECTION] {result} score={score:.4f} "
            f"threshold={threshold:.4f}{gate_info}"
        )

        return result, score, reason, False, heatmap

    def check_laser_mark(self, roi):
        self._last_marking = self.marking_service.check(roi, self.current_recipe)
        status = self._last_marking["status"]
        verdict = "MATCH" if status in ("MATCH", "DISABLED") else "FAIL"
        return verdict, self._last_marking.get("reason", "")

    def check_orientation(self, roi):
        """Match the top-mark template and detect an upside-down package.

        Returns ``(is_upside_down, angle, error_message)``. Missing templates
        and template-loading errors still fail closed; the match itself is
        used as-is, so the verdict follows the best-scoring orientation
        angle.
        """
        template_path = self.current_recipe.get("top_mark_template")
        if not template_path:
            return False, 0, (
                "No top-mark template — capture it with the crop button"
            )

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


class InspectionApp(QObject):
    """Owns the Qt widgets and applies worker results on the GUI thread.

    Deriving from QObject gives the connected slots GUI-thread affinity, so
    the worker's signals are queued into the main event loop instead of
    running on the worker thread.
    """

    def __init__(self, app, detection_model=None, detection_model_path=None):
        super().__init__()
        self.overlay = OverlayWindow()
        self.status_widget = StatusWidget(self.overlay)

        self.worker = InspectionWorker(detection_model, detection_model_path)
        self.worker.overlay_update.connect(self._apply_overlay)
        self.worker.status_update.connect(self._apply_status)
        self.status_widget.recipe_changed.connect(self.worker.apply_recipe)
        self.status_widget.capture_mode_changed.connect(self.worker.set_capture_only)

        geometry = app.primaryScreen().availableGeometry()
        self.status_widget.move(
            geometry.width() - self.status_widget.width() - 20,
            20,
        )

        app.aboutToQuit.connect(self._shutdown)

        self.overlay.show()
        self.status_widget.show()

        self.worker.start()

    @Slot(object, object, object, object)
    def _apply_overlay(self, frame, rect, boxes, hwnd):
        """Apply one overlay frame on the GUI thread.

        ``hwnd`` is the source window's native handle: the overlay is
        re-inserted directly above it on every update, so windows the user
        opens over the source cover the overlay as well.
        """
        if self.overlay.roi_mode:
            return
        if frame is None or rect is None:
            self.overlay.hide()
            return

        left, top, width, height = rect
        self.overlay.setGeometry(int(left), int(top), int(width), int(height))
        if self.status_widget.capture_mode:
            # The full-frame copy is only needed by the independent Capture
            # tab. Avoid copying every camera frame during normal inspection.
            self.status_widget.set_capture_frame(frame)
            self.overlay.set_capture_frame(frame)
            self.overlay.update_boxes([])
            self.overlay.show()
            self.overlay.sync_above_window(hwnd)
            return
        self.overlay.set_frame(frame)
        self.overlay.update_boxes(boxes)
        self.overlay.show()
        self.overlay.sync_above_window(hwnd)

    @Slot(dict)
    def _apply_status(self, payload):
        self.status_widget.update_status(**payload)

    def _shutdown(self):
        self.worker.stop()
        self.worker.wait(3000)


def main():
    start_session_log()

    app = QApplication(sys.argv)

    model_path = resource_path("models/detection/best.pt")
    if not model_path.exists():
        print(f"Detection model not found: {model_path}")
        QMessageBox.critical(
            None,
            "VisionAI",
            f"Detection model not found:\n{model_path}",
        )
        sys.exit(1)

    # Construct the UI immediately. The detector is loaded on the worker
    # after the window is visible, so startup no longer looks frozen.
    app.inspection_controller = InspectionApp(
        app, detection_model_path=model_path
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
