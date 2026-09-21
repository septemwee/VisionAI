"""Entry point for the IC inspection overlay application.

The application captures a live source window, locates the IC package with a
YOLO oriented-bounding-box detector, verifies its orientation by matching the
top-mark template and scores surface anomalies with a PatchCore model.

The capture/inspection pipeline runs on a dedicated worker thread so the Qt
GUI thread never blocks on model inference, window capture or file IO.
"""

import math
import os
import sys
import time
import threading

import cv2
import mss
import numpy as np
import pygetwindow as gw
from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from ultralytics import YOLO

from services.patchcore_service import PatchCoreService
from services.latest_frame import LatestFrame
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

TARGET_WINDOW_TITLE = os.environ.get("VISIONAI_SOURCE_TITLE", "PowerPoint")
DETECTION_CONFIDENCE = 0.95
DETECTION_IMAGE_SIZE = 640
DETECTION_MAX_OBJECTS = 32
KEEP_DETECTION_SECONDS = 1.5
OBB_SMOOTHING_ALPHA = 0.8
PROCESS_INTERVAL_MS = 300
ORIENTATION_RECHECK_TICKS = 3
ORIENTATION_POSITION_TOLERANCE = 2.0
MAX_INSPECTIONS_PER_FRAME = 8
FALLBACK_TARGET_WIDTH = 1000
FALLBACK_TARGET_HEIGHT = 700
INSPECTION_TRACE = os.environ.get("VISIONAI_INSPECTION_TRACE", "0") == "1"
OVERLAY_Z_ORDER_INTERVAL_SECONDS = 0.5

_NO_PENDING_RECIPE = object()


class InspectionWorker(QThread):
    """Inspect the latest detection while an independent producer runs YOLO."""

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
        self.active_recipe_request_id = 0
        self.current_recipe = None
        self.target_width = FALLBACK_TARGET_WIDTH
        self.target_height = FALLBACK_TARGET_HEIGHT
        self.last_boxes = []
        self.last_detect_time = 0.0
        self.last_points = None
        self.orientation_state = None
        self.orientation_tick = 0
        self.orientation_states = {}
        self.orientation_ticks = {}
        self.orientation_dirty = False
        self._last_normalized_roi = None

        self.sct = None
        self.last_capture_from_screen = False
        self.capture_only = False
        self._source_window = None
        self.live_frames = LatestFrame()
        self.completed_frames = LatestFrame()
        self.pipeline_epoch = 0
        self._inspection_source = None
        self._last_perf_log = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def apply_recipe(self, recipe_data, request_id=0):
        """Queue a recipe switch; the model is loaded on this thread."""
        self.pending_recipe = (recipe_data, int(request_id))
        self.pipeline_epoch += 1

    def set_capture_only(self, enabled):
        self.capture_only = bool(enabled)
        self.pipeline_epoch += 1

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
        self.orientation_states.clear()
        self.orientation_ticks.clear()
        self.orientation_dirty = False

    def run(self):
        """Consume the latest available capture on this thread.

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

        producer = threading.Thread(target=self._detect_latest, name="live-detection")
        producer.start()
        consumed = 0
        try:
            while not self.isInterruptionRequested():
                version, packet = self.live_frames.read()
                if packet is None or version == consumed or self.capture_only:
                    self.msleep(10)
                    continue
                consumed = version
                if packet["epoch"] != self.pipeline_epoch:
                    continue
                try:
                    inspection_started = time.monotonic()
                    self.tick(packet)
                    finished = time.monotonic()
                    if finished - self._last_perf_log >= 5.0:
                        print(f"[PERF] capture_yolo_ms={packet.get('detection_ms', 0):.1f} "
                              f"inspection_ms={(finished-inspection_started)*1000:.1f} "
                              f"sample_age_ms={(finished-packet['captured_at'])*1000:.1f}")
                        self._last_perf_log = finished
                except Exception as error:
                    print(f"Pipeline tick error: {error}")
                    self.state_manager.set_state(ProgramState.ERROR, str(error))
        finally:
            self.requestInterruption()
            producer.join()
            self.sct.close()

    def _detect_latest(self):
        """YOLO has one owner; publish only the newest capture, never a queue."""
        detector = InspectionWorker(self.detection_model)
        detector.sct = mss.mss()
        source_id = 0
        previous_source = None
        try:
            while not self.isInterruptionRequested():
                if self.capture_only:
                    previous_source = None
                    time.sleep(0.05)
                    continue
                started = time.monotonic()
                epoch = self.pipeline_epoch
                window = None
                frame = rect = result = None
                fps = 0.0
                error = ""
                boxes = []
                try:
                    window = detector.find_window()
                    if window is not None and not getattr(window, "isMinimized", False):
                        frame, rect = detector.grab_frame(window)
                        result, fps = detector.detect(frame)
                        boxes = detector.parse_boxes(result)
                    else:
                        window = None
                except Exception as exc:
                    error = str(exc)
                    window = None
                source = (getattr(window, "_hWnd", None), rect, bool(boxes)) if window else None
                if source != previous_source:
                    source_id += 1
                    previous_source = source
                if epoch == self.pipeline_epoch:
                    self.live_frames.put(dict(
                        epoch=epoch, source_id=source_id, window=window,
                        frame=frame, rect=rect, result=result, fps=fps,
                        boxes=boxes,
                        screen=detector.last_capture_from_screen, error=error,
                        captured_at=started,
                        detection_ms=(time.monotonic()-started)*1000,
                    ))
                time.sleep(max(0.001, 0.05 - (time.monotonic() - started)))
        finally:
            detector.sct.close()

    # ------------------------------------------------------------------
    # Recipe handling
    # ------------------------------------------------------------------

    def _load_pending_recipe(self):
        if self.pending_recipe is _NO_PENDING_RECIPE:
            return

        pending, self.pending_recipe = self.pending_recipe, _NO_PENDING_RECIPE
        if isinstance(pending, tuple) and len(pending) == 2:
            recipe_data, request_id = pending
        else:
            recipe_data, request_id = pending, self.active_recipe_request_id + 1
        self.active_recipe_request_id = request_id
        self.orientation_state = None

        if recipe_data is None:
            self.current_recipe = None
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
            "model_loading": True,
            "recipe_request_id": request_id,
        })

        model_path = recipe_data.get("model", {}).get("path")
        if not model_path:
            print("[PATCHCORE] Recipe has no trained model")
            self.current_recipe = recipe_data
            self.patchcore_service.model = None
            self.status_update.emit({
                "system_state": self.state_manager.get_state(),
                "inspection_result": "SELECT PACKAGE",
                "count": 0,
                "fps": 0,
                "model_loading": False,
                "fail_reason": "Recipe model is not configured",
                "recipe_request_id": request_id,
            })
            return

        try:
            self.patchcore_service.load_model(model_path)
            if self.pending_recipe is not _NO_PENDING_RECIPE:
                # A newer selection arrived while this model was loading.
                # Discard this result and let the next tick load only the
                # latest request.
                self.patchcore_service.model = None
                return
            self.current_recipe = recipe_data
            self.status_update.emit({
                "system_state": self.state_manager.get_state(),
                "inspection_result": "READY",
                "count": 0,
                "fps": 0,
                "model_loading": False,
                "recipe_request_id": request_id,
            })
        except Exception as error:
            # Fail closed: never keep scoring with the PREVIOUS recipe's
            # model under the new recipe's name.
            self.patchcore_service.model = None
            self.current_recipe = None
            self.state_manager.set_state(ProgramState.ERROR, str(error))
            print(f"Model load error: {error}")
            self.status_update.emit({
                "system_state": self.state_manager.get_state(),
                "inspection_result": "NO SOURCE",
                "count": 0,
                "fps": 0,
                "model_loading": False,
                "error_message": f"Recipe model load failed: {error}",
                "recipe_request_id": request_id,
            })

    # ------------------------------------------------------------------
    # Window capture
    # ------------------------------------------------------------------

    def find_window(self):
        """Return the source window, or ``None`` when it is not open."""
        source_title = TARGET_WINDOW_TITLE
        cached = self._source_window
        if cached is not None:
            try:
                if cached.title and source_title.lower() in cached.title.lower():
                    if not getattr(cached, "isMinimized", False):
                        return cached
            except Exception:
                pass
            self._source_window = None
        try:
            for window in gw.getAllWindows():
                if window.title and source_title.lower() in window.title.lower():
                    self._source_window = window
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
            frame,
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMAGE_SIZE,
            max_det=DETECTION_MAX_OBJECTS,
            verbose=False,
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

        # Use the current detection for both crop and display. Temporal
        # smoothing here makes the crop trail a moving physical package.

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

    def tick(self, packet=None):
        """Run one capture-detect-inspect cycle on the worker thread."""
        if self.capture_only:
            return
        self._load_pending_recipe()
        if packet is not None:
            if packet["epoch"] != self.pipeline_epoch:
                return
            source = (packet["epoch"], packet["source_id"])
            if source != self._inspection_source:
                self.reset_scene_memory()
                self._inspection_source = source

        window = packet["window"] if packet is not None else self.find_window()

        # A minimized source cannot be captured reliably; treat it like a
        # missing source so the overlay and verdicts reset instead of
        # drawing stale boxes over other windows.
        if window is None or getattr(window, "isMinimized", False):
            if packet is not None:
                self.reset_scene_memory()
                return  # The GUI polls source state independently of inference.
            self._source_window = None
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
            if packet is not None:
                frame, rect = packet["frame"], packet["rect"]
                self.last_capture_from_screen = packet["screen"]
            else:
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
            result, fps = (packet["result"], packet["fps"]) if packet is not None else self.detect(frame)
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

        # Publish one coherent frame with its verdict below. Publishing a
        # DETECTED frame here resets the result colour on every cycle.

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
            self.orientation_states.clear()
            self.orientation_ticks.clear()
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

            for object_index, box in enumerate(
                sorted(boxes, key=lambda item: item["conf"], reverse=True)[:MAX_INSPECTIONS_PER_FRAME]
            ):
                self._last_roi_shape = None
                self._last_anomaly_map = None
                self._last_marking = {"status": "NOT CHECKED", "score": None}
                self._last_normalized_roi = None
                try:
                    try:
                        result, item_score, item_reason, upside_down, item_heatmap = self.inspect(
                            frame, box["points"], object_index, render_heatmap=False
                        )
                    except TypeError as error:
                        # Preserve compatibility with injected legacy
                        # inspectors used by integrations and tests.
                        if "render_heatmap" not in str(error) and "positional" not in str(error):
                            raise
                        try:
                            result, item_score, item_reason, upside_down, item_heatmap = self.inspect(
                                frame, box["points"], object_index
                            )
                        except TypeError as fallback_error:
                            if "positional" not in str(fallback_error):
                                raise
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
                box["_normalized_roi"] = self._last_normalized_roi
                box["_anomaly_map"] = self._last_anomaly_map
                # A PASS has no anomaly segment to show. Skipping this image
                # mapping avoids several large resize/morphology operations
                # on every normal inspection tick.
                box["segments"] = (
                    self._segment_contours(
                        box["points"],
                        getattr(self, "_last_roi_shape", None),
                        getattr(self, "_last_anomaly_map", None),
                        self.target_width,
                        self.target_height,
                    ) if result == "FAIL" else []
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
                if not self.last_capture_from_screen:
                    heatmap = self._summary_heatmap(summary)
                marking = summary.get("marking")

            # Raw crops/maps are worker-only data. Do not retain or queue
            # them with overlay payloads after the status heatmap is built.
            for box in boxes:
                box.pop("_normalized_roi", None)
                box.pop("_anomaly_map", None)

        payload = {
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
                "recipe_request_id": self.active_recipe_request_id,
            }
        if packet is not None:
            self.completed_frames.put(dict(packet, boxes=boxes, payload=payload))
        else:
            self.overlay_update.emit(frame, rect, boxes, getattr(window, "_hWnd", None))
            self.status_update.emit(payload)

    def _summary_heatmap(self, box):
        """Build one status heatmap instead of one per detected package."""
        roi = box.get("_normalized_roi")
        anomaly_map = box.get("_anomaly_map")
        if roi is None or anomaly_map is None:
            return None
        try:
            return heatmap_overlay(roi, anomaly_map, opacity=0.45)
        except Exception as error:
            if INSPECTION_TRACE:
                print(f"[HEATMAP] status rendering failed: {error}")
            return None

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

    def cached_orientation(self, roi, points, object_key=None):
        """Reuse the last orientation result while the package is stationary.

        The template check reruns when the smoothed box moves beyond a small
        tolerance, every few ticks as a fallback, or after the template path
        changes, so a changed scene is still caught quickly.
        """
        if object_key is None:
            object_key = 0
        tick = self.orientation_ticks.get(object_key, 0) + 1

        template_path = (
            self.current_recipe.get("top_mark_template")
            if self.current_recipe is not None
            else None
        )

        cached_state = self.orientation_states.get(object_key)
        if cached_state is not None:
            cached_points, is_upside_down, angle, cached_path = cached_state
            moved = not np.allclose(
                points, cached_points, atol=ORIENTATION_POSITION_TOLERANCE
            )
            due = tick >= ORIENTATION_RECHECK_TICKS
            if cached_path == template_path and not moved and not due:
                self.orientation_ticks[object_key] = tick
                return is_upside_down, angle, ""

        is_upside_down, angle, orientation_error = self.check_orientation(roi)
        if orientation_error:
            return is_upside_down, angle, orientation_error

        self.orientation_states[object_key] = (points.copy(), is_upside_down, angle, template_path)
        self.orientation_ticks[object_key] = 0
        self.orientation_state = self.orientation_states[object_key]
        self.orientation_tick = 0
        return is_upside_down, angle, ""

    @staticmethod
    def _package_roi(points):
        """Convert smoothed OBB corner points into a rotated-ROI dict.

        Trainer data preparation follows the YOLO OBB geometry used by live
        Inspection. This helper only derives display-oriented package data.
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

    def inspect(self, frame, points, object_key=None, render_heatmap=True):
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

        is_upside_down, angle, orientation_error = self.cached_orientation(
            roi, points, object_key
        )

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
        self._last_normalized_roi = normalized_roi
        score, anomaly_map = self.patchcore_service.predict_full(normalized_roi)
        self._last_anomaly_map = anomaly_map

        threshold = normalize_anomaly_threshold(raw_threshold)
        is_defect, reason, gate_region_px = evaluate_verdict(
            score, anomaly_map, threshold, self.current_recipe
        )

        heatmap = None
        if render_heatmap:
            heatmap = self._summary_heatmap({
                "_normalized_roi": normalized_roi,
                "_anomaly_map": anomaly_map,
            })

        result = "FAIL" if is_defect else "PASS"

        gate_info = f" gate_region={gate_region_px}px" if gate_region_px else ""
        if INSPECTION_TRACE:
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

        if INSPECTION_TRACE:
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
        self._last_overlay_z_order_sync = 0.0
        self._display_versions = None
        self._display_source = None
        self._status_result_version = 0
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(33)
        self.live_timer.timeout.connect(self._poll_live)
        self.live_timer.start()
        self.worker.overlay_update.connect(self._apply_overlay)
        self.worker.status_update.connect(self._apply_status)
        self.status_widget.recipe_request_changed.connect(self.worker.apply_recipe)
        self.status_widget.capture_mode_changed.connect(self.worker.set_capture_only)

        geometry = app.primaryScreen().availableGeometry()
        self.status_widget.move(
            geometry.width() - self.status_widget.width() - 20,
            20,
        )

        app.aboutToQuit.connect(self._shutdown)

        self.overlay.show()
        # Capture must remain accessible even without an inspection source.
        self.status_widget.show()

        self.worker.start()

    def _poll_live(self):
        """Render latest geometry; results explicitly describe a sampled frame."""
        if self.status_widget.capture_mode or self.overlay.roi_mode:
            return
        version, packet = self.worker.live_frames.read()
        result_version, completed = self.worker.completed_frames.read()
        if packet is None or packet["epoch"] != self.worker.pipeline_epoch:
            return
        versions = (version, result_version)
        if self._display_versions == versions:
            return
        self._display_versions = versions
        source = (packet["epoch"], packet["source_id"])
        if packet["window"] is None:
            self._apply_overlay(None, None, [], None)
            self.status_widget.update_status(
                system_state=ProgramState.WAITING_SOURCE,
                inspection_result="NO SOURCE", count=0, fps=0,
                error_message=packet["error"],
            )
            self._display_source = source
            return
        boxes = [dict(box, label="Package") for box in packet["boxes"]]
        valid_result = (
            completed is not None
            and (completed["epoch"], completed["source_id"]) == source
            and completed["payload"]["recipe_request_id"] == self.status_widget.recipe_request_id
        )
        if valid_result:
            # Only annotate geometrically corresponding boxes. The LAST
            # prefix distinguishes the completed sample from the live view;
            # never project old anomaly contours onto a moving frame.
            available = list(completed["boxes"])
            for box in boxes:
                match = next((old for old in available if np.allclose(
                    old["points"], box["points"], atol=2.0, rtol=0)), None)
                if match is not None:
                    available = [old for old in available if old is not match]
                    box.update(result=match.get("result", "DETECTED"),
                               score=match.get("score", 0), label=match["label"])
                    box["display_text"] = f"{match['label']} | LAST {box['result']} | {box['score']:.3f}"
                    box["segments"] = match.get("segments", [])
            if result_version != self._status_result_version:
                payload = dict(completed["payload"])
                elapsed = time.monotonic() - completed["captured_at"]
                payload["fail_reason"] = f"Last inspected sample ({elapsed:.2f}s): " + payload.get("fail_reason", "")
                self._apply_status(payload)
                self._status_result_version = result_version
        elif self._display_source != source:
            self.status_widget.update_status(
                system_state=ProgramState.READY,
                inspection_result="READY" if boxes else "NOT FOUND",
                count=len(boxes), fps=packet["fps"],
                fail_reason="Waiting for the first inspected sample",
                recipe_request_id=self.status_widget.recipe_request_id,
            )
        self._display_source = source
        self._apply_overlay(packet["frame"], packet["rect"], boxes,
                            getattr(packet["window"], "_hWnd", None))

    @Slot(object, object, object, object)
    def _apply_overlay(self, frame, rect, boxes, hwnd):
        """Apply one overlay frame on the GUI thread.

        ``hwnd`` is the source window's native handle: the overlay is
        re-inserted directly above it on every update, so windows the user
        opens over the source cover the overlay as well.
        """
        if self.status_widget.capture_mode or self.overlay.roi_mode:
            return
        if frame is None or rect is None:
            self.status_widget.set_capture_frame(None)
            self.overlay.capture_view_rect = None
            self.overlay.hide()
            return

        left, top, width, height = rect
        if (self.overlay.x(), self.overlay.y(), self.overlay.width(), self.overlay.height()) != (
            int(left), int(top), int(width), int(height)
        ):
            self.overlay.setGeometry(int(left), int(top), int(width), int(height))
        self._place_status_widget(left, top, width, height)
        if not self.status_widget.isVisible():
            self.status_widget.show()
        self.overlay.set_frame(frame)
        self.overlay.capture_view_rect = None
        self.overlay.update_boxes(boxes)
        self.overlay.show()
        now = time.monotonic()
        if now - self._last_overlay_z_order_sync >= OVERLAY_Z_ORDER_INTERVAL_SECONDS:
            self.overlay.sync_above_window(hwnd)
            self._last_overlay_z_order_sync = now

    def _place_status_widget(self, left, top, width, height):
        if QApplication.activePopupWidget() is not None:
            return
        if not self.status_widget.is_pinned or self.status_widget.is_minimized:
            return
        screen = self.status_widget.screen()
        if screen is None:
            return
        area = screen.availableGeometry()
        gap = 14
        right_x = int(left + width + gap)
        left_x = int(left - self.status_widget.width() - gap)
        if right_x + self.status_widget.width() <= area.right():
            x = right_x
        elif left_x >= area.left():
            x = left_x
        else:
            x = max(area.left(), min(right_x, area.right() - self.status_widget.width()))
        y = max(area.top(), min(int(top), area.bottom() - self.status_widget.height()))
        if self.status_widget.x() != x or self.status_widget.y() != y:
            self.status_widget.move(x, y)

    @Slot(dict)
    def _apply_status(self, payload):
        if self.status_widget.capture_mode:
            return
        self.status_widget.update_status(**payload)

    def _shutdown(self):
        self.live_timer.stop()
        self.worker.stop()
        # Do not destroy a running QThread/model while inference is finishing.
        self.worker.wait()


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

    print("Loading detection model...")
    try:
        detection_model = YOLO(str(model_path))
    except Exception as error:
        print(f"Failed to load detection model: {error}")
        QMessageBox.critical(
            None,
            "VisionAI",
            f"Failed to load the detection model:\n{error}",
        )
        sys.exit(1)
    print("Detection model loaded")

    # Open the inspection UI only after YOLO is ready.
    app.inspection_controller = InspectionApp(app, detection_model)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
