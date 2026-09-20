"""Pure-logic helpers for the guided training flow.

Everything in this module is Qt-free so it can be unit-tested headlessly.
Functions that need images receive them as numpy arrays or file paths;
YOLO / service calls stay with the callers (pages and workers).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from utils.image_utils import prepare_inspection_crop
from utils.paths import MODEL_REGISTRY_PATH, resolve_recipe_path

from services.dataset_service import IMAGE_EXTENSIONS

# Step identifiers, in flow order (S5 and S6 share the training page).
STEP_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

# Page index inside TrainerWindow for every step (Review page is not a
# gated step; S6 calibration lives on the same page as S5 training).
STEP_PAGE_INDEX = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 4, "S7": 6}

# Step statuses used in assistant_state and by the step bar.
STATUS_OK = "ok"
STATUS_ACTION_NEEDED = "action_needed"
STATUS_WARNING = "warning"
STATUS_BLOCKED = "blocked"

DEFAULT_MARGIN = 1.2
CEILING_FRACTION = 1.0      # a good-image score at 1.0 means the scale ceiling
FLOOR_MARGIN_DELTA = 0.02   # keep the proposal at least this far above P99

# Relative quality-screening thresholds.
BLUR_RELATIVE_FRACTION = 0.35
BLUR_ABSOLUTE_FLOOR = 25.0
BRIGHTNESS_RELATIVE_FRACTION = 0.4
SIZE_RELATIVE_FRACTION = 0.5

# Crop QA fallback tolerances (auto-tune stores per-recipe values).
DEFAULT_WIDTH_TOLERANCE = 0.25
DEFAULT_HEIGHT_TOLERANCE = 0.25
DEFAULT_RATIO_TOLERANCE = 0.2

# Recommendation rules (S4). Metadata-only scores are capped so a reuse
# recommendation never fires without visual confirmation.
REUSE_SIMILARITY = 0.85
REVIEW_SIMILARITY = 0.60
METADATA_WEIGHT = 0.4
VISUAL_WEIGHT = 0.6
METADATA_ONLY_CAP = 0.6
MATCH_SCALES = (0.75, 0.85, 0.95, 1.0)


def utc_now_iso():
    """Return the current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Threshold calibration (S6)
# ---------------------------------------------------------------------------

def recommend_threshold(scores, margin=DEFAULT_MARGIN):
    """Derive a threshold proposal from good-image scores (0-1 fractions).

    Returns ``(proposed, trace)``. ``proposed`` is clamped to ``(0, 1]``;
    ``trace`` carries the full statistics for the recipe calibration block.
    """
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("No scores were provided for calibration.")

    array = np.asarray(values, dtype=float)
    p99 = float(np.percentile(array, 99))

    proposed = min(1.0, p99 * float(margin))
    if proposed < p99 + FLOOR_MARGIN_DELTA:
        proposed = p99 + FLOOR_MARGIN_DELTA
    proposed = min(1.0, max(proposed, 1e-6))

    trace = {
        "method": "held_out_good_margin_v1",
        "p99": p99,
        "margin": float(margin),
        "proposed": proposed,
        "expected_false_alarms": int(np.count_nonzero(array > proposed)),
        # Warn both when the training set itself sits at the scale ceiling
        # and when the proposal saturates at 1.0: a threshold of exactly 1.0
        # can never be exceeded, so it would pass every part.
        "warning": bool(
            np.max(array) >= CEILING_FRACTION or proposed >= CEILING_FRACTION
        ),
        "good_summary": {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "p99": p99,
            "max": float(np.max(array)),
            "count": len(values),
        },
    }
    if trace["warning"]:
        trace["warning_reason"] = "Good-image scores reach the 1.0 scale ceiling."
    return proposed, trace


def recommend_threshold_with_anomalies(good_scores, anomaly_scores):
    """Choose an image threshold with <=1% held-out-good false alarms."""
    good = np.asarray([float(value) for value in good_scores], dtype=float)
    anomaly = np.asarray([float(value) for value in anomaly_scores], dtype=float)
    if not len(good) or not len(anomaly) or not np.isfinite(good).all() or not np.isfinite(anomaly).all():
        raise ValueError("Good and synthetic anomaly scores are required")
    values = sorted(set(np.concatenate([good, anomaly]).tolist()))
    candidates = [max(1e-6, values[0] - 1e-6)] + [
        (left + right) / 2 for left, right in zip(values, values[1:])
    ] + [values[-1] + 1e-6]
    max_false = int(np.floor(len(good) * 0.01))
    feasible = []
    for threshold in candidates:
        false_alarms = int(np.count_nonzero(good > threshold))
        detected = int(np.count_nonzero(anomaly > threshold))
        if false_alarms <= max_false:
            feasible.append((detected, -false_alarms, -threshold, threshold))
    if not feasible:
        raise ValueError("No threshold satisfies the good-image false-alarm limit")
    _, _, _, proposed = max(feasible)
    detection_rate = float(np.mean(anomaly > proposed))
    trace = {
        "method": "held_out_good_plus_synthetic_v1",
        "proposed": float(proposed),
        "expected_false_alarms": int(np.count_nonzero(good > proposed)),
        "synthetic_detected": int(np.count_nonzero(anomaly > proposed)),
        "synthetic_total": int(len(anomaly)),
        "synthetic_detection_rate": detection_rate,
        "warning": bool(detection_rate < 0.8),
        "good_summary": {"min": float(good.min()), "median": float(np.median(good)),
                         "p95": float(np.percentile(good, 95)), "p99": float(np.percentile(good, 99)),
                         "max": float(good.max()), "count": int(len(good))},
    }
    if trace["warning"]:
        trace["warning_reason"] = (
            "The model detects fewer than 80% of the synthetic validation defects."
        )
    return float(proposed), trace


# ---------------------------------------------------------------------------
# Dataset quality screening (S2)
# ---------------------------------------------------------------------------

def assess_dataset(image_paths, progress_cb=None):
    """Quality-screen a list of image files.

    Flags per image: ``corrupt``, ``blurry`` (Laplacian variance), and
    ``brightness`` / ``size`` outliers relative to the set median. Dataset
    size itself is never judged here — counts are informational only.
    """
    entries = []
    total = max(len(image_paths), 1)

    for index, path in enumerate(image_paths):
        entry = {"path": str(path), "name": Path(path).name, "flags": []}
        image = cv2.imread(str(path))

        if image is None:
            entry["flags"].append("corrupt")
            entries.append(entry)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            entry["width"] = int(image.shape[1])
            entry["height"] = int(image.shape[0])
            entry["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            entry["brightness"] = float(np.mean(gray))
            entries.append(entry)

        if progress_cb is not None:
            progress_cb(index + 1, total)

    readable = [entry for entry in entries if "corrupt" not in entry["flags"]]
    medians = {}
    if readable:
        medians = {
            "sharpness": float(np.median([e["sharpness"] for e in readable])),
            "brightness": float(np.median([e["brightness"] for e in readable])),
            "area": float(np.median([e["width"] * e["height"] for e in readable])),
        }

    for entry in readable:
        blur_limit = max(
            BLUR_ABSOLUTE_FLOOR, BLUR_RELATIVE_FRACTION * medians["sharpness"]
        )
        if medians["sharpness"] > 0 and entry["sharpness"] < blur_limit:
            entry["flags"].append("blurry")

        brightness_gap = abs(entry["brightness"] - medians["brightness"])
        if brightness_gap > BRIGHTNESS_RELATIVE_FRACTION * max(
            medians["brightness"], 1.0
        ):
            entry["flags"].append("brightness")

        area = entry["width"] * entry["height"]
        if medians["area"] > 0 and not (
            (1 - SIZE_RELATIVE_FRACTION) * medians["area"]
            <= area
            <= (1 + SIZE_RELATIVE_FRACTION) * medians["area"]
        ):
            entry["flags"].append("size")

    return {"results": entries, "medians": medians}


# ---------------------------------------------------------------------------
# Dataset geometry analysis + package registry matching (S2)
# ---------------------------------------------------------------------------

def load_package_registry(path=None):
    """Load the known-package registry (list of dicts); [] on any failure."""
    registry_path = Path(path) if path is not None else MODEL_REGISTRY_PATH
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def match_registry(dominant_class, registry):
    """Match a detection class name against the package registry."""
    if not dominant_class or not registry:
        return None

    needle = str(dominant_class).strip().lower()
    if not needle:
        return None

    for entry in registry:
        if not isinstance(entry, dict):
            continue
        for key in ("package_type", "package_family", "type_name"):
            value = str(entry.get(key, "")).strip().lower()
            if value and (value == needle or value in needle or needle in value):
                return {
                    "package_family": entry.get("package_family", ""),
                    "package_type": entry.get("package_type", ""),
                    "pin_count": entry.get("pin_count"),
                    "match_score": 1.0,
                }
    return None


def analyze_dataset_geometry(detections, registry=None):
    """Aggregate per-image YOLO OBB detections into dataset-level facts.

    ``detections`` is a list with one entry per image; each entry is a list
    of detection dicts carrying at least ``width``, ``height`` and optionally
    ``class_name``. Dataset size never gates anything — every number here is
    informational.
    """
    if registry is None:
        registry = load_package_registry()

    flat = [detection for per_image in detections for detection in per_image]
    result = {
        "images_analyzed": len(detections),
        "images_with_detection": sum(1 for per_image in detections if per_image),
        "detection_count": len(flat),
        "dominant_class": None,
        "registry_match": None,
    }
    if not flat:
        return result

    widths = [float(d["width"]) for d in flat if d.get("width")]
    heights = [float(d["height"]) for d in flat if d.get("height")]
    aspects = [
        w / h for w, h in zip(widths, heights) if w and h
    ]

    if widths and heights:
        result["size_stats"] = {
            "median_width": float(np.median(widths)),
            "median_height": float(np.median(heights)),
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
            "min_height": float(np.min(heights)),
            "max_height": float(np.max(heights)),
        }
    if aspects:
        result["median_aspect_ratio"] = float(np.median(aspects))

    class_names = [
        str(d.get("class_name")) for d in flat if d.get("class_name")
    ]
    if class_names:
        values, counts = np.unique(np.asarray(class_names), return_counts=True)
        result["dominant_class"] = str(values[int(np.argmax(counts))])
        result["registry_match"] = match_registry(
            result["dominant_class"], registry
        )

    return result


# ---------------------------------------------------------------------------
# ROI auto-tune + crop QA (S3)
# ---------------------------------------------------------------------------

def auto_tune_plan(detections):
    """Derive ROI defaults + QA tolerances from detected OBB sizes.

    ``detections`` is a flat list of dicts with ``width`` / ``height``.
    Returns a plan dict or ``None`` when too few packages were detected.
    """
    widths = [float(d["width"]) for d in detections if d.get("width")]
    heights = [float(d["height"]) for d in detections if d.get("height")]
    if len(widths) < 3 or len(heights) < 3:
        return None

    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    ratios = [w / h for w, h in zip(widths, heights) if w and h]
    med_ratio = float(np.median(ratios)) if ratios else 1.0

    def tolerance(values, median):
        if median <= 0:
            return 0.5
        spread = float(np.percentile(values, 95) - np.percentile(values, 5))
        return float(min(0.5, max(0.08, spread / (2.0 * median) + 0.05)))

    return {
        "width_scale": 1.0,
        "height_scale": 1.0,
        "padding": int(min(60, max(2, round(0.05 * min(med_w, med_h))))),
        "angle_offset": 0.0,
        "width_tolerance": tolerance(widths, med_w),
        "height_tolerance": tolerance(heights, med_h),
        "ratio_tolerance": tolerance(ratios, med_ratio),
    }


def qa_bounds(avg_width, avg_height, avg_ratio, tolerances):
    """Compute accept/reject bounds for generated crops."""
    width_tol = float(
        tolerances.get("width_tolerance", DEFAULT_WIDTH_TOLERANCE)
    )
    height_tol = float(
        tolerances.get("height_tolerance", DEFAULT_HEIGHT_TOLERANCE)
    )
    ratio_tol = float(
        tolerances.get("ratio_tolerance", DEFAULT_RATIO_TOLERANCE)
    )
    return {
        "min_width": avg_width * (1 - width_tol),
        "max_width": avg_width * (1 + width_tol),
        "min_height": avg_height * (1 - height_tol),
        "max_height": avg_height * (1 + height_tol),
        "min_ratio": avg_ratio * (1 - ratio_tol),
        "max_ratio": avg_ratio * (1 + ratio_tol),
    }


def classify_crop(item, bounds):
    """Return the list of QA reject reasons for one crop item ([] = pass)."""
    reasons = []
    if not bounds["min_width"] <= item["width"] <= bounds["max_width"]:
        reasons.append(
            f"width {item['width']:.0f} outside "
            f"{bounds['min_width']:.0f}-{bounds['max_width']:.0f}"
        )
    if not bounds["min_height"] <= item["height"] <= bounds["max_height"]:
        reasons.append(
            f"height {item['height']:.0f} outside "
            f"{bounds['min_height']:.0f}-{bounds['max_height']:.0f}"
        )
    if not bounds["min_ratio"] <= item["ratio"] <= bounds["max_ratio"]:
        reasons.append(
            f"aspect ratio {item['ratio']:.2f} outside "
            f"{bounds['min_ratio']:.2f}-{bounds['max_ratio']:.2f}"
        )
    return reasons


# ---------------------------------------------------------------------------
# Assistant state persisted in the recipe (S1-S7)
# ---------------------------------------------------------------------------

def get_assistant_state(recipe):
    """Return (creating if needed) the assistant_state dict of a recipe."""
    if not isinstance(recipe, dict):
        return {"steps": {}, "chosen_path": None, "chosen_recipe": None}

    state = recipe.get("assistant_state")
    if not isinstance(state, dict):
        state = {}
        recipe["assistant_state"] = state

    if not isinstance(state.get("steps"), dict):
        state["steps"] = {}

    return state


def set_step_status(recipe, step_id, status):
    """Record one step status (with timestamp) on the recipe dict."""
    state = get_assistant_state(recipe)
    state["steps"][step_id] = {"status": status, "updated_at": utc_now_iso()}


def get_step_status(recipe, step_id, default=STATUS_ACTION_NEEDED):
    """Read one step status, falling back to ``default``."""
    entry = get_assistant_state(recipe)["steps"].get(step_id)
    if isinstance(entry, dict):
        return entry.get("status", default)
    return default


def set_chosen_path(recipe, path, chosen_recipe=None):
    """Record the S4 decision (reuse / augment / train_new) on the recipe."""
    state = get_assistant_state(recipe)
    state["chosen_path"] = path
    state["chosen_recipe"] = chosen_recipe


def _count_images(folder):
    if not folder:
        return 0
    folder = Path(folder)
    if not folder.exists():
        return 0
    count = 0
    for extension in IMAGE_EXTENSIONS:
        count += sum(1 for _ in folder.glob(extension))
    return count


def _count_crops(recipe):
    prepared = resolve_recipe_path(recipe.get("prepared_dataset_path"))
    if prepared is None:
        return 0
    good_dir = prepared / "train" / "good"
    if not good_dir.exists():
        return 0
    count = 0
    for extension in IMAGE_EXTENSIONS:
        count += sum(1 for _ in good_dir.glob(extension))
    return count


def compute_step_status(recipe):
    """Compute S1-S7 statuses from the recipe (no Qt, no workers).

    Returns ``{step_id: {"status", "detail", "page_index"}}``. Dataset size
    never gates any step — S2 only needs at least one readable image entry.
    """
    statuses = {}

    if not recipe:
        for step_id in STEP_IDS:
            statuses[step_id] = {
                "status": STATUS_ACTION_NEEDED,
                "detail": "No recipe selected.",
                "page_index": STEP_PAGE_INDEX[step_id],
            }
        return statuses

    # S1 Recipe
    missing = []
    for field in ("recipe_name", "package_family", "package_type"):
        if not str(recipe.get(field) or "").strip():
            missing.append(field)
    try:
        pin_ok = int(recipe.get("pin_count") or 0) > 0
    except (TypeError, ValueError):
        pin_ok = False
    if not pin_ok:
        missing.append("pin_count")

    statuses["S1"] = {
        "status": STATUS_OK if not missing else STATUS_ACTION_NEEDED,
        "detail": (
            "Metadata complete."
            if not missing
            else "Missing: " + ", ".join(missing)
        ),
        "page_index": STEP_PAGE_INDEX["S1"],
    }

    # S2 Dataset (size is informational only)
    dataset_path = str(recipe.get("dataset_path") or "")
    image_count = _count_images(dataset_path) if dataset_path else 0
    if dataset_path and image_count >= 1:
        statuses["S2"] = {
            "status": STATUS_OK,
            "detail": f"{image_count} images found.",
            "page_index": STEP_PAGE_INDEX["S2"],
        }
    else:
        statuses["S2"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Select a dataset folder with at least one image.",
            "page_index": STEP_PAGE_INDEX["S2"],
        }

    # S3 ROI / crops
    crops = _count_crops(recipe)
    prepared = bool(recipe.get("dataset_prepared"))
    if prepared and crops >= 10:
        statuses["S3"] = {
            "status": STATUS_OK,
            "detail": f"{crops} crops saved.",
            "page_index": STEP_PAGE_INDEX["S3"],
        }
    elif prepared:
        statuses["S3"] = {
            "status": STATUS_WARNING,
            "detail": f"Only {crops} crops saved (10+ recommended).",
            "page_index": STEP_PAGE_INDEX["S3"],
        }
    else:
        statuses["S3"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Generate the training crops.",
            "page_index": STEP_PAGE_INDEX["S3"],
        }

    # S4 Recommend
    state = get_assistant_state(recipe)
    chosen = state.get("chosen_path")
    if chosen:
        detail = f"Path chosen: {chosen}"
        if state.get("chosen_recipe"):
            detail += f" ({state['chosen_recipe']})"
        statuses["S4"] = {
            "status": STATUS_OK,
            "detail": detail,
            "page_index": STEP_PAGE_INDEX["S4"],
        }
    else:
        statuses["S4"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Choose reuse / add-to-memory / train-new.",
            "page_index": STEP_PAGE_INDEX["S4"],
        }

    # S5 Train
    model = recipe.get("model") or {}
    if model.get("trained"):
        statuses["S5"] = {
            "status": STATUS_OK,
            "detail": "Model trained.",
            "page_index": STEP_PAGE_INDEX["S5"],
        }
    else:
        statuses["S5"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Train the PatchCore model.",
            "page_index": STEP_PAGE_INDEX["S5"],
        }

    # S6 Calibrate
    calibration = recipe.get("calibration")
    if isinstance(calibration, dict) and calibration.get("proposed") is not None:
        statuses["S6"] = {
            "status": STATUS_OK,
            "detail": f"Threshold {float(calibration['proposed']):.2f} applied.",
            "page_index": STEP_PAGE_INDEX["S6"],
        }
    else:
        statuses["S6"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Calibrate the anomaly threshold.",
            "page_index": STEP_PAGE_INDEX["S6"],
        }

    # S7 Finalize
    if recipe.get("validated"):
        statuses["S7"] = {
            "status": STATUS_OK,
            "detail": "Recipe finalized.",
            "page_index": STEP_PAGE_INDEX["S7"],
        }
    else:
        statuses["S7"] = {
            "status": STATUS_ACTION_NEEDED,
            "detail": "Review the summary and finalize.",
            "page_index": STEP_PAGE_INDEX["S7"],
        }

    return statuses


def next_step(recipe):
    """Return ``(step_id, entry)`` of the first non-OK step, or (None, None)."""
    statuses = compute_step_status(recipe)
    for step_id in STEP_IDS:
        if statuses[step_id]["status"] != STATUS_OK:
            return step_id, statuses[step_id]
    return None, None


def pre_flight(recipe):
    """Checklist gating the Train button. Returns ``(allowed, reasons)``."""
    if not recipe:
        return False, ["No recipe selected."]

    reasons = []
    statuses = compute_step_status(recipe)

    if statuses["S2"]["status"] != STATUS_OK:
        reasons.append("Select a dataset folder with images first (S2).")
    if statuses["S3"]["status"] != STATUS_OK:
        reasons.append("Generate at least 10 training crops first (S3).")
    if statuses["S4"]["status"] != STATUS_OK:
        reasons.append("Choose a model path on the recommendation page first (S4).")

    return (not reasons), reasons


# ---------------------------------------------------------------------------
# S4 entry recommender (placed after ROI/crop generation)
# ---------------------------------------------------------------------------

def metadata_similarity(new_recipe, candidate):
    """Metadata-only similarity in [0, 1] (family/type/pin count)."""
    score = 0.0

    new_family = str(new_recipe.get("package_family") or "").strip().lower()
    cand_family = str(candidate.get("package_family") or "").strip().lower()
    if new_family and new_family == cand_family:
        score += 0.5

    new_type = str(new_recipe.get("package_type") or "").strip().lower()
    cand_type = str(candidate.get("package_type") or "").strip().lower()
    if new_type and new_type == cand_type:
        score += 0.3

    try:
        new_pins = int(new_recipe.get("pin_count") or 0)
    except (TypeError, ValueError):
        new_pins = 0
    try:
        cand_pins = int(candidate.get("pin_count") or 0)
    except (TypeError, ValueError):
        cand_pins = 0
    if new_pins and new_pins == cand_pins:
        score += 0.2

    return min(1.0, score)


def visual_similarity(sample_gray, reference_gray, scales=MATCH_SCALES):
    """Multi-scale template match of the sample inside the reference.

    Both inputs are grayscale numpy arrays. Returns a similarity in [0, 1],
    or ``None`` when either image is unusable. Scales shrink the sample
    (template must stay inside the reference crop).
    """
    if sample_gray is None or reference_gray is None:
        return None

    if sample_gray.ndim == 3:
        sample_gray = cv2.cvtColor(sample_gray, cv2.COLOR_BGR2GRAY)
    if reference_gray.ndim == 3:
        reference_gray = cv2.cvtColor(reference_gray, cv2.COLOR_BGR2GRAY)

    if sample_gray.size == 0 or reference_gray.size == 0:
        return None

    best = -1.0
    for scale in scales:
        width = max(int(reference_gray.shape[1] * scale), 8)
        height = max(int(reference_gray.shape[0] * scale), 8)
        template = cv2.resize(sample_gray, (width, height))
        if (
            template.shape[0] > reference_gray.shape[0]
            or template.shape[1] > reference_gray.shape[1]
        ):
            continue
        result = cv2.matchTemplate(reference_gray, template, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(np.max(result)))

    if best < 0:
        return None
    return max(0.0, min(1.0, best))


def _candidate_reference_crop(candidate):
    """Return the first saved crop of a candidate recipe as grayscale."""
    prepared = resolve_recipe_path(candidate.get("prepared_dataset_path"))
    if prepared is None:
        return None

    good_dir = prepared / "train" / "good"
    if not good_dir.exists():
        return None

    files = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(good_dir.glob(extension))
    if not files:
        return None

    image = cv2.imread(str(sorted(files)[0]))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def recommend_entry(new_recipe, candidates, sample_image_path=None, detect_roi=None):
    """Score candidate recipes for reuse / add-to-memory / train-new.

    The raw sample image is re-cropped with EACH candidate's own ROI config
    and letterboxed to its target size, so both sides of the visual match
    come from the same crop pipeline. Returns::

        {"decision": "reuse"|"review"|"train_new",
         "chosen": candidate name or None,
         "candidates": [{"name", "similarity", "metadata_similarity",
                         "visual_similarity"}]}
    """
    sample_image = None
    roi = None
    if sample_image_path and detect_roi is not None:
        try:
            sample_image = cv2.imread(str(sample_image_path))
            if sample_image is not None:
                roi = detect_roi(str(sample_image_path))
        except Exception:
            sample_image = None

    scored = []
    for candidate in candidates:
        meta = metadata_similarity(new_recipe, candidate)
        visual = None
        if sample_image is not None and roi is not None:
            reference = _candidate_reference_crop(candidate)
            if reference is not None:
                config = dict(candidate.get("roi_default") or {})
                stats = candidate.get("roi_statistics") or {}
                target_w = int(stats.get("target_width") or 256)
                target_h = int(stats.get("target_height") or 256)
                try:
                    canvas = prepare_inspection_crop(
                        sample_image, roi["points"], target_w, target_h
                    )
                    visual = visual_similarity(
                        cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), reference
                    )
                except Exception:
                    visual = None

        if visual is not None:
            combined = METADATA_WEIGHT * meta + VISUAL_WEIGHT * visual
        else:
            combined = METADATA_ONLY_CAP * meta

        scored.append(
            {
                "name": str(candidate.get("recipe_name") or ""),
                "similarity": combined,
                "metadata_similarity": meta,
                "visual_similarity": visual,
            }
        )

    scored.sort(key=lambda entry: entry["similarity"], reverse=True)
    best = scored[0] if scored else None

    decision = "train_new"
    chosen = None
    if best is not None:
        if best["similarity"] >= REUSE_SIMILARITY:
            decision = "reuse"
            chosen = best["name"]
        elif best["similarity"] >= REVIEW_SIMILARITY:
            decision = "review"
            chosen = best["name"]

    return {"decision": decision, "chosen": chosen, "candidates": scored}


# ---------------------------------------------------------------------------
# Finalize (S7)
# ---------------------------------------------------------------------------

def finalize_summary(recipe):
    """Collect the export-page summary fields for a recipe."""
    warnings = []
    model = recipe.get("model") or {}
    stats = recipe.get("roi_statistics") or {}
    calibration = recipe.get("calibration")
    if not isinstance(calibration, dict):
        calibration = None
    analysis = recipe.get("dataset_analysis")
    if not isinstance(analysis, dict):
        analysis = None

    if not model.get("trained"):
        warnings.append("Model is not trained yet.")
    if calibration is None:
        warnings.append("Threshold is the hand-picked default; run calibration.")
    elif calibration.get("warning"):
        warnings.append(
            calibration.get("warning_reason")
            or "Calibration requires review before production use."
        )
    if stats and int(stats.get("saved_count", 0) or 0) < 10:
        warnings.append("Fewer than 10 crops were saved.")
    if analysis:
        geometry = analysis.get("geometry") or {}
        if geometry.get("images_analyzed") and not geometry.get(
            "images_with_detection"
        ):
            warnings.append("YOLO found no package in the dataset images.")

    return {
        "recipe_name": recipe.get("recipe_name", ""),
        "package_family": recipe.get("package_family", ""),
        "package_type": recipe.get("package_type", ""),
        "pin_count": recipe.get("pin_count"),
        "model_trained": bool(model.get("trained")),
        "model_path": model.get("path", ""),
        "crops_saved": stats.get("saved_count"),
        "crops_rejected": stats.get("rejected_count"),
        "anomaly_threshold": recipe.get("anomaly_threshold"),
        "calibration": calibration,
        "dataset_analysis": analysis,
        "version": recipe.get("version", ""),
        "last_trained": recipe.get("last_trained", ""),
        "warnings": warnings,
    }


def apply_finalize(recipe):
    """Bump version / last_trained, set validated, mark S7 OK."""
    version = str(recipe.get("version") or "1.0")
    try:
        major, minor = version.split(".", 1)
        new_version = f"{major}.{int(minor) + 1}"
    except ValueError:
        new_version = "1.1"

    recipe["version"] = new_version
    recipe["last_trained"] = utc_now_iso()
    recipe["validated"] = True
    set_step_status(recipe, "S7", STATUS_OK)
    return new_version
