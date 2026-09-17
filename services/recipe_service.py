"""Creation, loading and persistence of inspection recipes.

A recipe is a JSON file stored under ``recipes/<recipe_name>/recipe.json``.
It describes the package, the dataset, the ROI settings, the trained model
location and the inspection thresholds for one IC package type.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from utils.paths import RECIPES_DIR

DEFAULT_ANOMALY_THRESHOLD = 0.60

# Provisional localized-anomaly (pixel gate) defaults. The gate fails a part
# when the anomaly map holds a compact region above a fraction of the image
# threshold. Like the other provisional safety gates, these must be
# recalibrated against measured validation data before production approval.
DEFAULT_PIXEL_GATE_RATIO = 0.65
DEFAULT_PIXEL_GATE_MIN_AREA = 12
DEFAULT_PIXEL_GATE_MAX_AREA = 600


def stored_threshold_usable(value):
    """True when a stored threshold is parseable and in a meaningful range.

    Calibrated thresholds are 0-1 fractions; legacy percent values in
    (1, 100) migrate through ``normalize_anomaly_threshold``. Anything else
    (empty, unparseable, <= 0, exactly 1.0, >= 100) is unusable: a threshold
    of exactly 1.0 can never be exceeded, so it is rejected as well.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return False

    return 0.0 < threshold < 1.0 or 1.0 < threshold < 100.0


def normalize_anomaly_threshold(value):
    """Coerce a stored anomaly threshold into a 0-1 fraction.

    Values in (0, 1) are kept as-is, percent values in (1, 100) are divided
    by 100, anything else falls back to the default. Callers that must never
    guess (calibrated recipes) should check ``stored_threshold_usable`` first
    and fail closed instead of accepting the substituted default.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        if value not in (None, ""):
            print(
                f"[THRESHOLD] Invalid value {value!r}, "
                f"using {DEFAULT_ANOMALY_THRESHOLD}"
            )
        return DEFAULT_ANOMALY_THRESHOLD

    if stored_threshold_usable(value):
        if threshold < 1.0:
            return threshold

        migrated = threshold / 100.0
        print(f"[THRESHOLD] Migrating percent value {threshold} to {migrated}")
        return migrated

    print(
        f"[THRESHOLD] Value {threshold} out of range, "
        f"using {DEFAULT_ANOMALY_THRESHOLD}"
    )
    return DEFAULT_ANOMALY_THRESHOLD


def default_pixel_gate():
    """Return a fresh pixel-gate config block for recipes."""
    return {
        "enabled": True,
        "threshold_ratio": DEFAULT_PIXEL_GATE_RATIO,
        "min_area_px": DEFAULT_PIXEL_GATE_MIN_AREA,
        "max_area_px": DEFAULT_PIXEL_GATE_MAX_AREA,
    }


def _positive_fraction(value):
    """Return ``value`` as a positive float, or ``None`` when unusable.

    The upper clamp into (0, 1] happens in ``normalize_pixel_gate``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _positive_int(value):
    """Return ``value`` as an int >= 1, or ``None`` when unusable."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def normalize_pixel_gate(recipe):
    """Coerce a recipe's ``pixel_gate`` block into a validated config.

    Recipes created before the gate simply have no block and receive the
    defaults; a non-dict block is replaced wholesale. Individual fields are
    coerced per key so one corrupt number never silently disables or
    distorts the others: ``enabled`` must be a real bool, ``threshold_ratio``
    is clamped into (0, 1] (the region limit must stay below the image
    limit) and the area bounds are clamped to ``max_area_px >= min_area_px
    >= 1``.
    """
    gate = default_pixel_gate()
    block = recipe.get("pixel_gate") if isinstance(recipe, dict) else None
    if not isinstance(block, dict):
        if block is not None:
            print(f"[PIXEL-GATE] Invalid block {block!r}, using defaults")
        return gate

    if isinstance(block.get("enabled"), bool):
        gate["enabled"] = block["enabled"]
    else:
        print(
            f"[PIXEL-GATE] Invalid enabled {block.get('enabled')!r}, "
            f"using {gate['enabled']}"
        )

    ratio = _positive_fraction(block.get("threshold_ratio"))
    if ratio is None:
        print(
            f"[PIXEL-GATE] Invalid threshold_ratio "
            f"{block.get('threshold_ratio')!r}, using "
            f"{gate['threshold_ratio']}"
        )
    else:
        gate["threshold_ratio"] = min(ratio, 1.0)

    min_area = _positive_int(block.get("min_area_px"))
    if min_area is None:
        print(
            f"[PIXEL-GATE] Invalid min_area_px "
            f"{block.get('min_area_px')!r}, using {gate['min_area_px']}"
        )
    else:
        gate["min_area_px"] = min_area

    max_area = _positive_int(block.get("max_area_px"))
    if max_area is None:
        print(
            f"[PIXEL-GATE] Invalid max_area_px "
            f"{block.get('max_area_px')!r}, using {gate['max_area_px']}"
        )
    else:
        gate["max_area_px"] = max_area

    gate["min_area_px"] = max(gate["min_area_px"], 1)
    gate["max_area_px"] = max(gate["max_area_px"], gate["min_area_px"])
    return gate


class RecipeService:
    """Manages recipes stored as JSON files under the recipes directory."""

    def __init__(self):
        self.recipe_root = RECIPES_DIR
        self.recipe_root.mkdir(parents=True, exist_ok=True)

    def _recipe_file(self, recipe_name):
        self._validate_recipe_name(recipe_name)
        return self.recipe_root / recipe_name / "recipe.json"

    @staticmethod
    def _validate_recipe_name(recipe_name):
        """Reject recipe names that could escape the recipes directory."""
        name = str(recipe_name or "").strip()
        if not name:
            raise ValueError("Recipe name must not be empty.")

        normalized = name.replace("\\", "/")
        if "/" in normalized:
            raise ValueError(
                f"Invalid recipe name {name!r}: path separators are not allowed."
            )
        if any(part == ".." for part in normalized.split("/")):
            raise ValueError(
                f"Invalid recipe name {name!r}: '..' components are not allowed."
            )

        return name

    @staticmethod
    def write_json_file(path, data):
        """Atomically write ``data`` as pretty JSON to ``path``.

        The payload lands in a UNIQUE sibling temp file first and is then
        moved over the target with ``os.replace``, so a crash or power loss
        mid-write can never leave a truncated production recipe behind. The
        unique name also keeps two concurrent writers (trainer pages, the
        inspection status widget) from interleaving into the same temp file.
        """
        path = Path(path)
        fd, temp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=4, ensure_ascii=False)
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_json(path, data):
        RecipeService.write_json_file(path, data)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_recipe(self, recipe_name, recipe_data):
        name = self._validate_recipe_name(recipe_name)
        recipe_dir = self.recipe_root / name
        recipe_dir.mkdir(parents=True, exist_ok=True)

        # Persist the resolved pixel-gate config so the recipe of record
        # documents the exact limits the inspection app enforces (defaults
        # included) instead of inheriting invisible code defaults.
        recipe_data["pixel_gate"] = normalize_pixel_gate(recipe_data)

        recipe_file = recipe_dir / "recipe.json"
        self._write_json(recipe_file, recipe_data)
        return recipe_file

    def load_recipe(self, recipe_name):
        recipe_file = self._recipe_file(recipe_name)
        if not recipe_file.exists():
            return None

        with open(recipe_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_recipe(self, recipe_name, recipe_data):
        # Persist the resolved pixel-gate config so the recipe of record
        # documents the exact limits the inspection app enforces (defaults
        # included) instead of inheriting invisible code defaults.
        recipe_data["pixel_gate"] = normalize_pixel_gate(recipe_data)

        recipe_file = self._recipe_file(recipe_name)
        recipe_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(recipe_file, recipe_data)

    def delete_recipe(self, recipe_name):
        name = self._validate_recipe_name(recipe_name)
        recipe_dir = self.recipe_root / name
        if recipe_dir.exists():
            shutil.rmtree(recipe_dir)

    def rename_recipe(self, old_name, new_name):
        old = self._validate_recipe_name(old_name)
        new = self._validate_recipe_name(new_name)
        old_dir = self.recipe_root / old
        new_dir = self.recipe_root / new
        old_dir.rename(new_dir)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recipe_list(self):
        return sorted(
            folder.name for folder in self.recipe_root.iterdir() if folder.is_dir()
        )

    def recipe_exists(self, recipe_name):
        name = self._validate_recipe_name(recipe_name)
        return (self.recipe_root / name).exists()

    def get_recipe_path(self, recipe_name):
        name = self._validate_recipe_name(recipe_name)
        return str(self.recipe_root / name)

    def get_model_path(self, recipe_name):
        recipe = self.load_recipe(recipe_name)
        if recipe is None:
            return ""
        return recipe.get("model", {}).get("path", "")

    def get_recipe_model_info(self, recipe_name):
        recipe = self.load_recipe(recipe_name)
        if recipe is None:
            return None
        return recipe.get("model", {})

    # ------------------------------------------------------------------
    # Field updates
    # ------------------------------------------------------------------

    def update_dataset_path(self, recipe_name, dataset_path):
        self._update_field(recipe_name, "dataset_path", dataset_path)

    def update_roi_path(self, recipe_name, roi_path):
        self._update_field(recipe_name, "roi_path", roi_path)

    def update_model_path(self, recipe_name, model_path):
        self._update_field(recipe_name, "anomaly_model", model_path)

    def _update_field(self, recipe_name, field, value):
        recipe = self.load_recipe(recipe_name)
        if recipe is None:
            return

        recipe[field] = value
        self.save_recipe(recipe_name, recipe)

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------

    @staticmethod
    def get_recipe_template():
        """Return the default structure for a new recipe."""
        return {
            "recipe_name": "",
            "package_type": "",
            "package_family": "",
            "package_size": "",
            "type_name": "",
            "package_version": "",
            "pin_count": 0,
            "target_count": 1,
            "dataset_path": "",
            "roi_path": "",
            "thumbnail_path": "",
            "confidence_threshold": 0.95,
            "created_date": "",
            "last_trained": "",
            "version": "1.0",
            "notes": "",
            "golden_image": "",
            "top_mark_template": "",
            "top_mark_roi": {},
            "laser_mark_template": "",
            "laser_mark_roi": {},
            "laser_mark_threshold": 0.80,
            "model_type": "patchcore",
            "backbone": "wide_resnet50_2",
            "dataset_prepared": "",
            "memory_bank_path": "",
            "roi_default": {
                "width_scale": 1.0,
                "height_scale": 1.0,
                "padding": 10,
                "angle_offset": 0,
            },
            "roi_overrides": {},
            "anomaly_threshold": DEFAULT_ANOMALY_THRESHOLD,
            "pixel_gate": default_pixel_gate(),
        }
