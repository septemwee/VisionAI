"""Creation, loading and persistence of inspection recipes.

A recipe is a JSON file stored under ``recipes/<recipe_name>/recipe.json``.
It describes the package, the dataset, the ROI settings, the trained model
location and the inspection thresholds for one IC package type.
"""

import json
import shutil

from utils.paths import RECIPES_DIR


class RecipeService:
    """Manages recipes stored as JSON files under the recipes directory."""

    def __init__(self):
        self.recipe_root = RECIPES_DIR
        self.recipe_root.mkdir(parents=True, exist_ok=True)

    def _recipe_file(self, recipe_name):
        return self.recipe_root / recipe_name / "recipe.json"

    @staticmethod
    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_recipe(self, recipe_name, recipe_data):
        recipe_dir = self.recipe_root / recipe_name
        recipe_dir.mkdir(parents=True, exist_ok=True)

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
        self._write_json(self._recipe_file(recipe_name), recipe_data)

    def delete_recipe(self, recipe_name):
        recipe_dir = self.recipe_root / recipe_name
        if recipe_dir.exists():
            shutil.rmtree(recipe_dir)

    def rename_recipe(self, old_name, new_name):
        old_dir = self.recipe_root / old_name
        new_dir = self.recipe_root / new_name
        old_dir.rename(new_dir)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recipe_list(self):
        return sorted(
            folder.name for folder in self.recipe_root.iterdir() if folder.is_dir()
        )

    def recipe_exists(self, recipe_name):
        return (self.recipe_root / recipe_name).exists()

    def get_recipe_path(self, recipe_name):
        return str(self.recipe_root / recipe_name)

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

    def get_threshold(self, recipe_name):
        recipe = self.load_recipe(recipe_name)
        if recipe is None:
            return 0.5
        return recipe.get("anomaly_threshold", 0.5)

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
            "anomaly_threshold": "",
        }
