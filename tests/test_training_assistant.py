"""Tests for the pure-logic training assistant helpers."""

import numpy as np
import pytest

from services import training_assistant as ta


class TestRecommendThreshold:
    def test_normal_scores(self):
        scores = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
        proposed, trace = ta.recommend_threshold(scores, 1.2)
        p99 = float(np.percentile(scores, 99))
        assert proposed == pytest.approx(p99 * 1.2)
        assert trace["expected_false_alarms"] == 0
        assert trace["warning"] is False
        assert trace["good_summary"]["count"] == 6

    def test_tight_distribution_hits_floor(self):
        proposed, _ = ta.recommend_threshold([0.5] * 5, 1.0)
        assert proposed == pytest.approx(0.52)

    def test_ceiling_warning(self):
        _, trace = ta.recommend_threshold([0.5, 0.6, 1.0], 1.2)
        assert trace["warning"] is True

    def test_proposal_clamped_to_one(self):
        proposed, _ = ta.recommend_threshold([0.99, 0.995, 1.0], 1.2)
        assert proposed <= 1.0

    def test_tiny_set(self):
        proposed, trace = ta.recommend_threshold([0.4], 1.2)
        assert proposed == pytest.approx(0.48)
        assert trace["good_summary"]["count"] == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ta.recommend_threshold([])


class TestAssistantState:
    def test_state_created_and_persisted_on_dict(self):
        recipe = {}
        ta.set_step_status(recipe, "S6", ta.STATUS_OK)
        assert recipe["assistant_state"]["steps"]["S6"]["status"] == "ok"
        assert recipe["assistant_state"]["steps"]["S6"]["updated_at"]
        assert ta.get_step_status(recipe, "S6") == "ok"

    def test_default_status(self):
        assert ta.get_step_status({}, "S4") == ta.STATUS_ACTION_NEEDED

    def test_chosen_path(self):
        recipe = {}
        ta.set_chosen_path(recipe, "reuse", "OTHER_R")
        state = ta.get_assistant_state(recipe)
        assert state["chosen_path"] == "reuse"
        assert state["chosen_recipe"] == "OTHER_R"


def _recipe_with_crops(tmp_path, crop_count=12):
    """A recipe pointing at a synthetic dataset + prepared crop folder."""
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "image_000.png").write_bytes(b"fake")

    good_dir = tmp_path / "patchcore_dataset" / "train" / "good"
    good_dir.mkdir(parents=True, exist_ok=True)
    for index in range(crop_count):
        (good_dir / f"crop_{index:03d}.png").write_bytes(b"fake")

    return {
        "recipe_name": "R",
        "package_family": "SO",
        "package_type": "SO14",
        "pin_count": 14,
        "dataset_path": str(dataset),
        "prepared_dataset_path": str(tmp_path / "patchcore_dataset"),
        "dataset_prepared": True,
    }


class TestComputeStepStatus:
    def test_complete_recipe_all_ok(self, tmp_path):
        recipe = _recipe_with_crops(tmp_path)
        ta.set_chosen_path(recipe, "train_new")
        recipe["model"] = {"trained": True}
        recipe["calibration"] = {"proposed": 0.66}
        recipe["validated"] = True

        statuses = ta.compute_step_status(recipe)
        assert all(
            entry["status"] == ta.STATUS_OK for entry in statuses.values()
        )

    def test_empty_recipe_action_needed(self):
        statuses = ta.compute_step_status({"recipe_name": "R"})
        assert statuses["S1"]["status"] == ta.STATUS_ACTION_NEEDED
        assert "pin_count" in statuses["S1"]["detail"]

    def test_no_recipe(self):
        statuses = ta.compute_step_status(None)
        assert statuses["S1"]["status"] == ta.STATUS_ACTION_NEEDED

    def test_dataset_size_never_gates(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "only.png").write_bytes(b"fake")

        statuses = ta.compute_step_status({"dataset_path": str(dataset)})
        assert statuses["S2"]["status"] == ta.STATUS_OK
        assert "1 images" in statuses["S2"]["detail"]

    def test_crops_below_ten_warn(self, tmp_path):
        recipe = _recipe_with_crops(tmp_path, crop_count=4)
        statuses = ta.compute_step_status(recipe)
        assert statuses["S3"]["status"] == ta.STATUS_WARNING

    def test_next_step_is_first_non_ok(self):
        step_id, _ = ta.next_step({"recipe_name": "R"})
        assert step_id == "S1"

    def test_page_indexes_match_wizard_order(self):
        statuses = ta.compute_step_status({"recipe_name": "R"})
        assert statuses["S3"]["page_index"] == 2
        assert statuses["S4"]["page_index"] == 3
        assert statuses["S6"]["page_index"] == 4
        assert statuses["S7"]["page_index"] == 6


class TestPreFlight:
    def test_blocked_without_crops_and_path(self):
        allowed, reasons = ta.pre_flight({"recipe_name": "R"})
        assert allowed is False
        assert any("S3" in reason for reason in reasons)
        assert any("S4" in reason for reason in reasons)

    def test_allowed_when_ready(self, tmp_path):
        recipe = _recipe_with_crops(tmp_path)
        ta.set_chosen_path(recipe, "train_new")
        allowed, reasons = ta.pre_flight(recipe)
        assert allowed is True, reasons


class TestAutoTunePlan:
    def test_plan_from_detections(self):
        detections = [
            {"width": 100 + index, "height": 50 + index} for index in range(10)
        ]
        plan = ta.auto_tune_plan(detections)
        assert plan is not None
        assert plan["width_scale"] == 1.0
        assert plan["height_scale"] == 1.0
        assert 2 <= plan["padding"] <= 60
        assert 0.08 <= plan["width_tolerance"] <= 0.5
        assert 0.08 <= plan["height_tolerance"] <= 0.5
        assert 0.08 <= plan["ratio_tolerance"] <= 0.5

    def test_too_few_detections(self):
        assert ta.auto_tune_plan([{"width": 10, "height": 5}]) is None
        assert ta.auto_tune_plan([]) is None


class TestCropQA:
    def test_bounds_and_classification(self):
        bounds = ta.qa_bounds(100, 50, 2.0, {})
        assert bounds["min_width"] == pytest.approx(75)
        assert bounds["max_width"] == pytest.approx(125)

        flagged = ta.classify_crop(
            {"width": 200, "height": 50, "ratio": 2.0}, bounds
        )
        assert len(flagged) == 1
        assert "width" in flagged[0]

        passing = ta.classify_crop(
            {"width": 100, "height": 50, "ratio": 2.0}, bounds
        )
        assert passing == []


class TestRegistryMatch:
    REGISTRY = [
        {"package_family": "SOIC", "package_type": "SO14", "pin_count": 14}
    ]

    def test_match_by_type(self):
        match = ta.match_registry("SO14", self.REGISTRY)
        assert match["pin_count"] == 14
        assert match["package_family"] == "SOIC"
        assert match["package_type"] == "SO14"

    def test_no_match(self):
        assert ta.match_registry("BGA225", self.REGISTRY) is None
        assert ta.match_registry("", self.REGISTRY) is None
        assert ta.match_registry("SO14", []) is None


class TestGeometryAnalysis:
    def test_aggregation_and_registry(self):
        detections = [
            [{"width": 100, "height": 50, "class_name": "SO14"}],
            [{"width": 102, "height": 52, "class_name": "SO14"}],
            [],
        ]
        result = ta.analyze_dataset_geometry(
            detections, registry=TestRegistryMatch.REGISTRY
        )
        assert result["images_analyzed"] == 3
        assert result["images_with_detection"] == 2
        assert result["detection_count"] == 2
        assert result["dominant_class"] == "SO14"
        assert result["registry_match"]["package_type"] == "SO14"
        assert result["size_stats"]["median_width"] == 101
        # Median of two values is their mean: (100/50 + 102/52) / 2.
        assert result["median_aspect_ratio"] == pytest.approx(
            (2.0 + 102.0 / 52.0) / 2.0
        )

    def test_empty_detections(self):
        result = ta.analyze_dataset_geometry([[], []], registry=[])
        assert result["images_analyzed"] == 2
        assert result["images_with_detection"] == 0
        assert result["registry_match"] is None
        assert "size_stats" not in result


class TestVisualSimilarity:
    def test_identical_images(self):
        rng = np.random.default_rng(7)
        image = rng.random((64, 64)).astype(np.float32)
        assert ta.visual_similarity(image, image) == pytest.approx(1.0)

    def test_unrelated_images_low(self):
        rng = np.random.default_rng(3)
        first = rng.random((64, 64)).astype(np.float32)
        second = rng.random((64, 64)).astype(np.float32)
        score = ta.visual_similarity(first, second)
        assert score is not None and score < 0.6

    def test_none_inputs(self):
        assert ta.visual_similarity(None, None) is None


class TestRecommendEntry:
    @staticmethod
    def _candidate(name):
        return {
            "recipe_name": name,
            "package_family": "SO",
            "package_type": "SO14",
            "pin_count": 14,
            "prepared_dataset_path": "",
        }

    def test_metadata_only_is_capped_at_review(self):
        result = ta.recommend_entry(
            self._candidate("NEW"), [self._candidate("X")]
        )
        assert result["decision"] == "review"
        assert result["chosen"] == "X"
        assert result["candidates"][0]["visual_similarity"] is None

    def test_metadata_only_low_scores_train_new(self):
        different = {
            "recipe_name": "Y",
            "package_family": "QFN",
            "package_type": "QFN32",
            "pin_count": 32,
            "prepared_dataset_path": "",
        }
        result = ta.recommend_entry(self._candidate("NEW"), [different])
        assert result["decision"] == "train_new"
        assert result["chosen"] is None

    def test_no_candidates(self):
        result = ta.recommend_entry(self._candidate("NEW"), [])
        assert result["decision"] == "train_new"
        assert result["candidates"] == []


class TestFinalize:
    def test_apply_finalize_bumps_version(self):
        recipe = {"version": "1.0"}
        new_version = ta.apply_finalize(recipe)
        assert new_version == "1.1"
        assert recipe["validated"] is True
        assert recipe["last_trained"]
        assert ta.get_step_status(recipe, "S7") == ta.STATUS_OK

    def test_summary_warnings_for_unfinished_recipe(self):
        summary = ta.finalize_summary({"recipe_name": "R"})
        assert any("not trained" in warning for warning in summary["warnings"])
        assert any("calibration" in warning.lower() for warning in summary["warnings"])

    def test_summary_no_warnings_when_complete(self):
        summary = ta.finalize_summary(
            {
                "recipe_name": "R",
                "model": {"trained": True},
                "calibration": {"proposed": 0.66, "warning": False},
                "roi_statistics": {"saved_count": 12},
                "anomaly_threshold": 0.66,
            }
        )
        assert summary["warnings"] == []
