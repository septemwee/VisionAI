import cv2
import numpy as np

from services.synthetic_defect_service import generate_synthetic_validation
from services.training_assistant import recommend_threshold_with_anomalies


def test_synthetic_generator_is_repeatable_and_keeps_sources_untouched(tmp_path):
    source = tmp_path / "good"
    source.mkdir()
    image = np.full((80, 120, 3), 110, np.uint8)
    cv2.rectangle(image, (10, 10), (109, 69), (160, 160, 160), -1)
    cv2.imwrite(str(source / "sample.png"), image)
    original = (source / "sample.png").read_bytes()
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = generate_synthetic_validation(source, first)
    generate_synthetic_validation(source, second)
    assert len(manifest) == 4
    assert {item["type"] for item in manifest} == {
        "scratch", "spot", "chip", "pin_discoloration"
    }
    assert (source / "sample.png").read_bytes() == original
    for item in manifest:
        assert (first / item["file"]).read_bytes() == (second / item["file"]).read_bytes()


def test_threshold_uses_held_out_good_and_synthetic_scores():
    threshold, trace = recommend_threshold_with_anomalies(
        [0.10, 0.11, 0.12, 0.13, 0.14], [0.20, 0.25, 0.30])
    assert 0.14 < threshold < 0.20
    assert trace["expected_false_alarms"] == 0
    assert trace["synthetic_detection_rate"] == 1.0
    assert trace["warning"] is False


def test_threshold_reports_synthetic_detection_failure_not_score_ceiling():
    _, trace = recommend_threshold_with_anomalies(
        [0.10, 0.11, 0.12], [0.105, 0.11, 0.115]
    )
    assert trace["warning"] is True
    assert "synthetic" in trace["warning_reason"].lower()


def test_optional_alpha_masks_preserve_images_and_describe_modified_pixels(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    image = np.full((80, 120, 3), 110, np.uint8)
    cv2.imwrite(str(source / 'sample.png'), image)
    baseline, output, masks = tmp_path / 'baseline', tmp_path / 'images', tmp_path / 'masks'
    original = generate_synthetic_validation(source, baseline)
    manifest = generate_synthetic_validation(source, output, mask_dir=masks)
    assert manifest == original
    assert len(list(output.glob('*.png'))) == 4
    for item in manifest:
        name = item['file']
        assert (baseline / name).read_bytes() == (output / name).read_bytes()
        # Ultralytics' imread wrapper retains a singleton grayscale channel.
        alpha = np.squeeze(cv2.imread(str(masks / name), cv2.IMREAD_GRAYSCALE))
        changed = np.any(cv2.imread(str(output / name)) != image, axis=2)
        assert alpha.shape == image.shape[:2]
        assert np.any(alpha > 25)
        assert np.all(changed[alpha > 25])
        assert not changed[alpha == 0].any()
