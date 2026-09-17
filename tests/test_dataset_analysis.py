"""Tests for dataset quality screening with synthetic images."""

import cv2
import numpy as np

from services.training_assistant import assess_dataset


def test_flags_blur_brightness_and_corrupt(tmp_path):
    rng = np.random.default_rng(11)
    sharp = (rng.random((120, 160)) * 255).astype("uint8")
    paths = []

    for index in range(6):
        path = tmp_path / f"sharp_{index}.png"
        cv2.imwrite(str(path), sharp)
        paths.append(path)

    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    blurry_path = tmp_path / "blurry.png"
    cv2.imwrite(str(blurry_path), blurry)
    paths.append(blurry_path)

    dark = (sharp * 0.1).astype("uint8")
    dark_path = tmp_path / "dark.png"
    cv2.imwrite(str(dark_path), dark)
    paths.append(dark_path)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    paths.append(corrupt)

    result = assess_dataset([str(path) for path in paths])
    by_name = {entry["name"]: entry for entry in result["results"]}

    assert "corrupt" in by_name["corrupt.png"]["flags"]
    assert "blurry" in by_name["blurry.png"]["flags"]
    assert "brightness" in by_name["dark.png"]["flags"]
    assert by_name["sharp_0.png"]["flags"] == []


def test_progress_callback(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.full((40, 40), 128, dtype="uint8"))

    calls = []
    assess_dataset(
        [str(path)], progress_cb=lambda done, total: calls.append((done, total))
    )
    assert calls == [(1, 1)]
