"""Deterministic Train/Calibration/Test assignment for prepared good crops."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SPLIT_VERSION = "good_split_v1"


def assign_good_splits(filenames, calibration_ratio=0.15, test_ratio=0.15):
    """Return stable per-image Train/Calibration/Test assignments.

    Source filenames in this project do not encode burst groups, so each
    image is assigned independently. Hash ordering keeps an unchanged input
    set reproducible without relying on filesystem order.
    """
    names = sorted(set(str(name) for name in filenames))
    ranked = sorted(names, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
    result = {}
    total = max(len(names), 1)
    calibration_target = round(total * calibration_ratio)
    test_target = round(total * test_ratio)
    counts = {"train": 0, "calibration": 0, "test": 0}
    for name in ranked:
        if counts["calibration"] < calibration_target:
            split = "calibration"
        elif counts["test"] < test_target:
            split = "test"
        else:
            split = "train"
        result[name] = split
        counts[split] += 1
    # Small datasets still need a usable train set.
    if counts["train"] == 0:
        for name in names:
            result[name] = "train"
        counts = {"train": len(names), "calibration": 0, "test": 0}
    return result, counts


def write_split_manifest(dataset_root, assignments, counts):
    payload = {
        "version": SPLIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "assignments": assignments,
    }
    path = Path(dataset_root) / "split_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
