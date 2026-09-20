"""Deterministic Train/Calibration/Test assignment for prepared good crops."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SPLIT_VERSION = "good_split_v1"


def sample_group(filename):
    """Group burst frames whose names differ only by trailing counters/times."""
    stem = Path(filename).stem.lower()
    stem = re.sub(r"(?:[_-]?\d{2,})+$", "", stem).strip("_-")
    return stem or Path(filename).stem.lower()


def assign_good_splits(filenames, calibration_ratio=0.15, test_ratio=0.15):
    """Return stable assignments while keeping each inferred group together."""
    names = sorted(set(str(name) for name in filenames))
    groups = {}
    for name in names:
        groups.setdefault(sample_group(name), []).append(name)
    ranked = sorted(groups, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
    result = {}
    total = max(len(names), 1)
    calibration_target = round(total * calibration_ratio)
    test_target = round(total * test_ratio)
    counts = {"train": 0, "calibration": 0, "test": 0}
    for group in ranked:
        items = groups[group]
        if counts["calibration"] < calibration_target:
            split = "calibration"
        elif counts["test"] < test_target:
            split = "test"
        else:
            split = "train"
        for name in items:
            result[name] = split
        counts[split] += len(items)
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
