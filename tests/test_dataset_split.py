import json

from services.dataset_split_service import assign_good_splits, write_split_manifest


def test_each_image_is_assigned_independently():
    names = [f"part_a_{index:04d}.png" for index in range(5)] + [
        f"part_{letter}_001.png" for letter in "bcdefghijkl"]
    assignments, counts = assign_good_splits(names)
    assert set(assignments) == set(names)
    assert sum(counts.values()) == len(names)
    assert counts["train"] > 0


def test_split_is_repeatable_and_manifest_is_auditable(tmp_path):
    names = [f"sample_{letter}.png" for letter in "abcdefghij"]
    first, counts = assign_good_splits(names)
    second, _ = assign_good_splits(reversed(names))
    assert first == second
    path = write_split_manifest(tmp_path, first, counts)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["assignments"] == first
    assert data["counts"] == counts
