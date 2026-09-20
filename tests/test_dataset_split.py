import json

from services.dataset_split_service import assign_good_splits, sample_group, write_split_manifest


def test_burst_frames_stay_in_one_split():
    names = [f"part_a_{index:04d}.png" for index in range(5)] + [
        f"part_{letter}_001.png" for letter in "bcdefghijkl"]
    assignments, counts = assign_good_splits(names)
    assert len({assignments[name] for name in names[:5]}) == 1
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


def test_group_name_removes_frame_counter():
    assert sample_group("lotA_20260920_0003.png") == "lota"
