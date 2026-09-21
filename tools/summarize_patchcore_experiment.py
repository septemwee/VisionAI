"""Validate recorded results and export a compact comparison table."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/patchcore_experiment_20260921"
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(data["results"]) == 12
    rows = []
    for config in data["results"]:
        calibration = [r for r in config["rows"] if r["role"] == "calibration"]
        test = [r for r in config["rows"] if r["role"] == "test"]
        assert len(calibration) == len(test) == 105
        threshold = config["threshold"]
        frozen = json.loads((out / f"frozen_calibration_{config['size']}.json").read_text())
        match = next(c for c in frozen if c["bank"] == config["bank"]
                     and c["neighbors"] == config["neighbors"])
        assert match["threshold"] == threshold
        assert match["rows"] == calibration
        for kind, summary in config["test"].items():
            assert summary["count"] == sum(r["kind"] == kind for r in test)
            assert summary["above"] == sum(r["kind"] == kind and r["score"] > threshold for r in test)
        rows.append(dict(
            image_size=config["size"], bank=config["bank"], neighbors=config["neighbors"],
            threshold=threshold, calibration_detected=config["calibration"]["synthetic_detected"],
            calibration_synthetic_total=84,
            test_detected=sum(v["above"] for k, v in config["test"].items() if k != "good"),
            test_synthetic_total=84, test_false_rejects=config["test"]["good"]["above"],
            test_good_total=21, test_synthetic_auc=config["test_auc"],
            median_ms=config["latency_median_ms"], p95_ms=config["latency_p95_ms"],
        ))
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Verified 12 configurations: frozen thresholds, calibration rows and test counts.")
    runtime_path = out / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        print("Runtime measurements:", len(runtime["measurements"]))
        print("Baseline evaluated images:", len(runtime["baseline_test"]))
        for mode in ("image_fail", "full_verdict_fail"):
            print(mode, {kind: sum(r[mode] for r in runtime["baseline_test"] if r["kind"] == kind)
                         for kind in sorted({r["kind"] for r in runtime["baseline_test"]})})
        if len(runtime["measurements"]) == 6 and len(runtime["baseline_test"]) == 105:
            selected = next(c for c in data["results"]
                            if all(c[k] == v for k, v in runtime["selected"].items()))
            test = [r for r in selected["rows"] if r["role"] == "test"]
            assert [(r["file"], r["kind"]) for r in test] == [
                (r["file"], r["kind"]) for r in runtime["baseline_test"]]
            expected = [r["score"] for r in test if r["kind"] == "good"][:8]
            for measurement in runtime["measurements"]:
                assert len(measurement["latency_samples_ms"]) == 8
                if measurement["model"] == "calibration_selected":
                    assert max(abs(a - b) for a, b in zip(expected, measurement["scores"])) < 1e-5
            assert not any("unavailable" in r["reason"].lower()
                           or "invalid" in r["reason"].lower()
                           for r in runtime["baseline_test"])
            print("Runtime complete: identical evaluation files, candidate scoring parity and valid baseline verdicts.")
        else:
            print("Runtime comparison incomplete; not validated as a completed run.")


if __name__ == "__main__":
    main()
