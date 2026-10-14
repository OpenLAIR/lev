"""Fetch frozen public training partitions and assemble Route B's extra data; no model execution."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import load_split, read_jsonl, write_jsonl


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    runtime = root
    parts = {"night2": read_jsonl(root / "evals/night2/all.jsonl")}
    for suite in ("hard-v1", "devtools-v1", "documents-v1"):
        parts[suite] = load_split(root / "evals" / suite, "train")
    base = load_split(root / "evals/v7/decision-v7", "train")
    calibration = load_split(root / "evals/v7/decision-v7", "calibration")
    out = runtime / "data/local/kev08b-release-mix.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, [r for rows in parts.values() for r in rows])
    report = {
        "components_before_token_filter": {key: len(rows) for key, rows in parts.items()},
        "extra_records": sum(map(len, parts.values())),
        "decision_v7_replay_records": len(base),
        "calibration_records": len(calibration),
        "output": str(out),
        "note": "Reconstructed from the published recipe. "
                "Concatenation: night2, hard-v1, devtools-v1, documents-v1; "
                "decision-v7 is added by --replay 12576. Trainer applies tokenizer-specific "
                "max_state=5120 filtering. No test partitions or JevBench items in training.",
    }
    (runtime / "outputs").mkdir(parents=True, exist_ok=True)
    (runtime / "outputs/training-data.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
