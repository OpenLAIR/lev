"""Per-depth logits of a natively looped checkpoint on frozen Kev partitions, one row per question, for the offline
adaptive-depth analysis (analysis/adaptive_depth.py). One forward pass per batch reads every step, so a row carries the
logits of depths 1..R. Raw logits (temperature 1); per-depth temperatures and thresholds are fitted from the
calibration rows offline, never from the panels they are applied to. fp32 with TF32 off, like
scripts/fit_depth_temperatures.py.

    python scripts/depth_rows.py --run <checkpoint> --read evals/v7/decision-v7:calibration --read evals/v9/transfer-v9:development --out <dir>

Writes <dir>/<suite name>-<split>.json: {"run", "suite", "split", "depths", "rows": [{id, group, question, source, task,
type, variant, label, target, state_tokens, tokens, logits: [[...] per depth]}]}. Records that carry them (the Lev
suites, docs/DATA.md) add _meta's tier, difficulty and unanswerable to each row. A locked test partition needs
--allow_test (phase 5's single read). --read also takes a labelled JSONL file (kev.data.load_records; e.g. evals/external/
ekzhang-mmlupro-v1/records.jsonl, written as <parent dir>-<file stem>.json). A checkpoint that is not a native loop (a dense
control) gives one "depth": the logits of its single readout.
"""
import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, LoadOptions   # noqa: E402
from kev.data import load_records, materialize   # noqa: E402
from kev.model import ContextOverflow   # noqa: E402
from kev.suite import load_split, write_json   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--read", action="append", required=True, help="suite:split, repeatable")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass (length-sorted); 1 = the exact single-record path of scripts/fit_depth_temperatures.py, larger batches pad and are not checked against it")
    ap.add_argument("--allow_test", action="store_true", help="read locked test partitions (phase 5 only, docs/REPRODUCING.md rule 4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tok, model = Checkpoint(a.run).load(a.device, LoadOptions(temperature=1.0))
    native = getattr(model, "native_loop", False)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.read:
        t0 = time.time()
        if Path(spec).is_file():
            suite, split, records = spec, None, load_records(spec)
        else:
            suite, split = spec.rsplit(":", 1)
            records = load_split(suite, split, allow_test=a.allow_test)
        encs, skipped = [], []
        for r in records:   # a record over the context under this backbone's tokenizer is skipped and listed (kev.benchmark's skip_overlong)
            rec = materialize(r)
            try: encs.append((r, rec, model.encode(tok, rec)))
            except ContextOverflow as e: skipped.append({"id": r["_meta"]["id"], "reason": str(e)})
        order = sorted(range(len(encs)), key=lambda i: len(encs[i][2]["ids"]))
        rows, depths = [], None
        for s in range(0, len(order), a.batch):
            chunk = [encs[i] for i in order[s:s + a.batch]]
            with torch.no_grad():
                steps = model.forward_steps_batch([e for _, _, e in chunk]) if native else [model.forward_batch([e for _, _, e in chunk])]
            depths = len(steps)
            for b, (r, rec, enc) in enumerate(chunk):
                meta = r["_meta"]
                for j, q in enumerate(rec["questions"]):
                    rows.append({"id": meta["id"], "group": meta.get("group_id", meta["id"]), "question": q["qid"],
                                 "source": meta["source"], "task": q["src"], "type": q["qtype"],
                                 "variant": meta.get("variant", "clean"), "label": int(q["label"]), "target": q.get("target"),
                                 "state_tokens": enc["state_tokens"], "tokens": len(enc["ids"]),
                                 "logits": [[float(x) for x in step[b][j].float().cpu()] for step in steps]}
                                | {k: meta[k] for k in ("tier", "difficulty", "unanswerable", "report_only") if k in meta})
        name = f"{Path(suite).name}-{split}" if split else f"{Path(suite).parent.name}-{Path(suite).stem}"
        write_json(out_dir / f"{name}.json", {"run": a.run, "suite": suite, "split": split, "depths": depths, "skipped": skipped, "rows": rows})
        print(f"{name}: {len(records)} records ({len(skipped)} skipped as overlong), {len(rows)} questions, {depths} depths, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
