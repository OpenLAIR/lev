"""Per-record latency of one checkpoint at every depth, for the offline cascade comparison (study, plan M1).

One request at a time, bf16 (the serving dtype), a short warm-up, the model computing only as many loop steps as the depth
read (model.set_depth), as scripts/jevbench_eval.py --latency times JevBench items. The records are a fixed sample of each
panel (the --n records whose crc32(id) is smallest, so every checkpoint times the same records); each is timed --repeats
times and the median kept. A record over this backbone's context is skipped, as scripts/depth_rows.py skips it.
Per depth a least-squares fit latency = a + b * tokens (the record's encoded length) lets analysis/cascade.py price every
row of a panel, including rows outside the sample. Run every checkpoint of a comparison on the same GPU, one after another
(roadmap rule 5: same harness, card, dtype and batch setting).

    python scripts/latency_table.py --run <checkpoint> --read evals/levbench-v1:development --read evals/breadth-v1:development --out <dir>

Writes <dir>/latency.json: {"run", "gpu", "dtype", "depths", "panels": {name: {"records": [{id, tokens, state_tokens,
questions, seconds: [per depth]}]}}, "fit": [{"depth", "a_s", "b_s_per_token", "r2", "p50_s"}]}.
"""
import argparse, json, statistics, sys, time, zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, LoadOptions   # noqa: E402
from kev.data import load_records, materialize   # noqa: E402
from kev.model import ContextOverflow   # noqa: E402
from kev.suite import load_split, write_json   # noqa: E402


def timed(model, enc, repeats):
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter(); model.probs(enc); torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--read", action="append", required=True, help="suite:split or a labelled JSONL file, repeatable")
    ap.add_argument("--n", type=int, default=300, help="records per panel")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok, model = Checkpoint(a.run).load("cuda", LoadOptions(dtype=torch.bfloat16))
    native = getattr(model, "native_loop", False)
    depths = list(range(1, model.depth + 1)) if native else [None]
    report = {"run": a.run, "gpu": torch.cuda.get_device_name(), "dtype": str(next(model.parameters()).dtype),
              "depths": len(depths), "n": a.n, "repeats": a.repeats, "panels": {}}
    samples = {}
    for spec in a.read:
        if Path(spec).is_file():
            records, name = load_records(spec), f"{Path(spec).parent.name}-{Path(spec).stem}"
        else:
            suite, split = spec.rsplit(":", 1); records, name = load_split(suite, split), f"{Path(suite).name}-{split}"
        records = sorted(records, key=lambda r: zlib.crc32(r["_meta"]["id"].encode()))
        encs = []
        for r in records:
            if len(encs) == a.n: break
            rec = materialize(r)
            try: encs.append((r["_meta"]["id"], rec, model.encode(tok, rec)))
            except ContextOverflow: continue
        samples[name] = encs
    for name, encs in samples.items():
        rows = [{"id": i, "tokens": len(e["ids"]), "state_tokens": e["state_tokens"], "questions": len(rec["questions"]), "seconds": []}
                for i, rec, e in encs]
        for k in depths:
            if native: model.set_depth(k)
            for _, _, e in encs[:5]: model.probs(e)   # warm-up
            for row, (_, _, e) in zip(rows, encs): row["seconds"].append(timed(model, e, a.repeats))
        report["panels"][name] = {"records": rows}
        print(f"{name}: {len(rows)} records timed at {len(depths)} depths", flush=True)
    if native: model.set_depth(depths[-1])
    allrows = [r for p in report["panels"].values() for r in p["records"]]
    x = np.array([r["tokens"] for r in allrows], float)
    report["fit"] = []
    for j in range(len(depths)):
        y = np.array([r["seconds"][j] for r in allrows])
        b, c = np.polyfit(x, y, 1); pred = c + b * x
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        report["fit"].append({"depth": j + 1, "a_s": float(c), "b_s_per_token": float(b), "r2": float(r2), "p50_s": float(np.median(y))})
        print(f"depth {j + 1}: p50 {np.median(y) * 1000:.1f} ms, latency = {c * 1000:.1f} ms + {b * 1e6:.2f} us/token (r2 {r2:.2f})", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "latency.json", report)


if __name__ == "__main__":
    main()
