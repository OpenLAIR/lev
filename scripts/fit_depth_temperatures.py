"""Fit one calibration temperature per depth on a held-out Kev partition (never on JevBench), the way
scripts/calibrate_checkpoint.py fits a released checkpoint's single temperature: minimum mean NLL over a log grid
(kev.metrics.TEMPERATURE_FIT: 121 points on 0.25..4, every question weighted equally). A natively looped checkpoint gets
a temperature for each step from one pass per record; a dense one gets a single temperature.

    python scripts/fit_depth_temperatures.py --run <checkpoint> --suite evals/v7/decision-v7 --split calibration --out <json>
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, LoadOptions   # noqa: E402
from kev.data import materialize   # noqa: E402
from kev.suite import load_split   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tok, model = Checkpoint(a.run).load(a.device, LoadOptions(temperature=1.0))
    native = getattr(model, "native_loop", False)
    logits = None   # [depth] -> list of (logit vector, label)
    for r in load_split(a.suite, a.split):
        rec = materialize(r)
        enc = model.encode(tok, rec)
        with torch.no_grad():
            steps = model.forward_steps_batch([enc]) if native else [model.forward_batch([enc])]
        if logits is None: logits = [[] for _ in steps]
        for k, step in enumerate(steps):
            for z, y in zip(step[0], enc["labels"]): logits[k].append((z.float().cpu().numpy(), y))
    grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))
    out = {"run": a.run, "suite": a.suite, "split": a.split, "native_loop": native, "questions": len(logits[0]), "temperatures": [], "nll_raw": [], "nll_fit": []}
    for rows in logits:
        def nll(t):
            return float(np.mean([np.logaddexp.reduce(z / t) - z[y] / t for z, y in rows]))
        best = min(grid, key=nll)
        out["temperatures"].append(float(best)); out["nll_raw"].append(nll(1.0)); out["nll_fit"].append(nll(best))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("questions", "temperatures", "nll_raw", "nll_fit")}))


if __name__ == "__main__":
    main()
