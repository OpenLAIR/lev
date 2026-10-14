"""Paired item-level comparisons on JevBench public items for the pre-registered claim rules (PROPOSAL.md §9):
(b) looping helps: Looped-Kev depth k vs its depth 1, and vs the dense control; (a) vs Jev on the same items.

Each side is a set of per-item outcomes: scripts/jevbench_eval.py rows (fp32 read) or Jev's committed public per-item
outcomes. Delta = accuracy difference on the items both scored, item bootstrap 95% interval (10,000 resamples, seed 0,
every item its own group, as Kev's scripts/jevbench_paired.py), discordant counts and the exact two-sided McNemar p;
overall and on the hard tier.

    python analysis/jevbench_paired.py --loop outputs/jevbench/Lev-1.4B-s0-fp32 --control outputs/jevbench/Qwen3-1.7B-s0-fp32 --out results/jevbench/paired.json
"""
import argparse, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES, SEED = 10000, 0


def rows(path):
    return {r["id"]: (bool(r["correct"]), r["tier"]) for r in json.loads(Path(path).read_text(encoding="utf-8"))}


def jev(root):
    per = json.load(open(root / "results/v1.2/jevbench-v1.2-per-task.json", encoding="utf-8"))
    tier = {t["id"]: t["tier"] for t in per["tasks"]}
    return {k: (o == "c", tier[k]) for k, (o, c) in per["systems"]["jev-1.13.0"]["public_tasks"].items()}


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(cand, ref, tier=None):
    keys = sorted(k for k in cand.keys() & ref.keys() if tier is None or cand[k][1] == tier)
    a = np.array([cand[k][0] for k in keys], float); b = np.array([ref[k][0] for k in keys], float)
    d = a - b
    idx = np.random.default_rng(SEED).integers(0, len(d), (SAMPLES, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    up, down = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"n": len(keys), "cand_acc": float(a.mean()), "ref_acc": float(b.mean()), "delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "newly_right": up, "newly_wrong": down, "mcnemar_p": mcnemar(up, down)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", required=True, help="Looped-Kev fp32 read dir (rows-depth<k>.json)")
    ap.add_argument("--control", required=True, help="dense control fp32 read dir")
    ap.add_argument("--jevbench", default=str(ROOT / "vendor/jevbench"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loop = {int(p.stem.removeprefix("rows-depth")): rows(p) for p in sorted(Path(a.loop).glob("rows-depth*.json"))}
    control = rows(Path(a.control) / "rows-depth1.json"); j = jev(Path(a.jevbench))
    out = {}
    for k, r in sorted(loop.items()):
        for name, ref in [("depth1", loop[1]), ("control", control), ("jev", j)]:
            if name == "depth1" and k == 1: continue
            out[f"depth{k}_vs_{name}"] = {"all": paired(r, ref), "hard": paired(r, ref, "hard")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("| comparison | tier | n | candidate | reference | Δ acc [95% CI] | +/− items | McNemar p |\n|---|---|---|---|---|---|---|---|")
    for name, v in out.items():
        for t, s in v.items():
            print(f"| {name} | {t} | {s['n']} | {s['cand_acc']:.3f} | {s['ref_acc']:.3f} | {s['delta']*100:+.1f} pp [{s['ci95'][0]*100:+.1f}, {s['ci95'][1]*100:+.1f}] | "
                  f"+{s['newly_right']} / −{s['newly_wrong']} | {s['mcnemar_p']:.3f} |")


if __name__ == "__main__":
    main()
