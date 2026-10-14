"""study (docs/REPRODUCING.md): one checkpoint per seed, pooled. For each panel, every question's
correctness under the adaptive stops (adaptive.json `stops`), at the deepest depth and on the fixed-depth frontier at that
seed's own cost is averaged over the seeds; the paired 95% CI of (adaptive - deepest) and (adaptive - frontier) resamples
record groups (2,000 draws, seed 0), as analysis/adaptive_depth.py does for one seed. Answerable, non-report-only rows.

    python analysis/seed_pool.py --run s0=<dir> --run s1=<dir> --run s2=<dir> --analysis s0=analysis-xfit-cal ... --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import bootstrap, load_rows, prepare  # noqa: E402

PANELS = ("eval-v1-development", "breadth-v1-development", "jevbench-public")


def per_question(d, analysis_dir, panel):
    a = json.loads((Path(d) / analysis_dir / "adaptive.json").read_text(encoding="utf-8"))
    if panel == "jevbench-public":
        from adaptive_depth import load_jevbench
        rows = load_jevbench(json.loads((Path(d) / "inputs.json").read_text(encoding="utf-8"))["jevbench_rows"])[1]
    else:
        rows = load_rows(Path(d) / "rows" / f"{panel}.json")[1]
    conf, right = prepare(rows, a["temperatures"])
    stop = np.array(a["stops"][panel]); D = right.shape[1]
    ans = np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])
    c = float(np.mean(stop[ans] + 1)) / D
    lo = min(max(int(np.floor(c * D)), 1), D - 1); lam = min(max(c * D - lo, 0.0), 1.0)
    keys = [(r["id"], r.get("question") or "") for r in rows]
    if len(set(keys)) != len(keys): raise ValueError(f"{d} {panel}: question keys are not unique")
    adaptive = right[np.arange(len(rows)), stop].astype(float)
    frontier = (1 - lam) * right[:, lo - 1] + lam * right[:, lo]
    return {k: (r["group"], adaptive[i], right[i, -1], frontier[i], stop[i]) for i, (k, r) in enumerate(zip(keys, rows)) if ans[i]}, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="SEED=dir (rows/, inputs.json, <analysis>/adaptive.json)")
    ap.add_argument("--analysis", action="append", default=[], help="SEED=analysis dir name (default analysis-xfit)")
    ap.add_argument("--panel", action="append", default=[], help="panels (default: eval-v1 / breadth-v1 development and JevBench public)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, an = dict(x.split("=", 1) for x in a.run), dict(x.split("=", 1) for x in a.analysis)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report, lines = {"seeds": list(runs)}, ["| panel | n | seeds | deepest (mean ± sd) | adaptive (mean ± sd) | cost (mean) | Δ vs deepest, pooled [CI] | Δ vs frontier, pooled [CI] |",
                                              "|---|---|---|---|---|---|---|---|"]
    for p in (a.panel or PANELS):
        per = {s: per_question(d, an.get(s, "analysis-xfit"), p) for s, d in runs.items()}
        common = set.intersection(*(set(q) for q, _ in per.values()))
        keys = sorted(common)
        groups = [per[next(iter(per))][0][k][0] for k in keys]
        stack = lambda j: np.mean([[per[s][0][k][j] for k in keys] for s in per], 0)
        adaptive, deep, front = stack(1), stack(2), stack(3)
        acc = {s: (float(np.mean([q[k][1] for k in keys])), float(np.mean([q[k][2] for k in keys])), c) for s, (q, c) in per.items()}
        ad, dp, cs = (np.array([v[i] for v in acc.values()]) for i in range(3))
        r = {"questions": len(keys), "per_seed": acc,
             "pooled_minus_deepest": {"delta": float((adaptive - deep).mean()), "ci95": bootstrap(groups, adaptive - deep)},
             "pooled_minus_frontier": {"delta": float((adaptive - front).mean()), "ci95": bootstrap(groups, adaptive - front)}}
        report[p] = r
        md, mf = r["pooled_minus_deepest"], r["pooled_minus_frontier"]
        lines.append(f"| {p} | {len(keys)} | {len(per)} | {dp.mean():.3f} ± {dp.std(ddof=1):.3f} | {ad.mean():.3f} ± {ad.std(ddof=1):.3f} | {cs.mean():.2f} | "
                     f"{md['delta'] * 100:+.1f} [{md['ci95'][0] * 100:+.1f}, {md['ci95'][1] * 100:+.1f}] | {mf['delta'] * 100:+.1f} [{mf['ci95'][0] * 100:+.1f}, {mf['ci95'][1] * 100:+.1f}] |")
    (out / "pooled.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "pooled.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
