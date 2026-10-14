"""Paired comparison of a recipe arm against its baseline checkpoint (study, study: docs/REPRODUCING.md), both read
by scripts/run_read.sh and analysed by analysis/adaptive_depth.py (adaptive.json `stops`). Per panel, on the questions
both read (answerable, non-report-only rows): the difference in adaptive accuracy (each under its own stopper), in deepest
accuracy, at a common absolute depth (--depth k: the k-th step of both; for step-count arms), and in "adaptive minus its
own fixed-depth frontier". Paired 95% CIs resample record groups (2,000 draws, seed 0).
Registered win rule (study): adaptive accuracy significantly higher on eval-v1 dev or breadth-v1 dev (CI lower bound
> 0), not significantly lower on the other (CI upper bound > 0), and the arm's own G1-G3 hold on eval-v1 dev.

    python analysis/arm_compare.py --arm NAME=<dir> --base NAME=<dir> [--base_analysis analysis-xfit-cal] [--depth 4] --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import bootstrap, load_rows, prepare  # noqa: E402

PANELS = ("eval-v1-development", "breadth-v1-development")


def per_question(d, analysis_dir, panel, depth):
    a = json.loads((Path(d) / analysis_dir / "adaptive.json").read_text(encoding="utf-8"))
    rows = load_rows(Path(d) / "rows" / f"{panel}.json")[1]
    conf, right = prepare(rows, a["temperatures"])
    stop = np.array(a["stops"][panel]); D = right.shape[1]
    ans = np.array([not r["unanswerable"] and not r["report_only"] for r in rows])
    c = float(np.mean(stop[ans] + 1)) / D
    lo = min(max(int(np.floor(c * D)), 1), D - 1); lam = min(max(c * D - lo, 0.0), 1.0)
    adaptive = right[np.arange(len(rows)), stop].astype(float)
    frontier = (1 - lam) * right[:, lo - 1] + lam * right[:, lo]
    k = min(depth or D, D) - 1
    return {(r["id"], r["question"] or ""): (r["group"], adaptive[i], float(right[i, -1]), float(right[i, k]), adaptive[i] - frontier[i])
            for i, r in enumerate(rows) if ans[i]}, a["panels"][panel]["gates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=dir, repeatable (each against --base)")
    ap.add_argument("--base", required=True, help="NAME=dir")
    ap.add_argument("--arm_analysis", default="analysis-xfit")
    ap.add_argument("--base_analysis", default="analysis-xfit-cal")
    ap.add_argument("--depth", type=int, default=0, help="also compare both at this absolute step (0 = no)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bname, bdir = a.base.split("=", 1)
    pct = lambda x: f"{x['delta'] * 100:+.1f} [{x['ci95'][0] * 100:+.1f}, {x['ci95'][1] * 100:+.1f}]"
    report, lines = {}, [f"Baseline: {bname}", "", "| arm | panel | n | Δ adaptive [CI] | Δ deepest [CI] | " + (f"Δ at step {a.depth} [CI] | " if a.depth else "")
                         + "Δ (adaptive − own frontier) [CI] | arm G1 / G2 / G3 |", "|---|---|---|---|---|" + ("---|" if a.depth else "") + "---|---|"]
    for spec in a.arm:
        name, d = spec.split("=", 1); report[name] = {}
        pooled = {"g": [], "over_frontier": [], "adaptive": []}   # both panels together (study's registered primary metric)
        for p in PANELS:
            A, gates = per_question(d, a.arm_analysis, p, a.depth); B, _ = per_question(bdir, a.base_analysis, p, a.depth)
            keys = sorted(set(A) & set(B)); g = [A[k][0] for k in keys]
            diff = lambda j: np.array([A[k][j] - B[k][j] for k in keys])
            pooled["g"] += [f"{p}/{x}" for x in g]; pooled["over_frontier"] += list(diff(4)); pooled["adaptive"] += list(diff(1))
            r = {"questions": len(keys), "adaptive": diff(1), "deepest": diff(2), "at_depth": diff(3), "over_frontier": diff(4)}
            r = {"questions": len(keys), "arm_gates": gates} | {m: {"delta": float(v.mean()), "ci95": bootstrap(g, v)} for m, v in r.items() if m != "questions"}
            report[name][p] = r
            lines.append(f"| {name} | {p} | {len(keys)} | {pct(r['adaptive'])} | {pct(r['deepest'])} | " + (f"{pct(r['at_depth'])} | " if a.depth else "")
                         + f"{pct(r['over_frontier'])} | " + " / ".join("P" if gates[x] else "F" for x in ("G1", "G2", "G3")) + " |")
        report[name]["pooled"] = {m: {"delta": float(np.mean(pooled[m])), "ci95": bootstrap(pooled["g"], np.array(pooled[m]))} for m in ("adaptive", "over_frontier")}
        lines.append(f"| {name} | both panels pooled | {len(pooled['g'])} | {pct(report[name]['pooled']['adaptive'])} | | " + ("| " if a.depth else "")
                     + f"{pct(report[name]['pooled']['over_frontier'])} | |")
        e, b = report[name]["eval-v1-development"], report[name]["breadth-v1-development"]
        win = ((e["adaptive"]["ci95"][0] > 0 and b["adaptive"]["ci95"][1] > 0) or (b["adaptive"]["ci95"][0] > 0 and e["adaptive"]["ci95"][1] > 0)) and all(e["arm_gates"].values())
        report[name]["wins"] = bool(win)
    lines += ["", "Registered win rule: " + ", ".join(f"{n}: {'wins' if r['wins'] else 'does not win'}" for n, r in report.items())]
    (out / "compare.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
