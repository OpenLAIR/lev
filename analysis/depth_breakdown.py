"""Where depth helps (plan analysis A6): the depth curve and on-demand depth broken down by question properties, and how
answers change from step to step.

For one checkpoint's reads (scripts/depth_rows.py rows + the stops of analysis/adaptive_depth.py --rule agree --fit xfit,
whose adaptive.json keeps them), every answerable, non-report-only question of a panel is grouped by
  options      option count: 2 / 3-4 / 5-9 / 10+
  state        state tokens (the encoded state, <state> included): < 128 / 128-511 / 512-2047 / >= 2048
  type         question type (choice / noul / score ...)
  area         breadth-v1's five areas (evals/breadth-v1/manifest.json), breadth-v1 only
and per group reports the accuracy at every depth, the depth gain (deepest - depth 1), on-demand accuracy and cost (the
xfit stops, question cost), and on-demand minus the group's own fixed-depth frontier with its paired 95% CI (record
groups, 2,000 draws, seed 0). Temperatures do not change argmaxes, so accuracy needs none; the stops already embed them.
Flips: for each pair of consecutive depths, the share of questions that go wrong -> right and right -> wrong, and the
share that is right at some depth but wrong at the deepest ("overthinking").

    python analysis/depth_breakdown.py --dir <read dir with rows/ and analysis-xfit-cal/> --panel eval-v1-development --panel breadth-v1-development --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import bootstrap   # noqa: E402

AREAS = {d: a for a, s in json.loads(Path("evals/breadth-v1/manifest.json").read_text(encoding="utf-8"))["areas"].items() for d in s["datasets"]}


def bucket(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v < e: return lab
    return labels[-1]


def keys_of(r, panel):
    n = len(r["logits"][0])
    out = {"options": bucket(n, [3, 5, 10], ["2", "3-4", "5-9", "10+"]),
           "state": bucket(r["state_tokens"], [128, 512, 2048], ["<128", "128-511", "512-2047", ">=2048"]),
           "type": r.get("type") or "?"}
    if panel.startswith("breadth-v1"): out["area"] = AREAS.get(r["source"], "?")
    return out


def group_stats(right, stop, groups):
    D = right.shape[1]
    adaptive = right[np.arange(len(stop)), stop].astype(float)
    c = float(np.mean(stop + 1)) / D
    lo = min(max(int(np.floor(c * D)), 1), D - 1); lam = min(max(c * D - lo, 0.0), 1.0)
    frontier = (1 - lam) * right[:, lo - 1] + lam * right[:, lo]
    return {"n": int(len(stop)), "fixed": [float(x) for x in right.mean(0)], "gain": float(right[:, -1].mean() - right[:, 0].mean()),
            "adaptive": float(adaptive.mean()), "cost": c, "vs_frontier": float((adaptive - frontier).mean()),
            "vs_frontier_ci": bootstrap(groups, adaptive - frontier) if len(set(groups)) > 1 else [float("nan")] * 2}


def flips(right):
    D = right.shape[1]
    out = {"wrong_to_right": [], "right_to_wrong": []}
    for k in range(1, D):
        out["wrong_to_right"].append(float((~right[:, k - 1] & right[:, k]).mean()))
        out["right_to_wrong"].append(float((right[:, k - 1] & ~right[:, k]).mean()))
    out["right_somewhere_wrong_at_deepest"] = float((right[:, :-1].any(1) & ~right[:, -1]).mean())
    out["wrong_everywhere"] = float((~right.any(1)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="a read dir: rows/<panel>.json and <analysis>/adaptive.json")
    ap.add_argument("--analysis", default="analysis-xfit-cal")
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--min_n", type=int, default=50, help="groups with fewer questions are listed without a CI")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    adaptive = json.loads((Path(a.dir) / a.analysis / "adaptive.json").read_text(encoding="utf-8"))
    report, md = {"dir": a.dir, "panels": {}}, [f"# Where depth helps: {a.dir}", ""]
    for p in a.panel:
        d = json.loads((Path(a.dir) / "rows" / f"{p}.json").read_text(encoding="utf-8"))
        rows, stop_all = d["rows"], np.array(adaptive["stops"][p])
        if len(rows) != len(stop_all): raise SystemExit(f"{p}: {len(rows)} rows, {len(stop_all)} stops")
        keep = [i for i, r in enumerate(rows) if not r.get("unanswerable") and not r.get("report_only")]
        rows = [rows[i] for i in keep]; stop = stop_all[keep]
        right = np.array([[int(np.argmax(z)) == r["label"] for z in r["logits"]] for r in rows])
        groups = [r["group"] for r in rows]
        D = right.shape[1]
        rep = {"all": group_stats(right, stop, groups), "flips": flips(right)}
        md += [f"## {p} ({len(rows)} questions)", "", "| key | group | n | " + " | ".join(f"d{k + 1}" for k in range(D)) +
               " | gain | on-demand | cost | Δ vs frontier [CI] |", "|---|---|---|" + "---|" * D + "---|---|---|---|"]
        def line(key, g, s):
            ci = "" if s["n"] < a.min_n or np.isnan(s["vs_frontier_ci"][0]) else f" [{s['vs_frontier_ci'][0] * 100:+.1f}, {s['vs_frontier_ci'][1] * 100:+.1f}]"
            return (f"| {key} | {g} | {s['n']} | " + " | ".join(f"{x:.3f}" for x in s["fixed"]) +
                    f" | {s['gain'] * 100:+.1f} | {s['adaptive']:.3f} | {s['cost']:.2f} | {s['vs_frontier'] * 100:+.1f}{ci} |")
        md.append(line("all", "", rep["all"]))
        ks = [keys_of(r, p) for r in rows]
        for key in ks[0]:
            rep[key] = {}
            for g in sorted({k[key] for k in ks}, key=str):
                m = np.array([k[key] == g for k in ks])
                s = group_stats(right[m], stop[m], [x for x, y in zip(groups, m) if y]); rep[key][g] = s
                md.append(line(key, g, s))
        f = rep["flips"]
        md += ["", "Flips between consecutive depths (share of questions): wrong → right " + " / ".join(f"{x * 100:.1f}%" for x in f["wrong_to_right"]) +
               "; right → wrong " + " / ".join(f"{x * 100:.1f}%" for x in f["right_to_wrong"]) +
               f"; right at a shallower depth but wrong at the deepest {f['right_somewhere_wrong_at_deepest'] * 100:.1f}%; wrong at every depth {f['wrong_everywhere'] * 100:.1f}%.", ""]
        report["panels"][p] = rep
    (out / "breakdown.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "breakdown.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
