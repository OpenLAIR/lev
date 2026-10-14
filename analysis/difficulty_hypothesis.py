"""study (docs/REPRODUCING.md): the training-difficulty hypothesis and the calibration of every
checkpoint, from the rows and the adaptive-depth analysis already written (no model is run).

    python analysis/difficulty_hypothesis.py --run S31-lev26=outputs/Lev-2.6B-s0 ... \
      --analysis analysis-xfit-cal --out analysis/lev-e13

Per checkpoint and panel, on the answerable rows (unknowable and report_only rows left out, as the gates do), with the
per-depth temperatures of its adaptive.json:
  depth_gain       accuracy at the deepest depth - accuracy at depth 1
  salvage_auroc    among the rows wrong at depth 1: AUROC of the depth-1 top probability for telling the salvageable ones
                   (right at the deepest depth) from the hopeless ones (still wrong); 0.5 = depth 1's confidence cannot tell
  overconfidence   among the rows the stopper stopped at depth 1 (adaptive.json `stops`): mean depth-1 top probability -
                   depth-1 accuracy; with the share of rows stopped there
  ece / brier      calibration at depth 1, at the deepest depth and of the adaptive answers (adaptive.json `calibration`)
The hypothesis (docs: docs/REPRODUCING.md, phase 3, data item 5) is supported when more / harder training data
raises salvage_auroc and lowers overconfidence on the out-of-distribution panels.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import load_rows, prepare  # noqa: E402

PANELS = ("eval-v1-development", "breadth-v1-development")


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half; None when either side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):   # average ranks over ties
        m = allv == v; ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def measure(rows_file, analysis, panel):
    _, rows = load_rows(rows_file)
    temps = analysis["temperatures"]
    conf, right = prepare(rows, temps)
    stops = np.array(analysis["stops"][panel])
    ans = np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])
    c, rt, st = conf[ans], right[ans], stops[ans]
    wrong1 = ~rt[:, 0]
    at1 = st == 0
    cal = analysis["panels"][panel].get("calibration", {})
    return {"questions": int(ans.sum()), "depths": int(rt.shape[1]),
            "acc_depth1": float(rt[:, 0].mean()), "acc_deepest": float(rt[:, -1].mean()), "depth_gain": float(rt[:, -1].mean() - rt[:, 0].mean()),
            "salvageable": int((wrong1 & rt[:, -1]).sum()), "hopeless": int((wrong1 & ~rt[:, -1]).sum()),
            "salvage_auroc": auroc(c[wrong1 & rt[:, -1], 0], c[wrong1 & ~rt[:, -1], 0]),
            "stopped_at_1": float(at1.mean()),
            "overconfidence": float(c[at1, 0].mean() - rt[at1, 0].mean()) if at1.any() else None,
            "ece": {"depth1": cal.get("fixed", [{}])[0].get("ece"), "deepest": cal.get("fixed", [{}])[-1].get("ece"), "adaptive": cal.get("adaptive", {}).get("ece")},
            "brier": {"depth1": cal.get("fixed", [{}])[0].get("brier"), "deepest": cal.get("fixed", [{}])[-1].get("brier"), "adaptive": cal.get("adaptive", {}).get("brier")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=dir with rows/ and <analysis>/adaptive.json, repeatable")
    ap.add_argument("--analysis", default="analysis-xfit-cal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    result, lines = {}, []
    for spec in a.run:
        name, d = spec.split("=", 1)
        analysis = json.loads((Path(d) / a.analysis / "adaptive.json").read_text(encoding="utf-8"))
        result[name] = {p: measure(Path(d) / "rows" / f"{p}.json", analysis, p) for p in PANELS}
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"
    for p in PANELS:
        lines += [f"### {p}", "", "| checkpoint | n | depth 1 / deepest | depth gain | salvageable / hopeless | salvage AUROC | stopped at 1 | overconfidence | ECE d1 / deepest / adaptive | Brier d1 / deepest / adaptive |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in result.items():
            m = r[p]
            oc = "—" if m["overconfidence"] is None else f"{m['overconfidence'] * 100:+.1f}"
            lines.append(f"| {name} | {m['questions']} | {f(m['acc_depth1'])} / {f(m['acc_deepest'])} | {m['depth_gain'] * 100:+.1f} | {m['salvageable']} / {m['hopeless']} | "
                         f"{f(m['salvage_auroc'], 2)} | {f(m['stopped_at_1'], 2)} | {oc} | "
                         f"{f(m['ece']['depth1'])} / {f(m['ece']['deepest'])} / {f(m['ece']['adaptive'])} | {f(m['brier']['depth1'])} / {f(m['brier']['deepest'])} / {f(m['brier']['adaptive'])} |")
        lines.append("")
    (out / "difficulty.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (out / "difficulty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
