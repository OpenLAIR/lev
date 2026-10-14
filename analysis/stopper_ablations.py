"""Stopping-rule baselines and stopper ablations on existing depth rows (plan M1-M4 companion, analyses A1-A4 and A7).

Everything reuses analysis/adaptive_depth.py (temperatures, features, stopper fit, label-free tau, frontier, paired
bootstrap) and changes one thing at a time. The default stopper here (agree target, all features, xfit, min_agree 0.98)
reproduces adaptive_depth.py --rule agree --fit xfit stop for stop; the script checks this against --reference when given.

Sections (--section, repeatable; default all):
  rules        A1  the early-exit rules of the literature on the same rows, each at the operating point its own rule picks
                   and as a sweep: top probability, margin and normalised entropy thresholds (the threshold: cheapest whose
                   calibration accuracy is within --target_drop of the deepest, as the 1a rule), patience (PABEE: stop
                   once the answer has not changed for p steps, p = 1, 2), patience with a top-probability threshold, and
                   the stopper. Per rule also the post-hoc best gap to the frontier at costs 0.5 / 0.625 / 0.75 / 0.875 (the
                   upper envelope of its sweep, an optimistic bound shown as in study).
  features     A2  the stopper with feature groups removed (confidence only / cross-step change only), with a correctness
                   target fitted on the labelled calibration rows, and a min_agree sweep.
  unlabelled   A3  xfit with n unlabelled rows of the other fold (n = 0 is the calibration-only fit), 5 draws each.
  transfer     A4  the stopper fitted on calibration plus all unlabelled rows of panel A, applied to panel B.
  temperature  A7  no temperature / one temperature for every depth / one per depth (the method): ECE per depth and the
                   stopper's outcome.
Answerable, non-report-only rows are scored (unanswerable rows enter fits without labels, as in adaptive_depth.py).

    python analysis/stopper_ablations.py --rows <depth_rows dir> --panel eval-v1-development --panel breadth-v1-development \
      [--panel <path/to/other rows.json>] [--reference <adaptive_depth --fit xfit out dir>] --out <dir>
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import (GRID, bootstrap, curve, ece, evaluate, features, fit_stopper, fit_temperature, load_rows,   # noqa: E402
                            prepare, softmax, stop_depth, stopper_scores, stopper_tau)

CONF_ONLY = [0, 1, 2, 5, 7]       # top probability, margin, entropy, log option count, depth
CHANGE_ONLY = [3, 4, 6, 7]        # same answer as before, KL(previous || this), change of the top probability, depth
COSTS = (0.5, 0.625, 0.75, 0.875)


def answerable(rows):
    return np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])


def folds(rows):
    return np.array([zlib.crc32(str(r["group"]).encode()) % 2 for r in rows])


def summary(rows, right, stop):
    """evaluate() on the answerable rows, trimmed to the numbers the tables show."""
    m = answerable(rows)
    res = evaluate([r for r, x in zip(rows, m) if x], right[m], stop[m], None)
    f = res["adaptive_minus_frontier"]; d = res["adaptive_minus_deepest"]
    return {"accuracy": res["adaptive"]["accuracy"], "cost": res["adaptive"]["cost"]["question"], "cost_tokens": res["adaptive"]["cost"]["tokens"],
            "vs_deepest": d["delta"], "vs_deepest_ci": d["ci95"], "vs_frontier": f["delta"], "vs_frontier_ci": f["ci95"],
            "stop_fraction": res["adaptive"]["stop_fraction"]}


def frontier_at(right, c):
    D = right.shape[1]
    lo = min(max(int(np.floor(c * D)), 1), D - 1); lam = min(max(c * D - lo, 0.0), 1.0)
    return float(((1 - lam) * right[:, lo - 1] + lam * right[:, lo]).mean())


def posthoc_gaps(pts, right):
    """Best accuracy of a sweep at cost <= c minus the frontier at c (upper envelope; optimistic, as study's post-hoc table)."""
    out = {}
    for c in COSTS:
        ok = [p["accuracy"] for p in pts if p["cost"] <= c + 1e-9]
        out[str(c)] = (max(ok) - frontier_at(right, c)) if ok else None
    return out


# ---- stopping rules: each returns (stop, scores, scores_only) for the rows of one panel ----

def threshold_rule(score_cal, right_cal, score, target_drop):
    """Stop at depth k when score[:, k] >= tau; tau is the cheapest on the calibration sweep within target_drop of the deepest."""
    pts = curve(score_cal, right_cal, [{"id": i, "tokens": 1} for i in range(len(score_cal))])
    ok = [p for p in pts if p["accuracy"] >= right_cal[:, -1].mean() - target_drop]
    tau = min(ok, key=lambda p: (p["cost"], -p["tau"]))["tau"]
    return stop_depth(score, tau), tau


def answers(rows, temps):
    return np.array([[int(np.argmax(r["logits"][k])) for k in range(len(temps))] for r in rows])


def patience_stop(ans, p, conf=None, tau=0.0):
    """PABEE: stop at the first depth k (0-based, k >= p) whose answer equals the answers of the p depths before it, and
    (with conf) whose top probability is >= tau; else the last depth."""
    n, D = ans.shape
    stop = np.full(n, D - 1)
    for i in range(n):
        for k in range(p, D - 1):
            if all(ans[i, k] == ans[i, k - j] for j in range(1, p + 1)) and (conf is None or conf[i, k] >= tau):
                stop[i] = k; break
    return stop


def agree_stop(Fc, Gc, F, G, rows, min_agree, fit="xfit", n_unlab=None, rng=None, cols=None):
    """The stopper of adaptive_depth.py (--rule agree), optionally on a feature subset and with n unlabelled target rows."""
    sel = (lambda X: X) if cols is None else (lambda X: X[..., cols])
    if fit == "calibration":
        clfs = fit_stopper(sel(Fc), Gc)
        tau = stopper_tau(stopper_scores(clfs, sel(Fc)), Gc, min_agree)
        sc = stopper_scores(clfs, sel(F))
        return stop_depth(sc, tau, True), sc
    fold, stop, sc = folds(rows), np.zeros(len(rows), int), np.zeros((len(rows), F.shape[1] - 1))
    for f in (0, 1):
        tr = np.where(fold != f)[0]
        if n_unlab is not None: tr = rng.choice(tr, min(n_unlab, len(tr)), replace=False) if n_unlab else tr[:0]
        clfs = fit_stopper(np.concatenate([sel(Fc), sel(F[tr])]), np.concatenate([Gc, G[tr]]))
        tau = stopper_tau(stopper_scores(clfs, sel(F[tr])), G[tr], min_agree) if len(tr) else stopper_tau(stopper_scores(clfs, sel(Fc)), Gc, min_agree)
        te = fold == f
        s = stopper_scores(clfs, sel(F[te])); sc[te] = s
        stop[te] = stop_depth(s, tau, True)
    return stop, sc


def correct_stop(Fc, right_c, F, target_drop):
    """Stopper with a correctness target ("the answer at depth k is right"), fitted on the labelled calibration rows; tau
    as the threshold rule (cheapest within target_drop of the deepest calibration accuracy)."""
    clfs = fit_stopper(Fc, right_c)
    return threshold_rule(np.concatenate([stopper_scores(clfs, Fc), np.ones((len(Fc), 1))], 1), right_c,
                          np.concatenate([stopper_scores(clfs, F), np.ones((len(F), 1))], 1), target_drop)[0]


def score_matrix(rows, temps):
    """[question, depth]: top probability, margin and 1 - normalised entropy."""
    D = len(temps)
    out = np.zeros((3, len(rows), D))
    for i, r in enumerate(rows):
        for k in range(D):
            p = softmax(r["logits"][k] / temps[k]); top = np.sort(p)[::-1]
            out[:, i, k] = [top[0], top[0] - top[1], 1 + (p * np.log(p + 1e-12)).sum() / np.log(len(p))]
    return out


def fmt(s):
    return (f"{s['accuracy']:.3f} | {s['cost']:.2f} | {s['vs_deepest'] * 100:+.1f} [{s['vs_deepest_ci'][0] * 100:+.1f}, {s['vs_deepest_ci'][1] * 100:+.1f}] | "
            f"{s['vs_frontier'] * 100:+.1f} [{s['vs_frontier_ci'][0] * 100:+.1f}, {s['vs_frontier_ci'][1] * 100:+.1f}]")


HEAD = "| accuracy | cost | Δ vs deepest [CI] | Δ vs frontier [CI] |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="scripts/depth_rows.py output directory (holds the calibration rows)")
    ap.add_argument("--calibration", default="decision-v7-calibration")
    ap.add_argument("--panel", action="append", required=True, help="a rows file name in --rows (without .json) or a path to a rows file")
    ap.add_argument("--section", action="append", choices=["rules", "features", "unlabelled", "transfer", "temperature"], default=[])
    ap.add_argument("--min_agree", type=float, default=0.98)
    ap.add_argument("--target_drop", type=float, default=0.01)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--reference", default="", help="adaptive_depth.py --rule agree --fit xfit output dir: check the default stopper reproduces its stops")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sections = a.section or ["rules", "features", "unlabelled", "transfer", "temperature"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, cal = load_rows(Path(a.rows) / f"{a.calibration}.json")
    if D < 2: raise SystemExit("single-depth rows have no stopping decision")
    temps = [fit_temperature(cal, k) for k in range(D)]
    panels = {}
    for p in a.panel:
        path = Path(p) if p.endswith(".json") else Path(a.rows) / f"{p}.json"
        Dp, rows = load_rows(path)
        if Dp != D: raise SystemExit(f"{path}: {Dp} depths, calibration {D}")
        panels[path.stem] = rows
    conf_c, right_c = prepare(cal, temps)
    Fc, Gc = features(cal, temps)
    sm_c = score_matrix(cal, temps)
    cache = {}
    for name, rows in panels.items():
        conf, right = prepare(rows, temps); F, G = features(rows, temps)
        cache[name] = (rows, conf, right, F, G, score_matrix(rows, temps), answers(rows, temps))
    report = {"rows": a.rows, "temperatures": temps, "min_agree": a.min_agree, "target_drop": a.target_drop, "panels": list(panels)}
    md = [f"# Stopping-rule analyses: {a.rows}", "", f"Temperatures {', '.join(f'{t:.2f}' for t in temps)}; answerable rows; CIs: paired bootstrap over record groups.", ""]

    default_stops = {}
    for name, (rows, conf, right, F, G, sm, ans) in cache.items():
        default_stops[name], _ = agree_stop(Fc, Gc, F, G, rows, a.min_agree)
    if a.reference:
        ref = json.loads((Path(a.reference) / "adaptive.json").read_text(encoding="utf-8"))
        for name, s in default_stops.items():
            if name in ref.get("stops", {}):
                same = np.array_equal(np.array(ref["stops"][name]), s)
                report.setdefault("reproduces_reference", {})[name] = bool(same)
                if not same: print(f"WARNING: {name}: default stopper differs from {a.reference}", flush=True)

    if "rules" in sections:
        report["rules"] = {}
        md += ["## A1 Stopping rules", ""]
        for name, (rows, conf, right, F, G, sm, ans) in cache.items():
            m = answerable(rows); r = {}
            for j, label in enumerate(["top probability", "margin", "1 - entropy"]):
                stop, tau = threshold_rule(sm_c[j], right_c, sm[j], a.target_drop)
                r[label] = summary(rows, right, stop) | {"tau": tau, "posthoc": posthoc_gaps(curve(sm[j][m], right[m], [x for x, y in zip(rows, m) if y]), right[m])}
            for p in (1, 2):
                r[f"patience {p}"] = summary(rows, right, patience_stop(ans, p)) | {"posthoc": None}
            # patience 1 with a top-probability threshold (study's "same answer as the previous depth and top probability >= tau"), tau as the threshold rule
            ans_c = answers(cal, temps)
            pts = []
            for t in np.linspace(0.3, 1.0, 71):
                s = patience_stop(ans_c, 1, conf_c, t); pts.append((t, float(np.mean(s + 1)) / D, float(right_c[np.arange(len(cal)), s].mean())))
            ok = [x for x in pts if x[2] >= right_c[:, -1].mean() - a.target_drop]
            tau = min(ok, key=lambda x: (x[1], -x[0]))[0]
            r["patience 1 + top probability"] = summary(rows, right, patience_stop(ans, 1, conf, tau)) | {"tau": float(tau), "posthoc": None}
            sc = agree_stop(Fc, Gc, F, G, rows, a.min_agree)[1]
            r["stopper"] = summary(rows, right, default_stops[name]) | {"posthoc": posthoc_gaps(curve(sc[m], right[m], [x for x, y in zip(rows, m) if y], True), right[m])}
            report["rules"][name] = r
            md += [f"### {name}", "", "| rule " + HEAD + " post-hoc best gap at cost .5 / .625 / .75 / .875 |", "|---|---|---|---|---|---|"]
            for label, s in r.items():
                ph = " / ".join("—" if v is None else f"{v * 100:+.1f}" for v in s["posthoc"].values()) if s.get("posthoc") else "—"
                md.append(f"| {label} | {fmt(s)} | {ph} |")
            md.append("")

    if "features" in sections:
        report["features"] = {}
        md += ["## A2 Stopper ablations (xfit unless stated)", ""]
        for name, (rows, conf, right, F, G, sm, ans) in cache.items():
            r = {"all features (method)": summary(rows, right, default_stops[name]),
                 "confidence features only": summary(rows, right, agree_stop(Fc, Gc, F, G, rows, a.min_agree, cols=CONF_ONLY)[0]),
                 "cross-step change only": summary(rows, right, agree_stop(Fc, Gc, F, G, rows, a.min_agree, cols=CHANGE_ONLY)[0]),
                 "agree target, calibration fit": summary(rows, right, agree_stop(Fc, Gc, F, G, rows, a.min_agree, fit="calibration")[0]),
                 "correct target, calibration fit (labels)": summary(rows, right, correct_stop(Fc, right_c, F, a.target_drop))}
            for ma in (0.90, 0.95, 0.97, 0.99, 0.995):
                r[f"min_agree {ma}"] = summary(rows, right, agree_stop(Fc, Gc, F, G, rows, ma)[0])
            report["features"][name] = r
            md += [f"### {name}", "", "| variant " + HEAD, "|---|---|---|---|---|"] + [f"| {k} | {fmt(s)} |" for k, s in r.items()] + [""]

    if "unlabelled" in sections:
        report["unlabelled"] = {}
        md += ["## A3 Unlabelled target rows in the stopper's fit (per fold; mean ± sd over draws)", ""]
        for name, (rows, conf, right, F, G, sm, ans) in cache.items():
            r = {}
            for n in (0, 50, 100, 250, 500, 1000, None):
                draws = [summary(rows, right, agree_stop(Fc, Gc, F, G, rows, a.min_agree, n_unlab=n, rng=np.random.default_rng(s))[0])
                         for s in range(1 if n in (0, None) else a.draws)]
                key = "all" if n is None else str(n)
                r[key] = {f: [float(np.mean([d[f] for d in draws])), float(np.std([d[f] for d in draws]))] for f in ("accuracy", "cost", "vs_deepest", "vs_frontier")}
            report["unlabelled"][name] = r
            md += [f"### {name}", "", "| unlabelled rows | accuracy | cost | Δ vs deepest | Δ vs frontier |", "|---|---|---|---|---|"]
            md += [f"| {k} | {v['accuracy'][0]:.3f} ± {v['accuracy'][1]:.3f} | {v['cost'][0]:.2f} ± {v['cost'][1]:.2f} | "
                   f"{v['vs_deepest'][0] * 100:+.1f} ± {v['vs_deepest'][1] * 100:.1f} | {v['vs_frontier'][0] * 100:+.1f} ± {v['vs_frontier'][1] * 100:.1f} |" for k, v in r.items()] + [""]

    if "transfer" in sections and len(cache) > 1:
        report["transfer"] = {}
        names = list(cache)
        md += ["## A4 Stopper fitted on calibration + unlabelled rows of panel A (rows), applied to panel B (columns): Δ vs frontier (pp), cost", "",
               "| fit on \\ apply to | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for fa in names:
            Fa, Ga = cache[fa][3], cache[fa][4]
            clfs = fit_stopper(np.concatenate([Fc, Fa]), np.concatenate([Gc, Ga]))
            tau = stopper_tau(stopper_scores(clfs, Fa), Ga, a.min_agree)
            cells, report["transfer"][fa] = [], {}
            for fb in names:
                rows, _, right, F, *_ = cache[fb]
                s = summary(rows, right, stop_depth(stopper_scores(clfs, F), tau, True)) if fb != fa else summary(rows, right, default_stops[fb])
                report["transfer"][fa][fb] = s
                cells.append(f"{s['vs_frontier'] * 100:+.1f}, {s['cost']:.2f}" + (" (xfit)" if fb == fa else ""))
            md.append(f"| {fa} | " + " | ".join(cells) + " |")
        md.append("")

    if "temperature" in sections:
        report["temperature"] = {}
        shared = float(min(GRID, key=lambda t: np.mean([np.logaddexp.reduce(r["logits"][k] / t) - r["logits"][k][r["label"]] / t for r in cal for k in range(D)])))
        variants = {"none": [1.0] * D, f"shared ({shared:.2f})": [shared] * D, "per depth (method)": temps}
        md += ["## A7 Temperature: ECE per depth and the stopper (xfit)", ""]
        for name, (rows, *_rest) in cache.items():
            m = answerable(rows); r = {}
            for label, tv in variants.items():
                conf, right = prepare(rows, tv)
                Fc2, Gc2 = features(cal, tv); F2, G2 = features(rows, tv)
                stop = agree_stop(Fc2, Gc2, F2, G2, rows, a.min_agree)[0]
                r[label] = summary(rows, right, stop) | {"ece": [ece(conf[m, k], right[m, k]) for k in range(D)],
                                                         "ece_adaptive": ece(conf[m][np.arange(m.sum()), stop[m]], right[m][np.arange(m.sum()), stop[m]])}
            report["temperature"][name] = r
            md += [f"### {name}", "", "| temperatures | ECE per depth | ECE on-demand " + HEAD, "|---|---|---|---|---|---|---|"]
            md += [f"| {k} | {' / '.join(f'{x:.3f}' for x in s['ece'])} | {s['ece_adaptive']:.3f} | {fmt(s)} |" for k, s in r.items()] + [""]

    (out / "ablations.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "ablations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
