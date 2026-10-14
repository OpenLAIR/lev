"""Phase 1 of docs/REPRODUCING.md: offline adaptive depth on one natively looped checkpoint.

A question is read at depth 1 and goes one loop deeper (reusing the loops already computed) until a stopping rule says
stop, or the last depth. One temperature per depth is fitted on the calibration rows (scripts/depth_rows.py on
decision-v7 calibration; minimum mean NLL over the 121-point log grid on 0.25..4, as scripts/fit_depth_temperatures.py).
Two rules (--rule):
  maxp   stop at depth k when the top probability >= tau; tau is the cheapest threshold whose calibration accuracy is
         within --target_drop of the deepest depth's (the 1a rule).
  agree  (the rule chosen on 2026-10-10, docs/REPRODUCING.md "1a 复核") a logistic regression per depth k < R predicts
         "the answer at depth k equals the answer at the last depth" from that depth's output (FEATURES); stop when its
         probability >= tau. Its target needs no labels, nor does tau: the cheapest tau under which the stopped answers
         agree with the last depth on >= --min_agree of the fitting rows.
Where the agree stopper is fitted (--fit):
  calibration  on the calibration rows only, then frozen for every panel;
  xfit         per panel, two folds by record group: fit on the calibration rows plus the other fold's rows of the same
               panel (their labels are never used), apply to this fold. This is a deployment that has unlabelled
               requests from its own workload.
The panels' labels are only used to score. Cost is the mean number of loops computed divided by the deepest depth, three ways:
  question = per question; record = a record runs as deep as its deepest question (the packed pass);
  tokens = the record cost weighted by its token count.
Baselines on the same rows: every fixed depth; random escalation (each question stops at depth k with the same
probability as under the threshold rule, independent of the question: its expected accuracy); oracle (the earliest depth
that is right, else depth 1). Gates (fixed in the roadmap before any read):
  G1  adaptive accuracy - deepest accuracy >= -1 pp, with the paired 95% CI lower bound > -2 pp, at question cost <= 0.70;
  G2  adaptive - random escalation at the same stop distribution: paired 95% CI lower bound > 0.
  G3  (added 2026-10-10 after the first read showed random escalation is a weak baseline) adaptive - the fixed-depth
      frontier at the same question cost (the two neighbouring fixed depths mixed to that cost): paired 95% CI lower bound > 0.
CIs: paired bootstrap over record groups, 2,000 resamples, seed 0.
Rows marked unanswerable (the Lev suites' unknowable tier, docs/DATA.md) take part in fitting the stopper,
which never sees labels, but not in accuracy or the gates: they are reported apart as the rate of confident answers
(top probability >= CONFIDENT) at each fixed depth and under the adaptive stops. Rows marked report_only (StrategyQA,
docs/DATA.md) are scored but kept out of the gates and the breakdowns. Rows that carry a tier or a difficulty k
are also reported per tier and per k. Rows without these fields (every suite frozen before) are read as before.
Each panel also gets the calibration of the answerable rows (ECE over 10 equal-width bins of the top probability, Brier
and NLL with the depth's temperature) at every fixed depth and for the adaptive answers, and adaptive.json keeps every
row's stop depth (`stops`, 0-based, in the rows file's order) for analysis/difficulty_hypothesis.py.
Rows with a single depth (a dense control, scripts/depth_rows.py) have no stopping decision: only the fixed accuracy is
reported (overall, per tier, per source, the unknowable rows' confident-answer rate), with the same temperature fit.

    python analysis/adaptive_depth.py --rows <depth_rows dir> --calibration decision-v7-calibration \
      --panel decision-v7-development --panel transfer-v9-development ... [--jevbench <jevbench_eval out dir>] \
      [--rule agree|maxp] [--fit calibration|xfit] --out <dir>
"""
import argparse, json, math
from pathlib import Path

import zlib

import numpy as np

GRID = np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))
TARGET_COST = 0.70
CONFIDENT = 0.9   # kev.metrics' confident-answer threshold


def softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def load_rows(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["depths"], [{"group": r["group"], "id": r["id"], "label": r["label"], "tokens": r["tokens"],
                          "logits": [np.asarray(z, float) for z in r["logits"]]}
                         | {"tier": r.get("tier"), "k": (r.get("difficulty") or {}).get("k"), "unanswerable": bool(r.get("unanswerable")),
                            "report_only": bool(r.get("report_only")), "source": r.get("source"), "question": r.get("question")}
                         for r in d["rows"]]


def load_jevbench(out_dir):
    """jevbench_eval.py rows-depth{k}.json: probabilities only; log p is a logit vector up to a constant."""
    files = sorted(Path(out_dir).glob("rows-depth*.json"), key=lambda p: int(p.stem.removeprefix("rows-depth")))
    per_depth = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows = []
    for i, item in enumerate(per_depth[0]):
        keys = sorted(item["probs"])
        if not all(d[i]["id"] == item["id"] for d in per_depth): raise ValueError("JevBench rows differ across depths")
        logits = [np.log(np.clip([d[i]["probs"][k] for k in keys], 1e-12, None)) for d in per_depth]
        # correctness comes from the evaluator (a temperature never changes the argmax); the label is only kept when known
        label = next((keys.index(d[i]["predicted"]) for d in per_depth if d[i]["correct"]), None)
        rows.append({"group": item["id"], "id": item["id"], "label": label, "tokens": 1, "logits": logits,
                     "correct": [bool(d[i]["correct"]) for d in per_depth], "tier": item.get("tier")})
    return len(files), rows


def fit_temperature(rows, k):
    def nll(t):
        return float(np.mean([np.logaddexp.reduce(r["logits"][k] / t) - r["logits"][k][r["label"]] / t for r in rows]))
    return float(min(GRID, key=nll))


def prepare(rows, temps):
    """conf[i, k] = top probability at depth k, right[i, k] = argmax is the label."""
    D = len(temps)
    conf, right = np.zeros((len(rows), D)), np.zeros((len(rows), D), bool)
    for i, r in enumerate(rows):
        for k in range(D):
            p = softmax(r["logits"][k] / temps[k])
            conf[i, k] = p.max()
            right[i, k] = r["correct"][k] if "correct" in r else int(p.argmax()) == r["label"]
    return conf, right


FEATURES = ["top probability", "margin", "normalised entropy", "same answer as the previous depth",
            "KL(previous depth || this depth)", "log option count", "top probability minus the previous depth's", "depth"]


def features(rows, temps):
    """F[i, k] = FEATURES of question i at depth k; agree[i, k] = its answer at depth k is its answer at the last depth."""
    D = len(temps)
    F, agree = np.zeros((len(rows), D, len(FEATURES))), np.zeros((len(rows), D), bool)
    for i, r in enumerate(rows):
        ps = [softmax(r["logits"][k] / temps[k]) for k in range(D)]
        for k, p in enumerate(ps):
            prev = ps[k - 1] if k else p
            top = np.sort(p)[::-1]
            F[i, k] = [top[0], top[0] - top[1], -(p * np.log(p + 1e-12)).sum() / np.log(len(p)), float(p.argmax() == prev.argmax()),
                       float((prev * np.log((prev + 1e-12) / (p + 1e-12))).sum()), np.log(len(p)), top[0] - prev.max(), k]
            agree[i, k] = p.argmax() == ps[-1].argmax()
    return F, agree


def fit_stopper(F, agree):
    from sklearn.linear_model import LogisticRegression
    return [LogisticRegression(max_iter=3000).fit(F[:, k], agree[:, k]) for k in range(F.shape[1] - 1)]


def stopper_scores(clfs, F):
    """[question, depth k < R]: the stopper's probability that the answer is already final."""
    return np.stack([c.predict_proba(F[:, k])[:, 1] for k, c in enumerate(clfs)], 1)


def stopper_tau(scores, agree, min_agree):
    """Label-free: the cheapest tau whose stopped answers agree with the last depth on >= min_agree of these rows."""
    D, best = agree.shape[1], None
    for t in np.linspace(0.5, 1.0, 201):
        stop = stop_depth(scores, t, scores_only=True)
        rate, cost = agree[np.arange(len(agree)), stop].mean(), (stop + 1).mean() / D
        if rate >= min_agree and (best is None or cost < best[1]): best = (float(t), cost)
    return best[0] if best else 1.01


def stop_depth(conf, tau, scores_only=False):
    """0-based depth each question stops at under threshold tau: conf is [question, depth] (the last depth is never
    tested), or with scores_only the stopper's [question, depth k < R] scores."""
    ok = (conf if scores_only else conf[:, :-1]) >= tau
    D = ok.shape[1] + 1
    return np.where(ok.any(1), ok.argmax(1), D - 1)


def costs(rows, stop, D):
    q = float(np.mean(stop + 1)) / D
    rec, tok = {}, {}
    for r, s in zip(rows, stop):
        rec[r["id"]] = max(rec.get(r["id"], 0), s + 1); tok[r["id"]] = r["tokens"]
    record = float(np.mean(list(rec.values()))) / D
    tokens = sum(rec[i] * tok[i] for i in rec) / (D * sum(tok.values()))
    return {"question": q, "record": record, "tokens": tokens}


def bootstrap(groups, diff, n=2000, seed=0):
    """Paired 95% CI of mean(diff), resampling record groups."""
    keys = sorted(set(groups)); idx = {g: [] for g in keys}
    for i, g in enumerate(groups): idx[g].append(i)
    sums = np.array([diff[idx[g]].sum() for g in keys]); cnts = np.array([len(idx[g]) for g in keys])
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(keys), (n, len(keys)))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(rows, right, stop, tau):
    D = right.shape[1]
    adaptive = right[np.arange(len(rows)), stop].astype(float)
    pi = np.bincount(stop, minlength=D) / len(stop)
    random_esc = right.astype(float) @ pi            # expected correctness under the same stop distribution
    any_right = right.any(1)
    oracle_stop = np.where(any_right, right.argmax(1), 0)
    groups = [r["group"] for r in rows]
    deep = right[:, -1].astype(float)
    c = float(np.mean(stop + 1)) / D                 # question cost; the frontier mixes depths floor and ceil of c * D
    lo = min(max(int(np.floor(c * D)), 1), D - 1); lam = min(max(c * D - lo, 0.0), 1.0)
    frontier = (1 - lam) * right[:, lo - 1] + lam * right[:, lo]
    return {
        "questions": len(rows), "tau": tau,
        "fixed": [{"depth": k + 1, "accuracy": float(right[:, k].mean()), "cost": (k + 1) / D} for k in range(D)],
        "adaptive": {"accuracy": float(adaptive.mean()), "cost": costs(rows, stop, D),
                     "stop_fraction": [float(x) for x in pi]},
        "random_same_stops": {"accuracy": float(random_esc.mean()), "cost": float(np.dot(pi, np.arange(1, D + 1))) / D},
        "oracle": {"accuracy": float(any_right.mean()), "cost": costs(rows, oracle_stop, D)},
        "adaptive_minus_deepest": {"delta": float((adaptive - deep).mean()), "ci95": bootstrap(groups, adaptive - deep)},
        "adaptive_minus_random": {"delta": float((adaptive - random_esc).mean()), "ci95": bootstrap(groups, adaptive - random_esc)},
        "adaptive_minus_frontier": {"frontier_accuracy": float(frontier.mean()), "delta": float((adaptive - frontier).mean()),
                                    "ci95": bootstrap(groups, adaptive - frontier)},
    }


def curve(scores, right, rows, scores_only=False):
    D = right.shape[1]
    tested = scores if scores_only else scores[:, :-1]
    taus = np.unique(np.concatenate([np.quantile(tested, np.linspace(0, 1, 101)), [0.0, 1.01]]))
    pts = []
    for t in taus:
        stop = stop_depth(scores, t, scores_only)
        pts.append({"tau": float(t), "cost": costs(rows, stop, D)["question"], "accuracy": float(right[np.arange(len(rows)), stop].mean())})
    return pts


def gates(res):
    d, ci = res["adaptive_minus_deepest"]["delta"], res["adaptive_minus_deepest"]["ci95"]
    g1 = d >= -0.01 and ci[0] > -0.02 and res["adaptive"]["cost"]["question"] <= TARGET_COST
    g2 = res["adaptive_minus_random"]["ci95"][0] > 0
    g3 = res["adaptive_minus_frontier"]["ci95"][0] > 0
    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)}


def ece(conf, right, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - right[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def calibration(rows, conf, right, stop, temps):
    """ECE / Brier / NLL of these rows at every fixed depth and for the adaptive answers (Brier and NLL need a label)."""
    D = right.shape[1]
    labelled = [r.get("label") is not None and "correct" not in r for r in rows]
    def at(k_of):   # k_of(i) -> depth index
        c = np.array([conf[i, k_of(i)] for i in range(len(rows))]); ok = np.array([right[i, k_of(i)] for i in range(len(rows))], float)
        out = {"ece": ece(c, ok)}
        if all(labelled):
            ps = [softmax(r["logits"][k_of(i)] / temps[k_of(i)]) for i, r in enumerate(rows)]
            out["brier"] = float(np.mean([((p - np.eye(len(p))[r["label"]]) ** 2).sum() for p, r in zip(ps, rows)]))
            out["nll"] = float(np.mean([-np.log(max(p[r["label"]], 1e-12)) for p, r in zip(ps, rows)]))
        return out
    return {"fixed": [at(lambda i, k=k: k) for k in range(D)], "adaptive": at(lambda i: int(stop[i]))}


def score(rows, conf, right, stop, taus, temps=None):
    """evaluate + gates on the answerable rows; the unanswerable rows' confident-answer rates; the per-tier and per-k
    breakdowns of the answerable rows (each with its own fixed-depth frontier)."""
    rep_only = np.array([bool(r.get("report_only")) and not r.get("unanswerable") for r in rows])
    ans = np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])
    keep = [r for r, m in zip(rows, ans) if m]
    res = evaluate(keep, right[ans], stop[ans], taus); res["gates"] = gates(res)
    if temps is not None and ans.any(): res["calibration"] = calibration(keep, conf[ans], right[ans], stop[ans], temps)
    if rep_only.any():
        sub = evaluate([r for r, m in zip(rows, rep_only) if m], right[rep_only], stop[rep_only], None)
        res["report_only"] = {k: sub[k] for k in ("questions", "fixed", "adaptive", "adaptive_minus_deepest", "adaptive_minus_frontier")}
    unk = np.array([bool(r.get("unanswerable")) for r in rows])
    if unk.any():   # the unknowable rows only (report_only rows are answerable and reported apart)
        c, s = conf[unk], stop[unk]
        res["unanswerable"] = {"questions": int(unk.sum()), "confident_threshold": CONFIDENT,
                               "fixed_confident_rate": [float((c[:, k] >= CONFIDENT).mean()) for k in range(c.shape[1])],
                               "adaptive_confident_rate": float((c[np.arange(len(s)), s] >= CONFIDENT).mean()),
                               "adaptive_cost": float(np.mean(s + 1)) / c.shape[1]}
    for key in ("tier", "k", "source"):
        values = sorted({r.get(key) for r in keep if r.get(key) is not None}, key=str)
        if not values: continue
        res[f"by_{key}"] = {}
        for v in values:
            m = np.array([r.get(key) == v for r in keep])
            sub = evaluate([r for r, x in zip(keep, m) if x], right[ans][m], stop[ans][m], None)
            res[f"by_{key}"][str(v)] = {k2: sub[k2] for k2 in ("questions", "fixed", "adaptive", "adaptive_minus_deepest", "adaptive_minus_frontier")}
    return res


def fixed_only(panels, temps, out):
    """Single-depth rows: accuracy (answerable, non-report-only rows), per tier and per source, unknowable confidence."""
    report, lines = {"temperatures": temps, "depths": 1, "panels": {}}, ["| panel | n | accuracy | unknowable confident |", "|---|---|---|---|"]
    for name, rows in panels.items():
        conf, right = prepare(rows, temps)
        ans = np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])
        unk = np.array([bool(r.get("unanswerable")) for r in rows])
        res = {"questions": int(ans.sum()), "accuracy": float(right[ans, 0].mean()) if ans.any() else None,
               "unknowable_confident_rate": float((conf[unk, 0] >= CONFIDENT).mean()) if unk.any() else None}
        for key in ("tier", "source"):
            vals = sorted({r.get(key) for r, m in zip(rows, ans) if m and r.get(key) is not None}, key=str)
            if vals: res[f"by_{key}"] = {str(v): float(right[[m and r.get(key) == v for r, m in zip(rows, ans)], 0].mean()) for v in vals}
        report["panels"][name] = res
        unk_text = "" if res["unknowable_confident_rate"] is None else f"{res['unknowable_confident_rate']:.2f}"
        lines.append(f"| {name} | {res['questions']} | {res['accuracy']:.3f} | {unk_text} |")
    (out / "adaptive.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "table.md").write_text("\n".join([f"Temperature {temps[0]:.2f}; single depth (no stopping)", ""] + lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot(curves, results, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        c, res = curves[name], results[name]
        ax.plot([p["cost"] for p in c], [p["accuracy"] for p in c], color="#0070f3", lw=1.6, label="adaptive (all tau)")
        ax.plot([f["cost"] for f in res["fixed"]], [f["accuracy"] for f in res["fixed"]], "o--", color="#666", label="fixed depth")
        ax.plot([res["adaptive"]["cost"]["question"]], [res["adaptive"]["accuracy"]], "*", ms=12, color="#0070f3", label="chosen tau")
        ax.plot([res["random_same_stops"]["cost"]], [res["random_same_stops"]["accuracy"]], "x", ms=9, color="#e00", label="random, same stops")
        ax.set_title(name, fontsize=9); ax.set_xlabel("loops computed / deepest"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy"); axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="scripts/depth_rows.py output directory")
    ap.add_argument("--calibration", default="decision-v7-calibration")
    ap.add_argument("--panel", action="append", default=[])
    ap.add_argument("--jevbench", default="", help="scripts/jevbench_eval.py --dtype fp32 output dir (rows-depth{k}.json)")
    ap.add_argument("--rule", choices=["maxp", "agree"], default="agree")
    ap.add_argument("--fit", choices=["calibration", "xfit"], default="calibration", help="--rule agree: where the stopper is fitted")
    ap.add_argument("--target_drop", type=float, default=0.01, help="--rule maxp")
    ap.add_argument("--min_agree", type=float, default=0.98, help="--rule agree")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, cal = load_rows(Path(a.rows) / f"{a.calibration}.json")
    temps = [fit_temperature(cal, k) for k in range(D)]
    if D == 1:
        panels = {a.calibration: cal} | {p: load_rows(Path(a.rows) / f"{p}.json")[1] for p in a.panel}
        if a.jevbench: panels["jevbench-public"] = load_jevbench(a.jevbench)[1]
        return fixed_only(panels, temps, out)
    conf, right = prepare(cal, temps)
    if a.rule == "maxp":
        cal_curve = curve(conf, right, cal)
        deepest = right[:, -1].mean()
        ok = [p for p in cal_curve if p["accuracy"] >= deepest - a.target_drop]
        tau = min(ok, key=lambda p: (p["cost"], -p["tau"]))["tau"]
    else:
        Fc, Gc = features(cal, temps)
        cal_clfs = fit_stopper(Fc, Gc)
        tau = stopper_tau(stopper_scores(cal_clfs, Fc), Gc, a.min_agree)
    report = {"rows": a.rows, "calibration": a.calibration, "temperatures": temps, "rule": a.rule,
              "fit": a.fit if a.rule == "agree" else "calibration", "tau": tau, "target_drop": a.target_drop,
              "min_agree": a.min_agree, "features": FEATURES if a.rule == "agree" else None, "panels": {}, "curves": {}}
    panels = {a.calibration: cal} | {p: load_rows(Path(a.rows) / f"{p}.json")[1] for p in a.panel}
    if a.jevbench:
        Dj, jb = load_jevbench(a.jevbench)
        if Dj != D: raise ValueError(f"JevBench rows have {Dj} depths, the calibration rows {D}")
        panels["jevbench-public"] = jb
    stops, rights, confs = {}, {}, {}
    for name, rows in panels.items():
        c, rt = prepare(rows, temps)
        if a.rule == "maxp":
            stop, taus, sc, only = stop_depth(c, tau), tau, c, False
        else:
            F, G = features(rows, temps)
            sc, only = stopper_scores(cal_clfs, F), True
            if a.fit == "calibration" or name == a.calibration:
                stop, taus = stop_depth(sc, tau, True), tau
            else:   # two folds by record group; the other fold's rows enter the fit without their labels
                fold = np.array([zlib.crc32(str(r["group"]).encode()) % 2 for r in rows])
                stop, taus = np.zeros(len(rows), int), []
                for f in (0, 1):
                    tr = fold != f
                    clfs = fit_stopper(np.concatenate([Fc, F[tr]]), np.concatenate([Gc, G[tr]]))
                    t = stopper_tau(stopper_scores(clfs, F[tr]), G[tr], a.min_agree)
                    stop[~tr] = stop_depth(stopper_scores(clfs, F[~tr]), t, True); taus.append(t)
        stops[name], rights[name], confs[name] = stop, rt, c
        report["panels"][name] = score(rows, c, rt, stop, taus, temps)
        report.setdefault("stops", {})[name] = [int(x) for x in stop]
        ans = np.array([not r.get("unanswerable") and not r.get("report_only") for r in rows])
        report["curves"][name] = curve(sc[ans], rt[ans], [r for r, m in zip(rows, ans) if m], only)
    if a.panel:   # pooled over the panels, each with its own stops
        rows = [r for p in a.panel for r in panels[p]]
        rt, stop = np.concatenate([rights[p] for p in a.panel]), np.concatenate([stops[p] for p in a.panel])
        report["panels"]["dev-pooled"] = score(rows, np.concatenate([confs[p] for p in a.panel]), rt, stop, None, temps)
    (out / "adaptive.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    fit = f"stopper fitted on {a.calibration}" if report["fit"] == "calibration" else "stopper cross-fitted per panel on unlabelled rows"
    lines = [f"Temperatures {', '.join(f'{t:.2f}' for t in temps)}; rule {a.rule}; tau {tau:.3f} on {a.calibration}"
             + (f"; {fit}" if a.rule == "agree" else ""), "",
             "| panel | n | " + " | ".join(f"depth {k + 1}" for k in range(D)) + " | adaptive | cost q / rec / tok | stops | random | oracle | Δ vs deepest [CI] | Δ vs random [CI] | Δ vs fixed-depth frontier [CI] | G1 | G2 | G3 |",
             "|" + "---|" * (D + 13)]
    for name, r in report["panels"].items():
        ad, dd, dr, df = r["adaptive"], r["adaptive_minus_deepest"], r["adaptive_minus_random"], r["adaptive_minus_frontier"]
        lines.append(f"| {name} | {r['questions']} | " + " | ".join(f"{f['accuracy']:.3f}" for f in r["fixed"]) +
                     f" | {ad['accuracy']:.3f} | {ad['cost']['question']:.2f} / {ad['cost']['record']:.2f} / {ad['cost']['tokens']:.2f} | " +
                     "/".join(f"{x:.2f}" for x in ad["stop_fraction"]) +
                     f" | {r['random_same_stops']['accuracy']:.3f} | {r['oracle']['accuracy']:.3f} | "
                     f"{dd['delta'] * 100:+.1f} [{dd['ci95'][0] * 100:+.1f}, {dd['ci95'][1] * 100:+.1f}] | "
                     f"{dr['delta'] * 100:+.1f} [{dr['ci95'][0] * 100:+.1f}, {dr['ci95'][1] * 100:+.1f}] | "
                     f"{df['delta'] * 100:+.1f} [{df['ci95'][0] * 100:+.1f}, {df['ci95'][1] * 100:+.1f}] | " +
                     " | ".join("pass" if r["gates"][g] else "fail" for g in ("G1", "G2", "G3")) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(report["curves"], {k: v for k, v in report["panels"].items() if k in report["curves"]}, out / "curves.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
