"""Offline cascade comparison (study, plan M1): on-demand depth of one looped checkpoint against a cascade of two models.

A cascade answers every request with a small model and sends a request on to a large model when a stopping rule says the
small model's answer may not be final; an escalated request pays for both models in full (nothing is reused). Two rules,
treated exactly as Lev's (analysis/adaptive_depth.py):
  maxp   the small model's top probability (its own temperature, fitted on decision-v7 calibration) >= tau; tau is the
         cheapest on the calibration rows whose cascade accuracy is within --target_drop of the large model's;
  agree  a logistic regression on the small model's output (top probability, margin, normalised entropy, log option
         count) predicts "the small model's answer equals the large model's"; label-free tau (min_agree of the fitting
         rows agree), fitted per panel on calibration plus the other fold's unlabelled rows (xfit), as Lev's stopper.
A request (record) is escalated when any of its questions is. A model inside a cascade may be a looped checkpoint read at
one depth (name@k).

Cost, per record, averaged over records:
  latency  seconds from scripts/latency_table.py's per-depth fit a + b * tokens on that system's own encoded length (the
           rows' `tokens`), measured for every system on one GPU (scripts/latency_table.py); JevBench items cost each
           system's p50 (scripts/jevbench_eval.py --latency, same job);
  flops    nominal: 2 x parameters (--params, billions) x tokens x loop steps computed.
Lev on-demand pays the latency (FLOPs) of the deepest depth any question of the record stops at (its xfit stops from
adaptive.json), i.e. the latency of a pass that computes exactly that many steps.

Comparison at matched cost: the cascade's sweep over tau gives (cost, accuracy) points; the two points around Lev's cost are
mixed to that cost exactly (per question, the mixture's expected correctness), and Lev minus this mixture gets a paired 95%
CI over record groups (2,000 draws, seed 0). The same is done at the cascade's own chosen tau against Lev's sweep. Rows are
the answerable, non-report-only questions every system read (intersection by record id and question).

    python analysis/cascade.py --lev lev26=<read dir> --system q2b=<read dir> --system q4b=<read dir> \
      --cascade q2b>q4b --panel eval-v1-development --panel breadth-v1-development \
      [--latency q2b=<latency dir> ...] [--params q2b=2 ...] --out <dir>
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import bootstrap, fit_temperature, load_jevbench, load_rows, softmax, stopper_tau   # noqa: E402


def load_system(spec_dir, panels):
    """{panel: rows} from a read dir (rows/<panel>.json; JevBench from inputs.json) plus its calibration rows and depth count."""
    d = Path(spec_dir)
    D, cal = load_rows(d / "rows" / "decision-v7-calibration.json")
    out = {}
    for p in panels:
        if p == "jevbench-public":
            jb = json.loads((d / "inputs.json").read_text(encoding="utf-8"))["jevbench_rows"] if (d / "inputs.json").exists() else None
            if jb: out[p] = load_jevbench(jb)[1]
        elif (d / "rows" / f"{p}.json").exists():
            out[p] = load_rows(d / "rows" / f"{p}.json")[1]
    return D, cal, out


def key(r):
    return (r["id"], r.get("question") or "")


def per_depth(rows, temps):
    """conf, right, answer, features [question, depth, ...] at every depth."""
    D = len(temps)
    conf, right, ans = np.zeros((len(rows), D)), np.zeros((len(rows), D), bool), np.zeros((len(rows), D), int)
    feats = np.zeros((len(rows), D, 4))
    for i, r in enumerate(rows):
        for k in range(D):
            p = softmax(r["logits"][k] / temps[k]); top = np.sort(p)[::-1]
            conf[i, k], ans[i, k] = top[0], int(p.argmax())
            right[i, k] = r["correct"][k] if "correct" in r else ans[i, k] == r["label"]
            feats[i, k] = [top[0], top[0] - top[1], -(p * np.log(p + 1e-12)).sum() / np.log(len(p)), np.log(len(p))]
    return conf, right, ans, feats


class Cost:
    """Per-record cost of one system at one depth (1-based)."""
    def __init__(self, name, latency_dir, params, depths):
        self.name, self.params, self.fit, self.jb = name, params, None, None
        if latency_dir:
            lat = json.loads((Path(latency_dir) / "latency.json").read_text(encoding="utf-8"))
            self.fit = {f["depth"]: (f["a_s"], f["b_s_per_token"]) for f in lat["fit"]}
            jr = Path(latency_dir) / "jevbench-bf16" / "report.json"
            if jr.exists():
                self.jb = {int(k): v["latency"]["p50_s"] for k, v in json.loads(jr.read_text(encoding="utf-8"))["depths"].items() if "latency" in v}

    def __call__(self, tokens, depth, axis, panel):
        if axis == "flops": return 2 * self.params * 1e9 * depth * (tokens if panel != "jevbench-public" else 1)
        if panel == "jevbench-public": return self.jb[depth]
        a, b = self.fit[depth]; return a + b * tokens


def record_mean(ids, per_row_cost):
    """Mean over records of a per-record cost (rows of one record carry the same cost: take its max)."""
    rec = {}
    for i, c in zip(ids, per_row_cost): rec[i] = max(rec.get(i, 0.0), c)
    return float(np.mean(list(rec.values())))


def escalate_by_record(ids, esc_q):
    rec = {}
    for i, e in zip(ids, esc_q): rec[i] = rec.get(i, False) or bool(e)
    return np.array([rec[i] for i in ids])


def mix_at(points, c):
    """The two sweep points around cost c and the weight of the costlier one, or None outside the sweep."""
    pts = sorted(points, key=lambda p: p["cost"])
    if c < pts[0]["cost"] - 1e-12 or c > pts[-1]["cost"] + 1e-12: return None
    for lo, hi in zip(pts, pts[1:]):
        if lo["cost"] <= c <= hi["cost"]:
            lam = 0.0 if hi["cost"] == lo["cost"] else (c - lo["cost"]) / (hi["cost"] - lo["cost"])
            return lo, hi, lam
    return pts[-1], pts[-1], 0.0


def upper_hull(points):
    """Upper concave envelope of (cost, accuracy) points: the best accuracy any mix of these operating points reaches at each cost."""
    pts = sorted(points, key=lambda q: (q["cost"], -q["accuracy"]))
    hull = []
    for q in pts:
        if hull and q["accuracy"] <= hull[-1]["accuracy"]: continue   # dearer and not better
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = (hull[-2]["cost"], hull[-2]["accuracy"]), (hull[-1]["cost"], hull[-1]["accuracy"])
            if (y2 - y1) * (q["cost"] - x1) <= (q["accuracy"] - y1) * (x2 - x1): hull.pop()   # hull[-1] lies under the chord
            else: break
        hull.append(q)
    return hull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lev", action="append", default=[], help="NAME=read dir of a looped checkpoint (rows/, analysis-xfit-cal/adaptive.json)")
    ap.add_argument("--system", action="append", default=[], help="NAME=read dir of any checkpoint that enters a cascade")
    ap.add_argument("--cascade", action="append", required=True, help="SMALL>LARGE; a looped system read at one depth is NAME@k")
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--latency", action="append", default=[], help="NAME=scripts/latency_table.py out dir")
    ap.add_argument("--params", action="append", default=[], help="NAME=parameters in billions (nominal FLOPs axis)")
    ap.add_argument("--axis", choices=["latency", "flops"], default="latency")
    ap.add_argument("--analysis", default="analysis-xfit-cal")
    ap.add_argument("--target_drop", type=float, default=0.01)
    ap.add_argument("--min_agree", type=float, default=0.98)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirs = dict(x.split("=", 1) for x in a.lev + a.system)
    lev_names = [x.split("=", 1)[0] for x in a.lev]
    lat, params = dict(x.split("=", 1) for x in a.latency), {k: float(v) for k, v in (x.split("=", 1) for x in a.params)}
    systems = {}
    for n, d in dirs.items():
        D, cal, panels = load_system(d, a.panel)
        temps = [fit_temperature(cal, k) for k in range(D)]
        systems[n] = {"D": D, "cal": cal, "panels": panels, "temps": temps, "cost": Cost(n, lat.get(n), params.get(n, 1.0), D)}
        if n in lev_names: systems[n]["stops"] = json.loads((Path(d) / a.analysis / "adaptive.json").read_text(encoding="utf-8"))["stops"]
    report, md = {"axis": a.axis, "panels": {}}, [f"# Cascade vs. on-demand depth (cost axis: {a.axis})", ""]
    for p in a.panel:
        # JevBench rows carry no token count: on the FLOPs axis an item costs 2 x parameters x steps (per token)
        unit = (lambda c: f"{c * 1000:.1f} ms") if a.axis == "latency" else (lambda c: f"{c / 1e9:.1f} GFLOP/token") if p == "jevbench-public" else (lambda c: f"{c / 1e12:.2f} TFLOP")
        if not all(p in s["panels"] for s in systems.values()): print(f"{p}: not read by every system, skipped"); continue
        common = set.intersection(*({key(r) for r in s["panels"][p]} for s in systems.values()))
        base = sorted((r for r in systems[next(iter(systems))]["panels"][p] if key(r) in common
                       and not r.get("unanswerable") and not r.get("report_only")), key=key)
        keys = [key(r) for r in base]; ids = [k[0] for k in keys]; groups = [r["group"] for r in base]
        view = {}
        for n, s in systems.items():
            idx = {key(r): i for i, r in enumerate(s["panels"][p])}
            rows = [s["panels"][p][idx[k]] for k in keys]
            conf, right, ans, feats = per_depth(rows, s["temps"])
            tok = np.array([r["tokens"] for r in rows], float)
            view[n] = {"rows": rows, "conf": conf, "right": right, "ans": ans, "feats": feats, "tok": tok,
                       "cost_at": lambda k, n=n, tok=tok: np.array([systems[n]["cost"](t, k, a.axis, p) for t in tok])}
            if "stops" in s: view[n]["stop"] = np.array([s["stops"][p][idx[k2]] for k2 in keys])
        rep = {"questions": len(keys), "fixed": {}, "lev": {}, "cascades": {}}
        dense_pool = []   # every operating point of a system that is not a Lev under test: dense models and cascades of them
        md += [f"## {p} ({len(keys)} questions every system read)", "", "| system | cost | accuracy |", "|---|---|---|"]
        for n, v in view.items():
            for k in range(1, systems[n]["D"] + 1):
                c = record_mean(ids, v["cost_at"](k)); acc = float(v["right"][:, k - 1].mean())
                rep["fixed"][f"{n}@{k}"] = {"cost": c, "accuracy": acc}
                if n not in lev_names: dense_pool.append({"name": n, "cost": c, "accuracy": acc, "correct": v["right"][:, k - 1].astype(float)})
                md.append(f"| {n}" + (f" depth {k}" if systems[n]["D"] > 1 else "") + f" | {unit(c)} | {acc:.3f} |")
        for n in lev_names:
            v = view[n]; D = systems[n]["D"]
            per_rec_depth = {}
            for i, s_ in zip(ids, v["stop"]): per_rec_depth[i] = max(per_rec_depth.get(i, 0), int(s_) + 1)
            costs_by_depth = {k: v["cost_at"](k) for k in range(1, D + 1)}
            c = record_mean(ids, [costs_by_depth[per_rec_depth[i]][j] for j, i in enumerate(ids)])
            corr = v["right"][np.arange(len(keys)), v["stop"]].astype(float)
            rep["lev"][n] = {"cost": c, "accuracy": float(corr.mean()), "correct": corr}
            md.append(f"| {n} on-demand | {unit(c)} | {corr.mean():.3f} |")
        md.append("")
        for spec in a.cascade:
            sm, lg = spec.split(">")
            (sn, sk), (ln, lk) = [(x.split("@")[0], int(x.split("@")[1]) if "@" in x else systems[x.split("@")[0]]["D"]) for x in (sm, lg)]
            S, L = view[sn], view[ln]
            s_cost, l_cost = S["cost_at"](sk), L["cost_at"](lk)
            s_right, l_right = S["right"][:, sk - 1], L["right"][:, lk - 1]
            # calibration rows of both models, aligned, for maxp's tau and the agree stopper's calibration part
            cs, cl = systems[sn], systems[ln]
            cidx = {key(r): i for i, r in enumerate(cl["cal"])}
            cal_pairs = [(r, cl["cal"][cidx[key(r)]]) for r in cs["cal"] if key(r) in cidx]
            c_s = per_depth([x for x, _ in cal_pairs], cs["temps"]); c_l = per_depth([y for _, y in cal_pairs], cl["temps"])
            cconf, cright_s, cans_s, cfeat = c_s[0][:, sk - 1], c_s[1][:, sk - 1], c_s[2][:, sk - 1], c_s[3][:, sk - 1]
            cright_l, cans_l = c_l[1][:, lk - 1], c_l[2][:, lk - 1]

            def sweep(score):   # escalate when score < tau; per-question correctness and per-record cost at each tau
                pts = []
                for t in np.unique(np.concatenate([np.quantile(score, np.linspace(0, 1, 101)), [-1.0, 2.0]])):
                    esc = escalate_by_record(ids, score < t)
                    corr = np.where(esc, l_right, s_right).astype(float)
                    pts.append({"tau": float(t), "cost": record_mean(ids, s_cost + esc * l_cost), "accuracy": float(corr.mean()), "correct": corr})
                return pts

            res = {}
            # maxp: tau on calibration (cheapest within target_drop of the large model's calibration accuracy; question-level escalation)
            ok = [t for t in np.linspace(0, 1, 201) if np.where(cconf < t, cright_l, cright_s).mean() >= cright_l.mean() - a.target_drop]
            tau = min(ok) if ok else 1.01
            sc_max = S["conf"][:, sk - 1]
            # agree: xfit on unlabelled panel rows (both models' answers, no labels)
            from sklearn.linear_model import LogisticRegression
            Fp, Gp = S["feats"][:, sk - 1], S["ans"][:, sk - 1] == L["ans"][:, lk - 1]
            Gc = cans_s == cans_l
            fold = np.array([zlib.crc32(str(g).encode()) % 2 for g in groups])
            sc_agree, taus = np.zeros(len(keys)), []
            for f in (0, 1):
                tr = fold != f
                clf = LogisticRegression(max_iter=3000).fit(np.concatenate([cfeat, Fp[tr]]), np.concatenate([Gc, Gp[tr]]))
                s_tr = clf.predict_proba(Fp[tr])[:, 1]
                t_ = stopper_tau(np.stack([s_tr], 1), np.stack([Gp[tr], np.ones(tr.sum(), bool)], 1), a.min_agree)
                # rescale so one global threshold reproduces each fold's own tau: score - tau_f + 0.5
                sc_agree[~tr] = clf.predict_proba(Fp[~tr])[:, 1] - t_ + 0.5; taus.append(t_)
            for rule, score, t in (("maxp", sc_max, tau), ("agree", sc_agree, 0.5)):
                pts = sweep(score)
                esc = escalate_by_record(ids, score < t)
                corr = np.where(esc, l_right, s_right).astype(float)
                chosen = {"tau": float(t), "cost": record_mean(ids, s_cost + esc * l_cost), "accuracy": float(corr.mean()), "escalated": float(esc.mean())}
                cmp = {}
                for n in lev_names:   # Lev at its own operating point vs. the cascade mixed to Lev's cost
                    lv = rep["lev"][n]; m = mix_at(pts, lv["cost"])
                    if m is None: cmp[n] = {"lev_cost": lv["cost"], "outside_cascade_range": True}; continue
                    lo, hi, lam = m; casc = (1 - lam) * lo["correct"] + lam * hi["correct"]
                    diff = lv["correct"] - casc
                    cmp[n] = {"lev_cost": lv["cost"], "lev_accuracy": lv["accuracy"], "cascade_accuracy_at_lev_cost": float(casc.mean()),
                              "delta": float(diff.mean()), "ci95": bootstrap(groups, diff)}
                if sn not in lev_names and ln not in lev_names: dense_pool.extend({"name": f"{spec} {rule}", **q} for q in pts)
                res[rule] = {"chosen": chosen, "vs_lev": cmp, "curve": [{k: v for k, v in q.items() if k != "correct"} for q in pts]}
            rep["cascades"][spec] = res
            md += [f"### cascade {spec}", "", "| rule | cost | accuracy | escalated | " + " | ".join(f"{n} on-demand − cascade at {n}'s cost [CI]" for n in lev_names) + " |",
                   "|---|---|---|---|" + "---|" * len(lev_names)]
            for rule, r in res.items():
                cells = []
                for n in lev_names:
                    c = r["vs_lev"][n]
                    cells.append("outside the cascade's range" if c.get("outside_cascade_range") else
                                 f"{c['delta'] * 100:+.1f} [{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}] (cascade {c['cascade_accuracy_at_lev_cost']:.3f})")
                md.append(f"| {rule} | {unit(r['chosen']['cost'])} | {r['chosen']['accuracy']:.3f} | {r['chosen']['escalated']:.2f} | " + " | ".join(cells) + " |")
            md.append("")
        hull = upper_hull(dense_pool)
        rep["dense_frontier"] = {"points": [{k: q[k] for k in ("name", "cost", "accuracy")} for q in hull], "vs_lev": {}}
        md += ["### Lev on-demand vs. the dense frontier (upper envelope of every dense model and dense cascade above, mixed to Lev's cost)", "",
               "Frontier points: " + "; ".join(f"{q['name']} {unit(q['cost'])} {q['accuracy']:.3f}" for q in hull), "",
               "| Lev | cost | accuracy | dense frontier at that cost | Δ [CI] |", "|---|---|---|---|---|"]
        for n in lev_names:
            lv = rep["lev"][n]
            m = (hull[-1], hull[-1], 0.0) if lv["cost"] >= hull[-1]["cost"] else mix_at(hull, lv["cost"])   # dearer than every dense point: its best one
            if m is None:
                rep["dense_frontier"]["vs_lev"][n] = {"cheaper_than_every_dense_point": lv["cost"] < hull[0]["cost"]}
                md.append(f"| {n} | {unit(lv['cost'])} | {lv['accuracy']:.3f} | " + ("cheaper than every dense point" if lv["cost"] < hull[0]["cost"] else "dearer than every dense point") + " | |")
                continue
            lo, hi, lam = m; fr = (1 - lam) * lo["correct"] + lam * hi["correct"]; diff = lv["correct"] - fr
            rep["dense_frontier"]["vs_lev"][n] = {"frontier_accuracy": float(fr.mean()), "delta": float(diff.mean()), "ci95": bootstrap(groups, diff)}
            ci = rep["dense_frontier"]["vs_lev"][n]["ci95"]
            md.append(f"| {n} | {unit(lv['cost'])} | {lv['accuracy']:.3f} | {fr.mean():.3f} | {diff.mean() * 100:+.1f} [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] |")
        md.append("")
        for n in rep["lev"]: rep["lev"][n].pop("correct")
        report["panels"][p] = rep
    (out / "cascade.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "cascade.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
