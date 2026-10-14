"""Pareto view of JevBench public results: quality against inference cost for the looped model at every depth, the dense
Kev checkpoints, and Jev.

Inputs are scripts/jevbench_eval.py report.json files (accuracy from the fp32 read, latency from the bf16 --latency read,
same GPU) plus Jev's committed public per-item outcomes and official Speed / Cost axes from the JevBench repository.

Quality: JevBench v1.3 chance-corrected Intelligence over the public tiers (composite_v13.intelligence), and hard-tier
accuracy. Cost: p50 latency of one request (bf16, the measuring GPU), and relative compute = parameters x depth.
The JevBench Score proxy combines public Intelligence, Calibration from the public hard-tier ECE (no gold-distribution
fidelity on public items), Speed from the measured latency under JevBench's self-hosted adjustment (x2 + 0.15 s), and
Cost from serial GPU time at --gpu_usd_per_hour; Jev's row uses its public Intelligence and ECE with its official Speed
and Cost axes. It is a proxy on public items, not an official JevBench score.

    python analysis/pareto.py --system "Kev-0.8B (dense)=dense,0.8,outputs/jevbench/kev-0.8b" --system "Kev-4B (dense)=dense,4,outputs/jevbench/kev-4b" \
        --system "Looped-Kev 1.4B=loop,1.4,outputs/jevbench/Lev-1.4B-s0" --out analysis/pareto
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"loop": "#2a78d6", "dense": "#eb6834", "jev": "#1baf7a"}     # dataviz reference palette, validated (light)
MARKERS = {"loop": "o", "dense": "s", "jev": "D"}
LOOP_COLORS = ["#2a78d6", "#8a4fd6", "#13a3b8", "#1f3f8f"]   # one per looped system, so their curves can be told apart
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dc", "#fcfcfb"


def jevbench(root):
    sys.path.insert(0, str(root))
    from jevbench import composite_v13 as comp, metrics
    return comp, metrics


def jev_reference(root, comp, metrics):
    per = json.load(open(root / "results/v1.2/jevbench-v1.2-per-task.json", encoding="utf-8"))
    tier = {t["id"]: t["tier"] for t in per["tasks"]}
    pt = per["systems"]["jev-1.13.0"]["public_tasks"]
    acc = {t: sum(o == "c" for k, (o, c) in pt.items() if tier[k] == t) / sum(tier[k] == t for k in pt) for t in ("easy", "standard", "hard")}
    ece = metrics.ece_top_label([(c, o == "c") for k, (o, c) in pt.items() if tier[k] == "hard" and c is not None])["ece"]
    official = json.load(open(root / "results/v1.4.2.2/jevbench-v1.4.2.2-results.json", encoding="utf-8"))
    is_jev = lambda r: r.get("key") == "jev-1.13.0"
    row = next(r for r in official["systems"] if is_jev(r))
    axes = row["axes"]
    return {"label": "Jev 1.13.0", "kind": "jev", "accuracy": acc, "intelligence": comp.intelligence(acc), "hard_ece": ece,
            "p50_ms": row["speed"]["p50_s_raw"] * 1000, "speed": axes["speed"], "cost": axes["cost"], "official_score": row["jevbench_score"]}


def system_points(label, kind, params_b, base, comp, gpu_usd_per_hour):
    acc = json.load(open(f"{base}-fp32/report.json", encoding="utf-8"))["depths"]
    lat = json.load(open(f"{base}-bf16/report.json", encoding="utf-8"))["depths"]
    pts = []
    for k in sorted(acc, key=int):
        a, l = acc[k], lat[k]["latency"]
        usd_per_1000 = l["mean_s"] * gpu_usd_per_hour / 3600 * 1000
        axes = {"intelligence": a["intelligence_public"], "calibration": comp.calibration(a["hard_ece"]),
                "speed": comp.speed(l["p50_s"], l["p95_s"], "gpu"), "cost": comp.cost(usd_per_1000)}
        pts.append({"label": label + (f" · depth {k}" if kind == "loop" else ""), "kind": kind, "depth": int(k), "params_b": params_b,
                    "compute": round(params_b * int(k), 2), "accuracy": a["accuracy"], "intelligence": a["intelligence_public"], "hard_ece": a["hard_ece"],
                    "p50_ms": l["p50_s"] * 1000, "usd_per_1000": usd_per_1000, "axes": axes, "score_proxy": comp.jevbench_score(axes)})
    return pts


def pareto_front(points, x, y):
    front = [p for p in points if not any(q[x] <= p[x] and q[y] >= p[y] and (q[x] < p[x] or q[y] > p[y]) for q in points)]
    return sorted(front, key=lambda p: p[x])


def plot(points, jev, x, y, xlabel, ylabel, title, path, logx=True):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for kind in ("dense", "loop"):
        ps = sorted([p for p in points if p["kind"] == kind], key=lambda p: (p["label"].split(" · ")[0], p[x]))
        groups = {}
        for p in ps: groups.setdefault(p["label"].split(" · ")[0], []).append(p)
        for gi, (name, g) in enumerate(groups.items()):
            legend = name if kind == "loop" else ("Kev, dense (released / control)" if gi == 0 else None)
            color = LOOP_COLORS[gi % len(LOOP_COLORS)] if kind == "loop" else COLORS[kind]
            ax.plot([p[x] for p in g], [p[y] for p in g], color=color, lw=2 if kind == "loop" else 0, marker=MARKERS[kind],
                    ms=8, mec=SURFACE, mew=2, label=legend, zorder=3)
            for p in g:
                text = f"d{p['depth']}" if kind == "loop" else name
                ax.annotate(text, (p[x], p[y]), textcoords="offset points", xytext=(7, -12 if kind == "loop" else 6), fontsize=8, color=INK2)
    if jev is not None and y in jev:
        if x in jev:
            ax.plot([jev[x]], [jev[y]], color=COLORS["jev"], marker=MARKERS["jev"], ms=8, mec=SURFACE, mew=2, lw=0, label=jev["label"] + " (production API)", zorder=3)
            ax.annotate("Jev 1.13.0", (jev[x], jev[y]), textcoords="offset points", xytext=(7, 6), fontsize=8, color=INK2)
        ax.axhline(jev[y], color=COLORS["jev"], lw=1, ls=(0, (4, 3)), zorder=1)
    front = pareto_front(points + ([jev] if jev is not None and x in jev and y in jev else []), x, y)
    ax.plot([p[x] for p in front], [p[y] for p in front], color=INK2, lw=1, ls=":", zorder=2, label="Pareto front")
    if logx:
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 25, 50, 100, 200, 500, 1000, 2000) if lo <= t <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, color=INK2, fontsize=9); ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, lw=0.8); ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, help='"Label=kind,params_B,jb_base" (reads <jb_base>-fp32 and <jb_base>-bf16)')
    ap.add_argument("--jevbench", default=str(ROOT / "vendor/jevbench"))
    ap.add_argument("--gpu_usd_per_hour", type=float, default=1.5, help="price of the measuring GPU for the Cost proxy (serial, one request at a time)")
    ap.add_argument("--hardware", default="one A100", help="the measuring GPU, for the latency axis label")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.jevbench); comp, metrics = jevbench(root)
    points = []
    for spec in a.system:
        label, rest = spec.split("=", 1); kind, params, base = rest.split(",", 2)
        points += system_points(label, kind, float(params), base, comp, a.gpu_usd_per_hour)
    jev = jev_reference(root, comp, metrics)
    jev["score_proxy"] = comp.jevbench_score({"intelligence": jev["intelligence"], "calibration": comp.calibration(jev["hard_ece"]),
                                              "speed": jev["speed"], "cost": jev["cost"]})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plot(points, jev, "p50_ms", "intelligence", f"p50 latency per decision, ms (log; ours: {a.hardware}, bf16)", "Intelligence (chance-corrected, public items)",
         "JevBench public: intelligence vs latency", out / "intelligence_vs_latency.png")
    for p in points: p["hard"] = p["accuracy"]["hard"]
    jev["hard"] = jev["accuracy"]["hard"]
    plot(points, jev, "compute", "hard", "Relative compute per token: parameters (B) × depth (log)", "Hard-tier accuracy (public, 111 items)",
         "JevBench public: hard tier vs compute", out / "hard_vs_compute.png")
    plot(points, jev, "p50_ms", "score_proxy", "p50 latency per decision, ms (log)", "JevBench Score proxy (public items)",
         "JevBench Score proxy vs latency", out / "score_vs_latency.png")
    lines = ["| system | compute (B×depth) | p50 ms | easy | standard | hard | Intelligence | hard ECE | Speed | Cost | Score proxy |", "|---" * 11 + "|"]
    for p in points + [jev]:
        ax_ = p.get("axes", {"speed": p.get("speed"), "cost": p.get("cost")})
        lines.append(f"| {p['label']} | {p.get('compute', '–')} | {p['p50_ms']:.0f} | {p['accuracy']['easy']:.3f} | {p['accuracy']['standard']:.3f} | {p['accuracy']['hard']:.3f} | "
                     f"{p['intelligence']:.1f} | {p['hard_ece']:.3f} | {ax_['speed']:.1f} | {ax_['cost']:.1f} | {p['score_proxy']:.1f} |")
    table = "\n".join(lines)
    (out / "table.md").write_text(table + "\n", encoding="utf-8")
    (out / "points.json").write_text(json.dumps({"points": points, "jev": jev, "gpu_usd_per_hour": a.gpu_usd_per_hour, "hardware": a.hardware}, indent=1, default=str), encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
