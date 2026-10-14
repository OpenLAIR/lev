"""Paper figures from the committed Route B results (no GPU, no model loads).

    python figures/make_figures.py            # from , H200 results (training run)
    python figures/make_figures.py --points ../../results/jevbench/pareto-1.4b/points.json   # the earlier A100 run

Writes fig_pareto.pdf (accuracy vs p50 latency) and fig_depth.pdf (tier accuracy vs depth) next to this script.
Each looped model is one series; the second-epoch Ouro-1.4B run is left out of both figures (it is reported in the text).
Overall accuracy is the item-weighted mean of the three public tiers (48 easy, 72 standard, 111 hard items).
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "results/jevbench-reports"
# Dense controls not in points.json: (label, fp32 JevBench read for accuracy, bf16 read for p50 latency)
EXTRA_DENSE = [
    ("Qwen3.5-2B control", "q35-2b-fp32", "q35-2b-bf16"),
    ("Qwen3.5-4B control", "q35-4b-fp32", "q35-4b-bf16"),
]
TIER_N = {"easy": 48, "standard": 72, "hard": 111}
LOOP, DENSE, KEV = "#1f5aa6", "#8c8c8c", "#4d4d4d"
LOOP_LIGHT = "#6f9bd1"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def overall(acc):
    return sum(acc[t] * n for t, n in TIER_N.items()) / sum(TIER_N.values())


def model_name(label):
    """'Looped-Kev 2.6B (Ouro) · depth 3' -> 'Ouro-2.6B'; 'Looped-Kev 1.4B (Ouro) epoch 2 · depth 1' -> None (left out)."""
    if "epoch 2" in label:
        return None
    return "Ouro-" + label.split()[1]


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    loops = {}
    for p in d["points"]:
        if p["kind"] == "loop" and (m := model_name(p["label"])):
            loops.setdefault(m, []).append(p)
    loops = {m: sorted(ps, key=lambda p: p["depth"]) for m, ps in sorted(loops.items())}
    dense = [p for p in d["points"] if p["kind"] == "dense"]   # the Qwen3-1.7B control first: fig_depth draws it
    for label, acc_dir, lat_dir in EXTRA_DENSE:
        acc, lat = (json.load(open(EXP / r / "report.json", encoding="utf-8"))["depths"]["1"] for r in (acc_dir, lat_dir))
        dense.append({"label": label, "kind": "dense", "accuracy": acc["accuracy"], "p50_ms": 1000 * lat["latency"]["p50_s"]})
    return loops, dense, d["jev"]


def short(label):
    return {"Qwen3-1.7B control": "Dense control (1.7B)", "Qwen3.5-2B control": "Dense control (2B)",
            "Qwen3.5-4B control": "Dense control (4B)"}.get(label, label)


def fig_pareto(loops, dense, jev, out):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    colors = [LOOP_LIGHT, LOOP]
    for (m, loop), c in zip(loops.items(), colors):
        xs, ys = [p["p50_ms"] for p in loop], [overall(p["accuracy"]) for p in loop]
        ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=1.3, zorder=3)
        for p, x, y in zip(loop, xs, ys):
            small = m == next(iter(loops))
            off = (5, -5) if p["depth"] == 1 and not small else (-6, 1) if p["depth"] == 2 and small else (0, -9)
            ax.annotate(f"{p['depth']}", (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=c, ha="center")
        if len(loops) > 1 and m == next(iter(loops)):  # the smaller model: label above its curve
            ax.annotate(f"Lev ({m})", (xs[2], ys[2]), xytext=(0, 7), textcoords="offset points", fontsize=7, color=c, ha="center", va="bottom")
        else:
            ax.annotate(f"Lev ({m})", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7, color=c, ha="left", va="center")
    for p in dense:
        x, y = p["p50_ms"], overall(p["accuracy"])
        if "control" in p["label"]:
            ax.plot([x], [y], "s", color=DENSE, ms=3.5, zorder=3)
            dx, dy = {"Qwen3.5-2B control": (40, -0.030), "Qwen3.5-4B control": (22, -0.030)}.get(p["label"], (12, -0.040))
            ax.annotate(short(p["label"]), (x, y), xytext=(x + dx, y + dy), textcoords="data", fontsize=7, color=DENSE, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=DENSE, lw=0.5, shrinkA=0, shrinkB=2))
        else:
            ax.plot([x], [y], "s", color=KEV, ms=3.5, zorder=3)
            right = "0.8B" in p["label"]
            ax.annotate(short(p["label"]), (x, y), xytext=(5, 0) if right else (0, 5), textcoords="offset points", fontsize=7, color=KEV,
                        ha="left" if right else "center", va="center" if right else "bottom")
    ja = overall(jev["accuracy"])
    ax.axhline(ja, color=KEV, lw=0.7, ls=":")
    ax.text(4, ja - 0.010, "Jev (closed API, latency not comparable)", fontsize=6.5, color=KEV, va="top")
    ax.set_xlim(0, 265); ax.set_ylim(0.55, 0.89)
    ax.set_xlabel("p50 latency per request (ms, one GPU, bf16)")
    ax.set_ylabel("JevBench public accuracy (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    fig.tight_layout(pad=0.3)
    fig.savefig(out); plt.close(fig)


def fig_depth(loops, dense, out):
    control = next(p for p in dense if "control" in p["label"])
    fig, axes = plt.subplots(1, len(loops), figsize=(3.2, 2.0), sharey=True)
    for ax, (m, loop) in zip(axes, loops.items()):
        depths = [p["depth"] for p in loop]
        for tier, mk in (("standard", "o"), ("hard", "^")):
            ys = [p["accuracy"][tier] for p in loop]
            ax.plot(depths, ys, "-" + mk, color=LOOP, ms=3.5, lw=1.2)
            ax.text(depths[-1] + 0.15, ys[-1], tier, color=LOOP, fontsize=6.5, va="center")
            ax.axhline(control["accuracy"][tier], color=DENSE, lw=0.8, ls="--")
        ax.set_title(f"Lev ({m})", fontsize=7.5, pad=3)
        ax.set_xticks(depths); ax.set_xlim(0.8, 5.0); ax.set_ylim(0.25, 0.95)
        ax.set_xlabel("Depth")
    axes[0].set_ylabel("Tier accuracy (%)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    for tier in ("standard", "hard"):
        axes[0].text(0.85, control["accuracy"][tier] + 0.008, f"control, {tier}", color=DENSE, fontsize=6, va="bottom")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(HERE.parent / "results/jevbench-ouro/pareto/points.json"))
    a = ap.parse_args()
    loops, dense, jev = load(a.points)
    fig_pareto(loops, dense, jev, HERE / "fig_pareto.pdf")
    fig_depth(loops, dense, HERE / "fig_depth.pdf")
    print("wrote", HERE / "fig_pareto.pdf", HERE / "fig_depth.pdf")


if __name__ == "__main__":
    main()
