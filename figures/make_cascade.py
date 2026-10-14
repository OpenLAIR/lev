"""On-demand depth against dense models and cascades at measured latency, on the locked test panels.

Reads the cascade analysis of the locked-test read (analysis/cascade.py --axis latency, latency priced per record from
the H200 latency table) and draws accuracy against mean latency: each Lev checkpoint at its fixed depths and on demand,
each dense model, and each dense cascade as its threshold moves.

    python figures/make_cascade.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import LOOP, LOOP_LIGHT, DENSE, KEV  # shared rcParams and colours

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/locked/cascade-latency.json"
NBG = "#c0762f"
LEVS = [("Lev-1.4B-s0", "Lev-1.4B", LOOP_LIGHT), ("Lev-2.6B-s0", "Lev-2.6B", LOOP), ("Lev-3B-s0", "Lev-3B", NBG)]
DENSE_NAMES = {"Qwen3-1.7B-S31-s0": "Qwen3-1.7B", "Qwen3.5-2B-S31-s0": "Qwen3.5-2B", "Qwen3.5-4B-S31-s0": "Qwen3.5-4B",
               "kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B"}
CASCADES = {"Qwen3-1.7B-S31-s0>Qwen3.5-4B-S31-s0": "1.7B$\\to$4B", "kev-0.8b>kev-4b": "Kev-0.8B$\\to$4B",
            "Qwen3.5-2B-S31-s0>Qwen3.5-4B-S31-s0": "2B$\\to$4B"}
PANELS = [("eval-v1-test", "LevBench (test)"), ("breadth-v1-test", "Kev-Breadth (test)")]


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))["panels"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.95), sharex=True)
    from matplotlib.lines import Line2D
    place = {"Lev-1.4B-s0": (-1, (5, -2), "left"), "Lev-2.6B-s0": (-1, (0, -5), "center"), "Lev-3B-s0": (0, (-4, 0), "right")}
    for i, (ax, (panel, title)) in enumerate(zip(axes, PANELS)):
        p = d[panel]
        for key in CASCADES:
            curve = sorted((c["cost"] * 1000, 100 * c["accuracy"]) for c in p["cascades"][key]["maxp"]["curve"])
            ax.plot(*zip(*curve), "--", color=DENSE, lw=0.7, alpha=0.7, zorder=2)
        for key, label in DENSE_NAMES.items():
            v = p["fixed"][key + "@1"]
            x, y = v["cost"] * 1000, 100 * v["accuracy"]
            c = KEV if key.startswith("kev") else DENSE
            ax.plot([x], [y], "s", color=c, ms=3, zorder=4)
            ax.annotate(label, (x, y), xytext=(3, -2), textcoords="offset points", fontsize=5.5, color=c, ha="left", va="top")
        for key, label, color in LEVS:
            pts = sorted((v["cost"] * 1000, 100 * v["accuracy"]) for k, v in p["fixed"].items() if k.startswith(key + "@"))
            ax.plot(*zip(*pts), "-o", color=color, ms=2.5, lw=1.0, zorder=3)
            od = p["lev"][key]
            ax.scatter([od["cost"] * 1000], [100 * od["accuracy"]], marker="*", s=55, color=color, edgecolor="white", linewidth=0.4, zorder=5)
            j, off, ha = place[key]
            ax.annotate(label, pts[j], xytext=off, textcoords="offset points", fontsize=6, color=color, ha=ha, va="top")
        ax.set_title(title, fontsize=7, loc="left", pad=2)
        ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
        ax.set_xlabel("mean latency (ms)", fontsize=7, labelpad=1)
        if i == 0:
            ax.set_ylabel("accuracy (%)", fontsize=7, labelpad=1.5)
    handles = [Line2D([], [], color="#555555", marker="o", ms=2.5, lw=1.0, label="Lev, fixed depth"),
               Line2D([], [], color="#555555", marker="*", ms=7, lw=0, label="Lev, on demand"),
               Line2D([], [], color=DENSE, marker="s", ms=3, lw=0, label="dense model"),
               Line2D([], [], color=DENSE, ls="--", lw=0.7, label="dense cascade")]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper center", ncol=4, handlelength=1.6,
               columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=1.0, rect=(0, 0, 1, 0.885))
    fig.savefig(HERE / "fig_cascade.pdf")


if __name__ == "__main__":
    main()
