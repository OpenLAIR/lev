"""fig:cascade, detailed version: on-demand depth against dense models, dense cascades and the dense frontier.

    python figures/make_cascade_v2.py [--column]

--column writes fig_cascade_col.pdf, the LevBench (test) panel alone one column wide (the paper's figure).

Writes fig_cascade_v2.pdf from the same cascade analysis as make_cascade.py (which, with fig_cascade.pdf, is unchanged).
Compared with fig_cascade it adds the dense frontier of cascade.json ("dense_frontier": upper envelope of every dense
model and dense cascade, mixed between its points), the gap of each on-demand Lev to it ("vs_lev", the "best dense
option" rows of tab:cascade), and the latency range that no dense model covers. Depth labels, the Lev-1.4B -> Lev-2.6B
cascade and a second dense shade were tried and left out for readability (the cascade is in tab:cascade).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import LOOP, DENSE, KEV  # shared rcParams and colours
from make_cascade import HERE, LEVS, SRC, PANELS, DENSE_NAMES

CASCADES = ["Qwen3-1.7B-S31-s0>Qwen3.5-4B-S31-s0", "Qwen3.5-2B-S31-s0>Qwen3.5-4B-S31-s0", "kev-0.8b>kev-4b"]
# (label offset in points, alignment) per checkpoint, and where its gap to the dense frontier is written
NAME = {"Lev-1.4B-s0": ((2, 6), "center", "bottom"), "Lev-2.6B-s0": ((0, -6), "center", "top"), "Lev-3B-s0": ((-6, 0), "right", "center")}
# gap label: offset from the middle of the connector in points; Lev-3B's sits left of the lines with a leader
GAP_OFF = {"Lev-1.4B-s0": (4, 0), "Lev-2.6B-s0": (5, -7), "Lev-3B-s0": None}   # None: under the foot of the connector
BOX = dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85)
DENSE_LABEL_OFF = {"Qwen3-1.7B-S31-s0": (3, -2), "Qwen3.5-2B-S31-s0": (3, -2), "Qwen3.5-4B-S31-s0": (3, -2), "kev-0.8b": (3, -2), "kev-4b": (3, 2)}


def frontier(points, x):
    xs = [q[0] for q in points]
    ys = [q[1] for q in points]
    if x <= xs[0]:
        return ys[0]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return ys[-1]


def panel(ax, p, title, first):
    xmax = 255
    dense_x = sorted(p["fixed"][k + "@1"]["cost"] * 1000 for k in DENSE_NAMES)
    ax.axvspan(dense_x[0] + 3, dense_x[1] - 3, color=LOOP, alpha=0.05, lw=0, zorder=0)
    for model_id in CASCADES:
        curve = sorted((c["cost"] * 1000, 100 * c["accuracy"]) for c in p["cascades"][model_id]["maxp"]["curve"])
        ax.plot(*zip(*curve), "-", color=DENSE, lw=0.6, alpha=0.35, zorder=2)
    fr = [(q["cost"] * 1000, 100 * q["accuracy"]) for q in p["dense_frontier"]["points"]]
    fr_line = fr + [(xmax, fr[-1][1])]
    ax.plot(*zip(*fr_line), "--", color=KEV, lw=1.1, zorder=3)
    for model_id, label in DENSE_NAMES.items():
        v = p["fixed"][model_id + "@1"]
        x, y = v["cost"] * 1000, 100 * v["accuracy"]
        c = "#7a7a7a"
        ax.plot([x], [y], "s", color=c, ms=3.2, zorder=5)
        off = (14, -12) if first and model_id == "Qwen3.5-4B-S31-s0" else DENSE_LABEL_OFF[model_id]   # off the Lev-2.6B line
        lead = dict(arrowstyle="-", color=c, lw=0.4, shrinkA=0, shrinkB=2) if off[0] > 5 else None
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=5.5, color=c,
                    ha="left", va="top" if off[1] < 0 else "bottom", bbox=BOX, zorder=8, arrowprops=lead)
    for model_id, label, color in LEVS:
        pts = sorted((v["cost"] * 1000, 100 * v["accuracy"]) for k, v in p["fixed"].items() if k.startswith(model_id + "@"))
        ax.plot(*zip(*pts), "-o", color=color, ms=2.6, lw=1.0, zorder=4)
        od = p["lev"][model_id]
        x, y = od["cost"] * 1000, 100 * od["accuracy"]
        ax.scatter([x], [y], marker="*", s=60, color=color, edgecolor="black", linewidth=0.4, zorder=7)
        off, ha, va = NAME[model_id]
        ax.annotate(label, pts[-1], xytext=off, textcoords="offset points", fontsize=6.3, color=color, ha=ha, va=va,
                    zorder=8)
        yf = frontier(fr, x)
        delta = 100 * p["dense_frontier"]["vs_lev"][model_id]["delta"]
        assert abs((y - yf) - delta) < 0.05, (model_id, y - yf, delta)
        ax.plot([x, x], [yf, y], "-", color=color, lw=0.6, alpha=0.9, zorder=6)
        if GAP_OFF[model_id] is None:
            ax.annotate(f"{delta:+.1f}", (x, yf), xytext=(0, -3), textcoords="offset points", fontsize=5.8, color=color,
                        ha="center", va="top", fontweight="bold", bbox=BOX, zorder=9)
        else:
            ax.annotate(f"{delta:+.1f}", (x, (y + yf) / 2), xytext=GAP_OFF[model_id], textcoords="offset points", fontsize=5.8,
                        color=color, ha="left", va="center", fontweight="bold", bbox=BOX, zorder=9)
    if first:
        ax.text((dense_x[0] + dense_x[1]) / 2, 0.985, "no dense\nmodel", transform=ax.get_xaxis_transform(), fontsize=5.3,
                color=LOOP, alpha=0.8, ha="center", va="top", linespacing=0.95)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=7, loc="left", pad=2)
    ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
    ax.set_xlabel("mean latency (ms)", fontsize=7, labelpad=1)
    if first:
        ax.set_ylabel("accuracy (%)", fontsize=7, labelpad=1.5)


def main_column(d):
    """The paper's figure: LevBench (test) alone, one column wide."""
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    name, title = PANELS[0]
    panel(ax, d[name], title, True)
    ax.set_title("", loc="left")
    print(name, {k: round(100 * v["delta"], 1) for k, v in d[name]["dense_frontier"]["vs_lev"].items()})
    handles = [Line2D([], [], color="#555555", marker="o", ms=2.5, lw=1.0, label="Lev, fixed depths"),
               Line2D([], [], color="#555555", marker="*", ms=7, lw=0, label="Lev, on demand"),
               Line2D([], [], color=LOOP, lw=0, marker="$+5$", ms=7, label="gap to best dense"),
               Line2D([], [], color="#7a7a7a", marker="s", ms=3, lw=0, label="dense model"),
               Line2D([], [], color=DENSE, lw=0.6, alpha=0.6, label="dense cascade"),
               Line2D([], [], color=KEV, ls="--", lw=1.1, label="best dense option")]
    fig.legend(handles=handles, frameon=False, fontsize=6, loc="upper center", ncol=3, handlelength=1.5,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.53, 1.0))
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.86))
    out = HERE / "fig_cascade_col.pdf"
    fig.savefig(out)
    print("wrote", out)


def main():
    import sys
    d = json.loads(SRC.read_text(encoding="utf-8"))["panels"]
    if "--column" in sys.argv[1:]:
        return main_column(d)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.15), sharex=True)
    for i, (ax, (name, title)) in enumerate(zip(axes, PANELS)):
        panel(ax, d[name], title, i == 0)
        print(name, {k: round(100 * v["delta"], 1) for k, v in d[name]["dense_frontier"]["vs_lev"].items()})
    handles = [Line2D([], [], color="#555555", marker="o", ms=2.5, lw=1.0, label="Lev, fixed depths"),
               Line2D([], [], color="#555555", marker="*", ms=7, lw=0, label="Lev, on demand"),
               Line2D([], [], color="#7a7a7a", marker="s", ms=3, lw=0, label="dense model"),
               Line2D([], [], color=DENSE, lw=0.6, alpha=0.6, label="dense cascade"),
               Line2D([], [], color=KEV, ls="--", lw=1.1, label="best dense option"),
               Line2D([], [], color=LOOP, lw=0, marker="$+5$", ms=7, label="gap of on demand to it")]
    fig.legend(handles=handles, frameon=False, fontsize=6.3, loc="upper center", ncol=6, handlelength=1.5,
               columnspacing=1.3, handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=1.0, rect=(0, 0, 1, 0.9))
    out = HERE / "fig_cascade_v2.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
