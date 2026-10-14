"""Appendix figures that replace four tables, plus a relabelled copy of fig_pareto.

    python figures/make_appendix.py

fig_paired.pdf      (fig:paired, was tab:paired)  forest plot of the paired JevBench differences with 95% CIs.
fig_unlabelled.pdf  (fig:unlabelled, was tab:unlabelled + tab:transfer)  (a, b) the stopper of Lev-2.6B fitted with n
                    unlabelled requests per fold; (c) fitted on the unlabelled requests of one panel, applied to another.
                    Values come from figures/extra_numbers.json (make_extra.py).
fig_difficulty.pdf  (fig:difficulty, was tab:depth-extra)  Lev-2.6B on LevBench by difficulty k, and the share of the
                    300 unknowable questions answered with top probability >= 0.9.
fig_pareto_v2.pdf   fig_pareto.pdf with depth labels moved off the dense-control markers (same data, make_figures.load), plus
                    Lev-3B (Nanbeige4.2, both loops trained) from analysis/jevbench-nanbeige/pareto/points.json.

The numbers of the removed tables are copied below from the .tex (TABLE_*); every value read from a JSON is checked
against them, so the figures show exactly what the tables showed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import make_figures as F
from make_figures import LOOP, LOOP_LIGHT, DENSE, KEV  # shared style and rcParams

HERE = Path(__file__).resolve().parent
ORANGE = "#c0762f"  # Lev-3B, as in make_compact.MODELS
MODEL_COLOR = {"Lev-1.4B": LOOP_LIGHT, "Lev-2.6B": LOOP, "Lev-3B": ORANGE}
GRID = "#e9e9e9"

# tab:paired: (group, depth, items, delta, lo, hi), in points.
TABLE_PAIRED = [
    ("Lev-1.4B vs. its depth 1", "depth 2", "all", 7.4, 1.7, 13.0), ("Lev-1.4B vs. its depth 1", "depth 3", "all", 10.0, 4.3, 15.6),
    ("Lev-1.4B vs. its depth 1", "depth 4", "all", 10.0, 4.3, 16.0), ("Lev-1.4B vs. its depth 1", "depth 4", "hard", 12.6, 2.7, 23.4),
    ("Lev-2.6B vs. its depth 1", "depth 2", "all", 7.8, 2.2, 13.4), ("Lev-2.6B vs. its depth 1", "depth 3", "all", 12.1, 6.1, 18.2),
    ("Lev-2.6B vs. its depth 1", "depth 4", "all", 13.0, 7.4, 19.0), ("Lev-2.6B vs. its depth 1", "depth 4", "hard", 17.1, 6.3, 27.9),
    ("Lev-3B vs. its depth 1", "depth 2", "all", 3.5, -1.7, 8.7),
    ("Lev-1.4B vs. Qwen3-1.7B control", "depth 1", "all", -5.6, -11.7, 0.4), ("Lev-1.4B vs. Qwen3-1.7B control", "depth 2", "all", 1.7, -3.5, 6.9),
    ("Lev-1.4B vs. Qwen3-1.7B control", "depth 4", "all", 4.3, -0.4, 9.1), ("Lev-1.4B vs. Qwen3-1.7B control", "depth 4", "hard", 9.9, 1.8, 18.0),
    ("Lev-2.6B vs. Qwen3-1.7B control", "depth 1", "all", -0.4, -5.6, 4.8), ("Lev-2.6B vs. Qwen3-1.7B control", "depth 4", "all", 12.6, 7.4, 17.7),
    ("Lev-2.6B vs. Qwen3.5-2B control", "depth 2", "all", 2.2, -2.6, 7.4), ("Lev-2.6B vs. Qwen3.5-2B control", "depth 4", "all", 7.4, 2.2, 12.6),
    ("Lev-3B vs. Qwen3.5-2B control", "depth 2", "all", 3.9, -0.9, 9.1),
    ("Lev-3B vs. Qwen3.5-4B control", "depth 1", "all", -4.3, -9.1, 0.0), ("Lev-3B vs. Qwen3.5-4B control", "depth 2", "all", -0.9, -4.3, 2.6),
    ("Lev-2.6B vs. Kev-4B", "depth 3", "all", -0.4, -5.2, 4.3), ("Lev-2.6B vs. Kev-4B", "depth 4", "all", 0.4, -4.3, 5.2),
    ("Lev-2.6B vs. Kev-4B", "depth 4", "hard", 2.7, -5.4, 10.8),
]
# tab:unlabelled: n -> (held-out: deepest, frontier, cost, Kev-Transfer: deepest, frontier, cost), points and cost share.
TABLE_UNLABELLED = {
    "0": (-1.5, -0.4, 0.73, -4.4, -0.7, 0.63), "50": (-1.6, -0.5, 0.73, -2.8, -0.0, 0.68), "100": (-1.4, -0.5, 0.75, -2.0, +0.2, 0.70),
    "250": (-1.0, -0.1, 0.76, -1.5, +0.5, 0.71), "500": (-0.8, +0.0, 0.76, -1.1, +0.9, 0.71), "1000": (-0.7, +0.1, 0.76, -0.9, +1.0, 0.71),
    "all": (-0.7, +0.2, 0.76, -0.9, +1.0, 0.71),
}
# tab:transfer: fitted on (rows) -> applied to (columns), difference to the frontier in points.
PANELS = [("decision-v7-development", "Kev-Seen"), ("transfer-v9-development", "Kev-Transfer"), ("hard-v1-development", "Kev-Skills"),
          ("eval-v1-development", "LevBench"), ("breadth-v1-development", "Kev-Breadth")]
TABLE_TRANSFER = [(+1.8, -0.6, -1.3, -0.8, -0.2), (+0.5, +1.1, -0.2, -0.0, +0.5), (+0.6, -0.2, -0.1, -0.1, +0.1),
                  (+0.5, +0.8, +0.0, +0.2, +0.3), (+1.0, +0.9, -0.7, -0.2, +0.4)]
# tab:depth-extra: k -> (n, depth 1..4, on-demand, cost); the unknowable row has no cost.
TABLE_DEPTH = {
    1: (534, 44.4, 64.6, 73.8, 76.2, 74.3, 0.72), 2: (534, 40.8, 57.7, 64.8, 67.4, 65.5, 0.81), 3: (527, 41.7, 52.8, 55.8, 55.8, 55.4, 0.86),
    4: (245, 36.3, 42.0, 56.3, 54.7, 53.9, 0.84), 5: (203, 39.9, 45.3, 49.3, 47.8, 46.8, 0.87), 6: (203, 32.5, 38.4, 43.8, 45.3, 44.8, 0.87),
    7: (136, 25.0, 27.2, 28.7, 29.4, 28.7, 0.90), 8: (136, 24.3, 40.4, 41.2, 42.6, 42.6, 0.90), 9: (55, 25.5, 27.3, 30.9, 36.4, 36.4, 0.82),
    10: (55, 12.7, 27.3, 30.9, 34.5, 34.5, 0.83),
}
TABLE_UNKNOWABLE = (300, 0.0, 2.0, 6.0, 4.0, 6.3)


def fmt(v):
    """A signed one-decimal number with a typographic minus, as in the tables (keeps -0.0)."""
    return f"{v:+.1f}".replace("-", "−")


def row_labels(ax, xs, labels, names, x_names):
    """Multi-line tick labels with a column of row names on the left, drawn as one more tick without a mark so both align."""
    ax.set_xticks([x_names] + list(xs))
    ax.set_xticklabels([names] + labels, fontsize=6.3)
    ax.get_xticklabels()[0].set_color(KEV)
    first = ax.xaxis.get_major_ticks()[0]
    first.tick1line.set_visible(False)
    first.label1.set_horizontalalignment("right")


def fig_paired(out):
    groups = []
    for g, d, items, v, lo, hi in TABLE_PAIRED:
        if not groups or groups[-1][0] != g:
            groups.append((g, []))
        groups[-1][1].append((d, items, v, lo, hi))
    n_lines = sum(1 + len(rows) for _, rows in groups) + len(groups) - 1
    fig, ax = plt.subplots(figsize=(3.2, 0.118 * n_lines + 0.45))
    y = 0
    for gi, (g, rows) in enumerate(groups):
        if gi:
            y -= 0.6
        model = g.split(" vs.")[0]
        c = MODEL_COLOR[model]
        ax.text(0.01, y, g, transform=blended_transform_factory(fig.transFigure, ax.transData), ha="left", va="center", fontsize=6.6,
                fontweight="bold", color=KEV, bbox=dict(facecolor="white", edgecolor="none", pad=0.4), zorder=5)
        y -= 1
        for d, items, v, lo, hi in rows:
            hard = items == "hard"
            ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([v], [y], "o", ms=3.4, color=c, mfc="white" if hard else c, mew=0.9, zorder=3)
            ax.text(-0.02, y, d + (", hard" if hard else ""), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.4)
            ax.text(1.02, y, f"{fmt(v)}  [{fmt(lo)}, {fmt(hi)}]", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.0,
                    color=KEV)
            y -= 1
    ax.axvline(0, color=KEV, lw=0.6, zorder=1)
    ax.set_ylim(y + 0.4, 0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-13, 29)
    ax.set_xticks([-10, 0, 10, 20])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Paired difference in accuracy (points)")
    fig.subplots_adjust(left=0.27, right=0.66, top=0.99, bottom=0.62 / (0.118 * n_lines + 0.45))
    fig.savefig(out); plt.close(fig)


def check_unlabelled(numbers):
    """The JSON behind tab:unlabelled, rounded as in the table."""
    for n, row in TABLE_UNLABELLED.items():
        for part, (deep, front, cost) in (("held-out", row[:3]), ("transfer-v9", row[3:])):
            r = numbers["unlabelled"][part][n]
            got = (round(100 * r["minus_deepest"], 1), round(100 * r["minus_frontier_tokens"], 1), round(r["cost_tokens"], 2))
            assert got == (deep, front, cost), (part, n, got, row)


def fig_unlabelled(out):
    numbers = json.loads((HERE / "extra_numbers.json").read_text(encoding="utf-8"))
    check_unlabelled(numbers)
    T = numbers["transfer"]
    cells = [[T[a][b]["minus_frontier_tokens"] for b, _ in PANELS] for a, _ in PANELS]
    for row, table_row in zip(cells, TABLE_TRANSFER):
        assert [round(100 * c[0], 1) for c in row] == list(table_row), (row, table_row)

    fig = plt.figure(figsize=(6.9, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.42, left=0.065, right=0.985, top=0.88, bottom=0.25)
    ns = list(TABLE_UNLABELLED)
    xs = range(len(ns))
    for k, (title, off) in enumerate((("(a) Held-out panels, pooled", 0), (r"(b) Kev-Transfer", 3))):
        ax = fig.add_subplot(gs[0, k])
        deep = [TABLE_UNLABELLED[n][off] for n in ns]
        front = [TABLE_UNLABELLED[n][off + 1] for n in ns]
        cost = [TABLE_UNLABELLED[n][off + 2] for n in ns]
        ax.axhline(0, color=KEV, lw=0.6, zorder=1)
        ax.plot(xs, front, "-o", color=LOOP, ms=3.2, lw=1.2, zorder=3)
        ax.plot(xs, deep, "--s", color=DENSE, ms=3.0, lw=1.0, zorder=3)
        ax.text(len(ns) - 0.7, front[-1], "vs. frontier", color=LOOP, fontsize=6.5, va="center")
        ax.text(len(ns) - 0.7, deep[-1], "vs. deepest", color=DENSE, fontsize=6.5, va="center")
        row_labels(ax, list(xs), [f"{n}\n{c:.2f}" for n, c in zip(ns, cost)], "$n$\ncost", -1.05)
        ax.set_xlim(-1.45, len(ns) + 0.9)
        ax.set_ylim(-4.9, 1.6)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, fontsize=7.5, pad=4, loc="left")
        if k == 0:
            ax.set_ylabel("Accuracy difference (points)")
    ax = fig.add_subplot(gs[0, 2])
    cmap = LinearSegmentedColormap.from_list("div", ["#c0762f", "#ffffff", LOOP])
    vals = [[100 * c[0] for c in row] for row in cells]
    ax.imshow(vals, cmap=cmap, vmin=-2.0, vmax=2.0, aspect="auto")
    for i, row in enumerate(cells):
        for j, (v, _) in enumerate(row):
            ax.text(j, i, fmt(round(100 * v, 1)), ha="center", va="center", fontsize=6.3, color="white" if abs(100 * v) > 1.5 else "black")
    names = [name for _, name in PANELS]
    ax.set_xticks(range(5)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.3)
    ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(c) Fitted on (row), applied to (column)", fontsize=7.5, pad=4, loc="left")
    fig.savefig(out); plt.close(fig)


def fig_difficulty(out):
    ks = list(TABLE_DEPTH)
    fig = plt.figure(figsize=(6.9, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1], wspace=0.22, left=0.065, right=0.985, top=0.88, bottom=0.25)
    ax = fig.add_subplot(gs[0, 0])
    shades = ["#c6d6ec", "#8fb0dc", "#4f80bf", LOOP]
    for d, c in enumerate(shades):
        ys = [TABLE_DEPTH[k][1 + d] for k in ks]
        ax.plot(ks, ys, "-o", color=c, ms=2.8, lw=1.1, zorder=3, label=f"depth {d + 1}")
    od = [TABLE_DEPTH[k][5] for k in ks]
    ax.plot(ks, od, "--", color=KEV, lw=1.0, zorder=4, label="on-demand")
    ax.legend(loc="upper right", fontsize=6.4, frameon=False, handlelength=2.0, labelspacing=0.3)
    row_labels(ax, ks, [f"{k}\n{TABLE_DEPTH[k][0]}\n{TABLE_DEPTH[k][6]:.2f}" for k in ks], "$k$\n$n$\ncost", 0.2)
    ax.set_xlim(-0.2, 10.4); ax.set_ylim(8, 82)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) LevBench by difficulty $k$", fontsize=7.5, pad=4, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["1", "2", "3", "4", "on-\ndemand"]
    vals = TABLE_UNKNOWABLE[1:]
    ax.bar(range(5), vals, color=shades + [KEV], width=0.65, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_xlabel("Depth", fontsize=7)
    ax.set_ylim(0, 8)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_ylabel("Confident answers (%)")
    ax.set_title(f"(b) {TABLE_UNKNOWABLE[0]} unknowable questions", fontsize=7.5, pad=4, loc="left")
    fig.savefig(out); plt.close(fig)


def fig_pareto_v2(out):
    """fig_pareto with the depth labels of the Ouro-2.6B curve moved off the dense-control squares."""
    loops, dense, jev = F.load(str(HERE.parent / "results/jevbench-ouro/pareto/points.json"))
    overall = F.overall
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    small = next(iter(loops))
    # (dx, dy, ha) in points per (model is small, depth)
    offsets = {(True, 1): (0, -9, "center"), (True, 2): (-5, 0, "right"), (True, 3): (0, -9, "center"), (True, 4): (0, -9, "center"),
               (False, 1): (5, -4, "left"), (False, 2): (-5, 1, "right"), (False, 3): (0, 4, "center"), (False, 4): (0, -9, "center")}
    for (m, loop), c in zip(loops.items(), [LOOP_LIGHT, LOOP]):
        xs, ys = [p["p50_ms"] for p in loop], [overall(p["accuracy"]) for p in loop]
        ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=1.3, zorder=3)
        for p, x, y in zip(loop, xs, ys):
            dx, dy, ha = offsets[(m == small, p["depth"])]
            ax.annotate(f"{p['depth']}", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=c, ha=ha)
        if m == small:
            ax.annotate(f"Lev-{m.split('-')[-1]}", (xs[3], ys[3]), xytext=(2, 7), textcoords="offset points", fontsize=7, color=c, ha="center", va="bottom")
        else:
            ax.annotate(f"Lev-{m.split('-')[-1]}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7, color=c, ha="left", va="center")
    for p in dense:
        x, y = p["p50_ms"], overall(p["accuracy"])
        if "control" in p["label"]:
            ax.plot([x], [y], "s", color=DENSE, ms=3.5, zorder=3)
            dx, dy = {"Qwen3.5-2B control": (40, -0.030), "Qwen3.5-4B control": (22, -0.030)}.get(p["label"], (12, -0.040))
            ax.annotate(p["label"].replace(" control", ""), (x, y), xytext=(x + dx, y + dy), textcoords="data", fontsize=7, color=DENSE, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=DENSE, lw=0.5, shrinkA=0, shrinkB=2))
        else:
            ax.plot([x], [y], "s", color=KEV, ms=3.5, zorder=3)
            right = "0.8B" in p["label"]
            ax.annotate(F.short(p["label"]), (x, y), xytext=(5, 0) if right else (0, 5), textcoords="offset points", fontsize=7, color=KEV,
                        ha="left" if right else "center", va="center" if right else "bottom")
    nbg = json.loads((HERE.parent / "results/jevbench-nanbeige/pareto/points.json").read_text(encoding="utf-8"))["points"]
    lev3 = sorted((p for p in nbg if "3B (Nanbeige4.2) both loops" in p["label"]), key=lambda p: p["depth"])
    xs, ys = [p["p50_ms"] for p in lev3], [overall(p["accuracy"]) for p in lev3]
    assert [round(100 * y, 1) for y in ys] == [69.3, 72.7] and [round(x) for x in xs] == [23, 45], (xs, ys)  # tab:jevbench
    ax.plot(xs, ys, "-o", color=ORANGE, ms=3.5, lw=1.3, zorder=3)
    for p, x, y in zip(lev3, xs, ys):
        ax.annotate(f"{p['depth']}", (x, y), xytext=(-5, 0) if p["depth"] == 1 else (5, 0), textcoords="offset points", fontsize=6.5,
                    color=ORANGE, ha="right" if p["depth"] == 1 else "left", va="center")
    ax.annotate("Lev-3B", (xs[-1], ys[-1]), xytext=(0, 6), textcoords="offset points", fontsize=7, color=ORANGE, ha="center", va="bottom")
    ja = overall(jev["accuracy"])
    ax.axhline(ja, color=KEV, lw=0.7, ls=":")
    ax.text(4, ja - 0.010, "Jev (closed API, latency not comparable)", fontsize=6.5, color=KEV, va="top")
    ax.set_xlim(0, 265); ax.set_ylim(0.55, 0.89)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Median latency per request (ms, one GPU, bf16)")
    ax.set_ylabel("JevBench public accuracy (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    fig.tight_layout(pad=0.3)
    fig.savefig(out); plt.close(fig)


def main():
    fig_paired(HERE / "fig_paired.pdf")
    fig_unlabelled(HERE / "fig_unlabelled.pdf")
    fig_difficulty(HERE / "fig_difficulty.pdf")
    fig_pareto_v2(HERE / "fig_pareto_v2.pdf")
    print("wrote fig_paired.pdf fig_unlabelled.pdf fig_difficulty.pdf fig_pareto_v2.pdf")


if __name__ == "__main__":
    main()
