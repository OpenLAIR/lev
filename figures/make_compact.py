"""Compact figures that stand in for two trend tables of the main text.

    python figures/make_compact.py

Both read figures/adaptive_numbers.json (written by make_adaptive.py, stopper rule "xfit"), so they need no GPU and no rows.

fig_scale.pdf  (replaces tab:scale, "Training data and recipe")
    Accuracy of the depth-1 and the deepest readout of Lev-1.4B (lev14), Lev-2.6B (lev26) and Lev-3B (lev3) trained with
    LoRA on S31, S60 and S120, plus the full-weight run on S31 (hollow markers). Left: the pooled held-out panels
    (`held-out-pooled`, the numbers of tab:scale); right: Kev-Breadth alone (`breadth-v1-development`, the deepest readout
    of tab:scale-panels; its depth-1 readout is in the json only).
fig_tiers.pdf  (replaces tab:tiers, "Where the stopper spends computation")
    Lev-2.6B (S31) on the tiers of LevBench (`eval-v1-development` -> `by_tier`): depth 1, depth 4 and on-demand accuracy, and
    the token cost of on-demand depth under each tier.

Every plotted value is checked against the table it replaces (TABLE_* below, copied from the .tex) and printed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from make_figures import LOOP, LOOP_LIGHT, DENSE, KEV  # noqa: F401  (shared style and rcParams)

HERE = Path(__file__).resolve().parent
NUMBERS = json.loads((HERE / "adaptive_numbers.json").read_text(encoding="utf-8"))
RULE = "xfit"

MODELS = [("lev14", "Lev-1.4B", LOOP_LIGHT, "o"), ("lev26", "Lev-2.6B", LOOP, "o"), ("lev3", "Lev-3B", "#c0762f", "s")]
DATA = ["s31", "s60", "s120"]

# tab:scale (pooled held-out panels): (depth 1, deepest) per model and data set; "full" = S31, all weights
TABLE_SCALE = {
    "lev26": {"s31": ("50.5", "65.3"), "s60": ("60.6", "71.2"), "s120": ("62.9", "73.0"), "s31-full": ("53.7", "63.4")},
    "lev14": {"s31": ("49.6", "61.3"), "s60": ("60.1", "69.1"), "s120": ("61.3", "69.8"), "s31-full": ("50.7", "60.0")},
    "lev3": {"s31": ("56.2", "62.1"), "s60": ("64.9", "69.3"), "s120": ("66.0", "70.7"), "s31-full": ("57.9", "62.3")},
}
# tab:scale-panels, Kev-Breadth deepest
TABLE_BREADTH = {
    "lev26": {"s31": "67.7", "s60": "67.3", "s120": "67.3", "s31-full": "66.0"},
    "lev14": {"s31": "66.0", "s60": "66.7", "s120": "65.2", "s31-full": "63.0"},
    "lev3": {"s31": "66.1", "s60": "66.5", "s120": "66.5", "s31-full": "66.1"},
}
# tab:tiers: depth 1, depth 4, on-demand, cost
TIERS = ["id", "near", "deep", "far"]
TABLE_TIERS = {"id": ("62.6", "75.8", "75.3", "0.66"), "near": ("42.0", "71.8", "69.8", "0.79"),
               "deep": ("34.5", "45.4", "44.9", "0.87"), "far": ("41.5", "57.5", "56.8", "0.75")}


def pct(x):
    return f"{100 * x:.1f}"


def scale_numbers():
    """{panel: {model: {data: (depth 1, deepest) in percent}}}, asserted against the tables."""
    out = {"held-out-pooled": {}, "breadth-v1-development": {}}
    for m, *_ in MODELS:
        for panel in out:
            out[panel][m] = {}
            for s in DATA + ["s31-full"]:
                fixed = NUMBERS[f"{m}-{s}"]["rules"][RULE][panel]["fixed"]
                d1, deep = round(100 * fixed[0], 1), round(100 * fixed[-1], 1)
                if panel == "held-out-pooled":
                    assert (pct(fixed[0]), pct(fixed[-1])) == TABLE_SCALE[m][s], (m, s, pct(fixed[0]), pct(fixed[-1]))
                else:
                    assert pct(fixed[-1]) == TABLE_BREADTH[m][s], (m, s, pct(fixed[-1]))
                out[panel][m][s] = (d1, deep)
    return out


def tier_numbers():
    out = {}
    for t in TIERS:
        r = NUMBERS["lev26-s31"]["rules"][RULE]["eval-v1-development"]["by_tier"][t]
        vals = (pct(r["fixed"][0]), pct(r["fixed"][-1]), pct(r["adaptive"]), f"{r['cost']['tokens']:.2f}")
        assert len(r["fixed"]) == 4 and vals == TABLE_TIERS[t], (t, vals)
        out[t] = tuple(float(v) for v in vals)
    return out


def fig_scale(nums, path):
    panels = [("held-out-pooled", "Pooled held-out panels"), ("breadth-v1-development", "Kev-Breadth")]
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.05), sharey=True)
    xs = list(range(len(DATA)))
    for ax, (panel, title) in zip(axes, panels):
        for m, name, color, mk in MODELS:
            d = nums[panel][m]
            ax.plot(xs, [d[s][1] for s in DATA], "-", marker=mk, color=color, ms=3.2, lw=1.2, zorder=3)
            ax.plot(xs, [d[s][0] for s in DATA], "--", marker=mk, color=color, ms=2.6, lw=0.9, alpha=0.75, zorder=2)
            # full-weight fine-tuning on S31: depth 1 and deepest, joined, left of the LoRA point (one offset per model)
            x = -0.2 - 0.13 * (2 - [n for _, n, *_ in MODELS].index(name))
            ax.plot([x, x], d["s31-full"], color=color, lw=0.6, zorder=3)
            ax.plot([x, x], d["s31-full"], marker=mk, ms=3.2, mfc="white", mec=color, mew=0.9, ls="none", zorder=4)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        ax.set_xticks(xs, [s[1:] + "k" for s in DATA])   # the paper names the sets Train-31k / 60k / 120k
        ax.set_xlim(-0.6, len(DATA) - 0.75)
        ax.set_xlabel("Training set")
        ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylim(47, 76)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].text(1.0, 72.6, "deepest", fontsize=6.5, color=KEV, ha="center", va="bottom")
    axes[0].text(1.0, 57.2, "depth 1", fontsize=6.5, color=KEV, ha="center", va="top")
    handles = [Line2D([], [], color=c, marker=mk, ms=3.2, lw=1.2, label=n) for _, n, c, mk in MODELS]
    handles.append(Line2D([], [], color=DENSE, marker="o", ms=3.4, mfc="white", mew=0.9, ls="none", label="full weights"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.3, handlelength=1.4,
               handletextpad=0.4, columnspacing=0.9, bbox_to_anchor=(0.53, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.91))
    fig.savefig(path); plt.close(fig)


def fig_tiers(nums, path):
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    series = [("depth 1", 0, "#d6d6d6"), ("depth 4", 1, DENSE), ("on-demand", 2, LOOP)]
    w = 0.26
    for j, (label, k, color) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(len(TIERS))]
        ys = [nums[t][k] for t in TIERS]
        ax.bar(xs, ys, w * 0.92, color=color, label=label, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.0, f"{y:.1f}", ha="center", va="bottom", fontsize=5.3, color=KEV)
    ax.set_xticks(range(len(TIERS)), [f"{t}\ncost {nums[t][3]:.2f}" for t in TIERS])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color="#e9e9e9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=c, label=l) for l, _, c in series], loc="upper right", ncol=3, frameon=False,
              fontsize=6.3, handlelength=1.0, handletextpad=0.4, columnspacing=0.9, bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout(pad=0.3)
    fig.savefig(path); plt.close(fig)


def main():
    scale, tiers = scale_numbers(), tier_numbers()
    for panel, ms in scale.items():
        for m, d in ms.items():
            print(f"{panel:24s} {m:7s} " + "  ".join(f"{s}: d1 {a:.1f} deepest {b:.1f}" for s, (a, b) in d.items()))
    for t, v in tiers.items():
        print(f"tier {t:5s} depth1 {v[0]:.1f} depth4 {v[1]:.1f} on-demand {v[2]:.1f} cost {v[3]:.2f}")
    fig_scale(scale, HERE / "fig_scale.pdf")
    fig_tiers(tiers, HERE / "fig_tiers.pdf")
    print("wrote", HERE / "fig_scale.pdf", HERE / "fig_tiers.pdf")


if __name__ == "__main__":
    main()
