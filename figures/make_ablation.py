"""Forest plot that stands in for the stopping-rule table of the main text (tab:ablation, now in the appendix).

    python figures/make_ablation.py

fig_ablation.pdf  (fig:ablation, "Why not stop on confidence?")
    Accuracy difference to the fixed-depth frontier at the same token cost, with its paired 95% CI, of every stopping rule and
    stopper variant on the pooled held-out panels (`pooled` = eval-v1-development + breadth-v1-development), one panel per
    model. Source: figures/extra_numbers.json["rules"] (written by make_extra.py, key `minus_frontier_tokens`).

Every plotted value is checked against tab:ablation (TABLE below, copied from the .tex) and printed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import LOOP, LOOP_LIGHT, DENSE, KEV  # shared style and rcParams

HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / "extra_numbers.json").read_text(encoding="utf-8"))["rules"]

MODELS = [("lev14", "Lev-1.4B"), ("lev26", "Lev-2.6B"), ("lev3", "Lev-3B")]
# (json key, label, group): early-exit rules, stopper variants, the stopper
ROWS = [("top probability", "top probability", "rule"), ("margin", "margin", "rule"), ("entropy", "entropy", "rule"),
        ("patience 1", "patience, $p=1$", "rule"), ("patience 2", "patience, $p=2$", "rule"),
        ("stopper, correctness target", "correctness target", "variant"),
        ("stopper, calibration only", "calibration only", "variant"),
        ("stopper, confidence features", "confidence features only", "variant"),
        ("top probability, label-free", "top prob., label-free", "variant"),
        ("stopper, change features", "change features only", "variant"),
        ("stopper", "stopper", "stopper")]
COLOR = {"rule": DENSE, "variant": LOOP_LIGHT, "stopper": LOOP}

# tab:ablation, "vs. frontier" column: (difference, CI low, CI high) in points; None = not applicable (two steps)
TABLE = {
    "lev14": ["-0.5 -1.1 +0.1", "-0.9 -1.5 -0.1", "-0.8 -1.3 -0.2", "-0.1 -0.5 +0.3", "+0.1 -0.2 +0.4",
              "-0.8 -1.4 -0.1", "+0.2 -0.3 +0.8", "+0.4 -0.1 +1.0", "+0.4 -0.1 +0.8", "+0.1 -0.2 +0.5", "+0.5 -0.0 +1.0"],
    "lev26": ["-0.4 -1.0 +0.1", "-0.7 -1.3 -0.0", "-1.6 -2.3 -0.9", "-0.1 -0.6 +0.3", "+0.1 -0.2 +0.4",
              "-0.5 -1.1 +0.1", "-0.4 -1.1 +0.1", "+0.1 -0.4 +0.7", "+0.1 -0.4 +0.7", "+0.2 -0.1 +0.6", "+0.2 -0.4 +0.8"],
    "lev3": ["+1.0 +0.5 +1.4", "+0.9 +0.4 +1.4", "+0.1 -0.4 +0.6", None, None,
             "+0.7 +0.2 +1.2", "+0.8 +0.3 +1.3", "+1.0 +0.6 +1.3", "+0.8 +0.5 +1.2", "+0.0 +0.0 +0.0", "+1.0 +0.6 +1.3"],
}


def pts(x):
    return f"{100 * x:+.1f}".replace("+-", "-")


def numbers():
    out = {}
    for m, _ in MODELS:
        out[m] = []
        for (key, _, _), want in zip(ROWS, TABLE[m]):
            if want is None:
                assert key not in RULES[m], (m, key)
                out[m].append(None)
                continue
            d, (lo, hi) = RULES[m][key]["pooled"]["minus_frontier_tokens"]
            got = " ".join(pts(v) for v in (d, lo, hi))
            # a CI bound that rounds to zero prints as -0.0 or +0.0 in the table
            assert got.replace("-0.0", "0.0").replace("+0.0", "0.0") == want.replace("-0.0", "0.0").replace("+0.0", "0.0"), (m, key, got, want)
            out[m].append(tuple(100 * v for v in (d, lo, hi)))
    return out


def plot(nums, path):
    fig, axes = plt.subplots(1, 3, figsize=(3.3, 2.0), sharey=True)
    ys = []
    y = 0.0
    for i, (_, _, group) in enumerate(ROWS):   # small gaps between the three groups
        if i and group != ROWS[i - 1][2]:
            y += 0.5
        ys.append(y)
        y += 1
    for ax, (m, title) in zip(axes, MODELS):
        ax.axvline(0, color=KEV, lw=0.6, zorder=1)
        for yy, (_, _, group), v in zip(ys, ROWS, nums[m]):
            if v is None:
                ax.text(0, yy, "n/a", ha="center", va="center", fontsize=5.3, color="#9a9a9a", zorder=3,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
                continue
            d, lo, hi = v
            ax.plot([lo, hi], [yy, yy], color=COLOR[group], lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([d], [yy], marker="o" if group != "stopper" else "D", ms=3.0 if group != "stopper" else 3.4,
                    color=COLOR[group], zorder=3)
        ax.set_title(title, fontsize=6.8, pad=2)
        ax.set_xlim(-2.6, 1.8)
        ax.set_xticks([-2, -1, 0, 1])
        ax.tick_params(axis="x", labelsize=6, pad=1.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e9e9e9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys, [label for _, label, _ in ROWS], fontsize=6)
    axes[0].set_ylim(ys[-1] + 0.6, ys[0] - 0.6)
    fig.supxlabel("Accuracy vs. fixed-depth frontier (points)", fontsize=6.8, y=0.02)
    fig.tight_layout(pad=0.2, w_pad=0.3, rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    nums = numbers()
    for m, _ in MODELS:
        for (_, label, _), v in zip(ROWS, nums[m]):
            print(f"{m:6s} {label:26s} " + ("n/a" if v is None else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*v)))
    plot(nums, HERE / "fig_ablation.pdf")
    print("wrote", HERE / "fig_ablation.pdf")


if __name__ == "__main__":
    main()
