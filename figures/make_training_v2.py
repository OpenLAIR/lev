"""Training-choices figure of the main text, version 2 (fig:training, fig_training_col_v2.pdf).

    python figures/make_training_v2.py

Same numbers as make_training.py --column (checked there against tab:joint and tab:scale); only the encoding changes:
- line style means one thing in both panels: solid = deepest readout, dashed = depth 1 (only (b) has both);
- (a) is Lev-2.6B on LevBench, in its (b) colour: joint training, the model trained at depth four only read at every depth
  (analysis/lev-e16/last_step_by_depth.json), and the models trained for depth one or two alone;
- (b) shows full-weight fine-tuning on Train-31k as hollow markers in a shaded "all weights" column left of the LoRA
  columns, each joined by a thin line to the LoRA result of the same model and readout; its legend gives the line styles.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import LOOP, DENSE, KEV  # shared style and rcParams
import make_compact as C
import make_training as T

HERE = Path(__file__).resolve().parent
PANEL = "eval-v1-development"   # (a) shows LevBench only; Kev-Breadth stays in tab:joint


def last_step_numbers():
    """LevBench accuracy (%) at depths 1-4 of the model trained at depth four only; its depth four is tab:joint's single-depth 4."""
    d = json.loads((T.ROOT / "results/training/last_step_by_depth.json").read_text(encoding="utf-8"))
    acc = [100 * a for a in d["panels"][PANEL]["accuracy_by_depth"]]
    assert f"{acc[3]:.1f}" == T.TABLE_JOINT[PANEL][4][1], acc
    return acc


def panel_joint(ax, nums, last):
    joint, single = nums[PANEL]
    ax.plot([1, 2, 3, 4], joint, "-", color=LOOP, marker="o", ms=3.0, lw=1.1, zorder=3)
    ax.plot([1, 2, 3, 4], last, "-", color=DENSE, marker="^", ms=3.4, lw=1.0, zorder=2)
    ax.plot([1, 2], [single[1], single[2]], marker="s", ms=3.8, mfc="white", mec=DENSE, mew=0.9, ls="none", zorder=4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(34, 67)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Lev-2.6B, LevBench", fontsize=7.2, loc="left", pad=3)
    ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=LOOP, marker="o", ms=3.0, lw=1.1, label="joint"),
                       Line2D([], [], color=DENSE, marker="^", ms=3.4, lw=1.0, label="depth 4 only"),
                       Line2D([], [], color=DENSE, marker="s", ms=3.8, mfc="white", mew=0.9, ls="none",
                              label="depth 1 or 2 only")],
              loc="lower right", frameon=False, fontsize=6.0, handlelength=1.4, handletextpad=0.4, borderaxespad=0.1)


def panel_scale(ax, scale):
    xs = [1, 2, 3]   # LoRA on Train-31k, -60k, -120k
    xf = 0           # all weights on Train-31k, in its own shaded column
    for m, name, color, mk in C.MODELS:
        d = scale["held-out-pooled"][m]
        ax.plot(xs, [d[s][1] for s in C.DATA], "-", marker=mk, color=color, ms=3.2, lw=1.2, zorder=3)
        ax.plot(xs, [d[s][0] for s in C.DATA], "--", marker=mk, color=color, ms=2.6, lw=0.9, alpha=0.75, zorder=2)
        full = d["s31-full"]
        for ls, k in (("-", 1), ("--", 0)):   # thin line to the LoRA result of the same model and readout
            ax.plot([xf, 1], [full[k], d["s31"][k]], ls, color=color, lw=0.6, alpha=0.45, zorder=1)
        ax.plot([xf, xf], full, marker=mk, ms=3.4, mfc="white", mec=color, mew=0.9, ls="none", zorder=4)
    ax.axvspan(-0.35, 0.5, color="#f2f2f2", lw=0, zorder=0)
    ax.text(0.075, 75.3, "all\nweights", fontsize=5.8, color=KEV, ha="center", va="top", linespacing=0.95)
    ax.text(2.0, 75.3, "LoRA", fontsize=5.8, color=KEV, ha="center", va="top")
    ax.set_title("(b) Training set, held-out", fontsize=7.2, loc="left", pad=3)
    ax.set_xticks([xf] + xs, ["31k"] + [s[1:] + "k" for s in C.DATA])
    ax.set_xlim(-0.35, 3.25)
    ax.set_ylim(47, 76)
    ax.set_xlabel("Training set")
    ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=KEV, lw=1.1, label="deepest"),
                       Line2D([], [], color=KEV, lw=0.9, ls="--", label="depth 1")],
              loc="lower right", frameon=False, fontsize=6.0, handlelength=1.6, handletextpad=0.4, borderaxespad=0.2)


def main():
    joint, scale, last = T.joint_numbers(), C.scale_numbers(), last_step_numbers()
    print("depth 4 only:", " ".join(f"{v:.1f}" for v in last))
    fig = plt.figure(figsize=(3.3, 2.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel_joint(a, joint, last)
    panel_scale(b, scale)
    handles = [Line2D([], [], color=c, marker=mk, ms=3.2, lw=1.2, label=n) for _, n, c, mk in C.MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.0, handlelength=1.3,
               handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    out = HERE / "fig_training_col_v2.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
