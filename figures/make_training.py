"""Training-choices figure of the main text (fig:training): joint training next to the data-scale panels of fig_scale.

    python figures/make_training.py [--column]

--column writes fig_training_col.pdf instead: panels (a) and (b) only, one column wide (the paper's current figure;
the Kev-Breadth numbers of (c) are quoted in the text).

fig_training.pdf  (fig:training, "How do training choices shape the depth curve?")
    (a) Joint vs. single-depth training (stands in for tab:joint, now in the appendix): accuracy of the jointly trained
        Lev-2.6B at each depth (figures/adaptive_numbers.json, `lev26-s31`, `fixed`) and of the Ouro-2.6B models trained
        for depth one, two, or four alone (analysis/lev-e16/joint_vs_specialist.json, `specialist`), on LevBench
        (`eval-v1-development`) and Kev-Breadth (`breadth-v1-development`).
    (b), (c) The two panels of fig_scale (make_compact.py, unchanged numbers): depth-1 and deepest accuracy by training
        set, on the pooled held-out panels and on Kev-Breadth.

Every plotted value of (a) is checked against tab:joint (TABLE_JOINT below, copied from the .tex) and printed;
(b) and (c) reuse make_compact.scale_numbers, which checks them against tab:scale.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import LOOP, DENSE, KEV  # shared style and rcParams
import make_compact as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOINT = json.loads((HERE / "adaptive_numbers.json").read_text(encoding="utf-8"))["lev26-s31"]["rules"]["xfit"]
SINGLE = json.loads((ROOT / "results/training/joint_vs_specialist.json").read_text(encoding="utf-8"))

PANELS = [("eval-v1-development", "LevBench", "-"), ("breadth-v1-development", "Kev-Breadth", "--")]
SINGLE_DEPTHS = [1, 2, 4]
# tab:joint: (joint, single) per depth and panel
TABLE_JOINT = {
    "eval-v1-development": {1: ("46.9", "48.2"), 2: ("58.6", "58.3"), 4: ("63.6", "63.2")},
    "breadth-v1-development": {1: ("55.7", "56.9"), 2: ("65.3", "64.7"), 4: ("67.7", "67.9")},
}


def pct(x):
    return f"{100 * x:.1f}"


def joint_numbers():
    """{panel: (joint accuracy at depths 1-4, {depth: single accuracy})} in percent, asserted against tab:joint."""
    out = {}
    for panel, _, _ in PANELS:
        fixed = JOINT[panel]["fixed"]
        single = {}
        for k in SINGLE_DEPTHS:
            s = SINGLE[f"depth{k}/{panel}"]
            assert abs(s["joint_s0"] - fixed[k - 1]) < 1e-12, (panel, k)
            assert (pct(fixed[k - 1]), pct(s["specialist"])) == TABLE_JOINT[panel][k], (panel, k, pct(fixed[k - 1]), pct(s["specialist"]))
            single[k] = 100 * s["specialist"]
        out[panel] = ([100 * f for f in fixed], single)
    return out


def panel_joint(ax, nums):
    for panel, name, ls in PANELS:
        joint, single = nums[panel]
        ax.plot([1, 2, 3, 4], joint, ls, color=LOOP, marker="o", ms=3.0, lw=1.1, zorder=3)
        ax.plot(SINGLE_DEPTHS, [single[k] for k in SINGLE_DEPTHS], marker="s", ms=3.6, mfc="white", mec=DENSE, mew=0.9,
                ls="none", zorder=4)
        ax.text(4.12, joint[-1], name, fontsize=6.2, color=KEV, ha="left", va="center")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.7, 5.3)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Joint vs. single-depth training", fontsize=7.5, loc="left", pad=3)
    ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=LOOP, marker="o", ms=3.0, lw=1.1, label="joint (one checkpoint)"),
                       Line2D([], [], color=DENSE, marker="s", ms=3.6, mfc="white", mew=0.9, ls="none",
                              label="trained for that depth")],
              loc="lower right", frameon=False, fontsize=6.2, handlelength=1.4, handletextpad=0.4,
              bbox_to_anchor=(1.0, 0.0))


def panels_scale(axes, nums):
    titles = ["(b) Training set, pooled held-out", "(c) Training set, Kev-Breadth"]
    xs = list(range(len(C.DATA)))
    for ax, panel, title in zip(axes, ["held-out-pooled", "breadth-v1-development"], titles):
        for i, (m, name, color, mk) in enumerate(C.MODELS):
            d = nums[panel][m]
            ax.plot(xs, [d[s][1] for s in C.DATA], "-", marker=mk, color=color, ms=3.2, lw=1.2, zorder=3)
            ax.plot(xs, [d[s][0] for s in C.DATA], "--", marker=mk, color=color, ms=2.6, lw=0.9, alpha=0.75, zorder=2)
            x = -0.2 - 0.13 * (2 - i)   # full-weight fine-tuning on S31: depth 1 and deepest, left of the LoRA point
            ax.plot([x, x], d["s31-full"], color=color, lw=0.6, zorder=3)
            ax.plot([x, x], d["s31-full"], marker=mk, ms=3.2, mfc="white", mec=color, mew=0.9, ls="none", zorder=4)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        ax.set_xticks(xs, [s[1:] + "k" for s in C.DATA])
        ax.set_xlim(-0.6, len(C.DATA) - 0.75)
        ax.set_xlabel("Training set")
        ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylim(47, 76)
    axes[1].set_ylim(47, 76)
    axes[1].tick_params(labelleft=False)
    axes[0].text(1.0, 72.6, "deepest", fontsize=6.5, color=KEV, ha="center", va="bottom")
    axes[0].text(1.0, 57.2, "depth 1", fontsize=6.5, color=KEV, ha="center", va="top")
    handles = [Line2D([], [], color=c, marker=mk, ms=3.2, lw=1.2, label=n) for _, n, c, mk in C.MODELS]
    handles.append(Line2D([], [], color=DENSE, marker="o", ms=3.4, mfc="white", mew=0.9, ls="none", label="full weights"))
    axes[1].legend(handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=6.0, handlelength=1.4,
                   handletextpad=0.4, columnspacing=0.8)


def plot(joint, scale, path):
    fig = plt.figure(figsize=(6.9, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    c = fig.add_subplot(gs[2], sharey=b)
    panel_joint(a, joint)
    panels_scale([b, c], scale)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(path)
    plt.close(fig)


def plot_column(joint, scale, path):
    fig = plt.figure(figsize=(3.3, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    panel_joint(a, joint)
    a.set_xlim(0.7, 5.9)
    a.set_title("(a) Joint vs. single depth", fontsize=7.5, loc="left", pad=3)
    a.legend(handles=[Line2D([], [], color=LOOP, marker="o", ms=3.0, lw=1.1, label="joint"),
                      Line2D([], [], color=DENSE, marker="s", ms=3.6, mfc="white", mew=0.9, ls="none",
                             label="one depth")],
             loc="lower right", frameon=False, fontsize=6.0, handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    xs = list(range(len(C.DATA)))
    for i, (m, name, color, mk) in enumerate(C.MODELS):
        d = scale["held-out-pooled"][m]
        b.plot(xs, [d[s][1] for s in C.DATA], "-", marker=mk, color=color, ms=3.2, lw=1.2, zorder=3)
        b.plot(xs, [d[s][0] for s in C.DATA], "--", marker=mk, color=color, ms=2.6, lw=0.9, alpha=0.75, zorder=2)
        x = -0.2 - 0.13 * (2 - i)
        b.plot([x, x], d["s31-full"], color=color, lw=0.6, zorder=3)
        b.plot([x, x], d["s31-full"], marker=mk, ms=3.2, mfc="white", mec=color, mew=0.9, ls="none", zorder=4)
    b.set_title("(b) Training set, held-out", fontsize=7.5, loc="left", pad=3)
    b.set_xticks(xs, [s[1:] + "k" for s in C.DATA])
    b.set_xlim(-0.6, len(C.DATA) - 0.75)
    b.set_ylim(47, 76)
    b.set_xlabel("Training set")
    b.grid(color="#e9e9e9", lw=0.5, zorder=0)
    b.set_axisbelow(True)
    b.text(1.0, 72.6, "deepest", fontsize=6.2, color=KEV, ha="center", va="bottom")
    b.text(1.0, 57.2, "depth 1", fontsize=6.2, color=KEV, ha="center", va="top")
    handles = [Line2D([], [], color=c, marker=mk, ms=3.2, lw=1.2, label=n) for _, n, c, mk in C.MODELS]
    handles.append(Line2D([], [], color=DENSE, marker="o", ms=3.4, mfc="white", mew=0.9, ls="none", label="full weights"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.0, handlelength=1.3,
               handletextpad=0.3, columnspacing=0.7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def main():
    import sys
    joint, scale = joint_numbers(), C.scale_numbers()
    for panel, (j, s) in joint.items():
        print(f"{panel:24s} joint " + " ".join(f"{v:.1f}" for v in j) + "  single " + " ".join(f"d{k} {v:.1f}" for k, v in s.items()))
    if "--column" in sys.argv[1:]:
        plot_column(joint, scale, HERE / "fig_training_col.pdf")
        print("wrote", HERE / "fig_training_col.pdf")
        return
    plot(joint, scale, HERE / "fig_training.pdf")
    print("wrote", HERE / "fig_training.pdf")


if __name__ == "__main__":
    main()
