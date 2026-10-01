"""Figure 1: the paper-level figure. (a) the factual-updating protocol, model-agnostic; (b) the
observation -- suppression, recovery, erosion (minimal model, ten seeds); (c) the explanation --
a common shift that is withdrawn and an individual displacement that persists.

    plots/fig1.{png,pdf}

Colour carries meaning throughout: navy = old facts, crimson = new facts, teal = common shift,
olive = individual displacement. Nothing here explains the toy (no beta, no D, no rms); that
is Section 3's job.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9.5,
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})

# six-point cluster, the same shape wherever a set of facts is drawn
CLUSTER = np.array([[0, 0], [0.42, 0.18], [-0.38, 0.3], [0.1, -0.4], [-0.3, -0.25], [0.35, -0.3]])


def box(ax, xy, w, h, text, fc, ec="none", fs=8, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.0))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.25)


def arrow(ax, p, q, color="0.3", lw=1.3, ls="-", alpha=1.0, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, color=color, lw=lw, ls=ls, alpha=alpha, arrowstyle="-|>", mutation_scale=ms))


def cluster(ax, c, color, ms=4.5):
    ax.plot(CLUSTER[:, 0] + c[0], CLUSTER[:, 1] + c[1], "o", color=color, ms=ms, zorder=3)


def protocol(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    note = dict(fontsize=6.5, color="0.4", linespacing=1.2)
    stage = dict(fontsize=5.8, color="0.62", ha="left", va="center", style="italic")
    # the model, in the middle
    bx, by, bw, bh = 5.5, 4.5, 2.4, 1.45
    box(ax, (bx, by), bw, bh, "model", "0.45", fs=9.5)
    # before update: the old facts are already in the model
    ax.text(0.0, 8.5, "before update", **stage)
    cluster(ax, (bx + bw / 2, 8.5), NAVY, ms=5.2)
    ax.text(bx + bw / 2 - 0.8, 8.5, "old facts $A$", ha="right", va="center", fontsize=9.5, color=NAVY)
    arrow(ax, (bx + bw / 2, 7.9), (bx + bw / 2, by + bh), color=NAVY, lw=1.0, alpha=0.5, ms=8)
    ax.text(bx + bw / 2 + 0.25, 6.95, "already known", ha="left", va="center", **note)
    # training: new facts go in, nothing else
    ax.text(0.0, by + bh / 2, "fine-tuning", **stage)
    cluster(ax, (3.3, by + bh / 2), CRIM, ms=5.2)
    ax.text(3.3, by + bh / 2 + 0.85, "new facts $B$", ha="center", va="bottom", fontsize=9.5, color=CRIM)
    arrow(ax, (4.0, by + bh / 2), (bx, by + bh / 2), color=CRIM, lw=1.4, ms=10)
    ax.text(3.3, by - 0.15, "train on $B$ only\nno replay of $A$", ha="center", va="top", **note)
    # evaluation: two probes, read throughout
    ax.text(0.0, 2.5, "evaluation", **stage)
    pw, ph, py = 3.25, 1.3, 1.75
    box(ax, (3.1, py), pw, ph, "accuracy on $A$", "white", ec=NAVY, tc=NAVY, fs=8.2)
    box(ax, (6.75, py), pw, ph, "accuracy on $B$", "white", ec=CRIM, tc=CRIM, fs=8.2)
    arrow(ax, (bx + 0.6, by), (3.1 + pw / 2, py + ph), color="0.3", lw=1.0, ms=8)
    arrow(ax, (bx + bw - 0.6, by), (6.75 + pw / 2, py + ph), color="0.3", lw=1.0, ms=8)
    ax.text(6.6, py - 0.35, "measured throughout fine-tuning", ha="center", va="top", **note)


def observation(ax):
    d = json.load(open(os.path.join(HERE, "paper_base.json")))
    runs = [r["rows"] for r in d.values() if r["norm"] == 1]
    st = np.array([x["step"] for x in runs[0]], float)
    A = np.array([[x["A"] for x in r] for r in runs]); B = np.array([[x["B"] for x in r] for r in runs])
    for Y, col, ls in ((A, NAVY, "-"), (B, CRIM, ":")):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=2.0)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 5000); ax.set_ylim(0, 1.05)
    ax.set_xlabel("fine-tuning step"); ax.set_ylabel("accuracy")
    # curve labels next to the curves, no legend
    ax.text(700, 0.92, "new facts", color=CRIM, fontsize=9, ha="left", va="top")
    ax.text(700, 0.64, "old facts", color=NAVY, fontsize=9, ha="left", va="bottom")
    m = A.mean(0); tr = int(m.argmin())
    kw = dict(fontsize=10.5, color="0.25", arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))
    ax.annotate("suppression", (st[tr], m[tr]), xytext=(1.6, 0.2), **kw)
    ax.annotate("recovery", (280, 0.50), xytext=(13, 0.75), **kw)
    ax.annotate("erosion", (3000, m[np.argmin(abs(st - 3000))]), xytext=(700, 0.30), **kw)


def explanation(ax):
    """Three scenes, the same objects in each. Old representations: a row of navy points.
    New facts B: a crimson cluster, up and to the right, once they are being learned.
    Suppression: every old point moves by the same teal vector, toward B, plus a small olive
    one of its own. Recovery / erosion: the teal vector is gone; only the olive ones remain,
    now longer, in no common direction."""
    ax.set_xlim(0, 11.6); ax.set_ylim(-1.1, 4.0); ax.axis("off")
    rng = np.random.default_rng(3)
    xs = np.linspace(0.3, 2.3, 5); y0 = 0.8
    ind = rng.normal(size=(5, 2)); ind /= np.linalg.norm(ind, axis=1, keepdims=True)
    ind *= rng.uniform(0.7, 1.3, size=(5, 1))                   # a direction and size of its own
    common = np.array([0.55, 1.2])                               # the same for every point
    b_at = np.array([2.35, 3.05])                                # where the new facts B are written
    cols = (0.0, 4.1, 8.2)
    titles = ("before update", "suppression", "recovery / erosion")
    for col, (x0, title) in enumerate(zip(cols, titles)):
        ax.text(x0 + 1.6, 3.75, title, ha="center", va="center", fontsize=8.5, color="0.25")
        if col > 0:
            ax.plot(CLUSTER[:, 0] * 0.5 + x0 + b_at[0], CLUSTER[:, 1] * 0.5 + b_at[1], "o",
                    color=CRIM, ms=4.5, zorder=3)
            ax.text(x0 + b_at[0] + 0.38, b_at[1], "$B$", color=CRIM, fontsize=8.5, ha="left", va="center")
        for i, x in enumerate(xs):
            base = np.array([x0 + x, y0])
            if col == 0:
                ax.plot(*base, "o", color=NAVY, ms=5.5, zorder=3)
                continue
            if col == 1:
                ax.plot(*base, "o", mfc="white", mec=NAVY, ms=5.5, zorder=2)          # where it was
                arrow(ax, base, base + common, color=TEAL, lw=1.9, ms=9)               # the common shift
                end = base + common + ind[i] * 0.22                                    # + a little of its own
                arrow(ax, base + common, end, color=OLIVE, lw=1.3, ms=5)
            else:
                ax.plot(*(base + common), "o", mfc="white", mec=NAVY, ms=5.5, zorder=2)  # where it was at the trough
                arrow(ax, base + common, base, color=TEAL, lw=1.3, ls=(0, (2.5, 2)), ms=8)  # the shift withdrawn
                end = base + ind[i] * 0.5                                              # only its own, larger
                arrow(ax, base, end, color=OLIVE, lw=1.4, ms=7)
            ax.plot(*end, "o", color=NAVY, ms=5.5, zorder=3)
    cap = dict(ha="center", va="top", fontsize=7, linespacing=1.2)
    ax.text(cols[0] + 1.6, 0.1, "old representations", color="0.35", **cap)
    ax.text(cols[1] + 1.6, 0.1, "common shift", color=TEAL, **cap)
    ax.text(cols[1] + 1.6, -0.3, "+ individual\ndisplacement", color=OLIVE, **cap)
    ax.text(cols[2] + 1.6, 0.1, "common shift withdrawn", color=TEAL, **cap)
    ax.text(cols[2] + 1.6, -0.3, "individual\ndisplacement accumulates", color=OLIVE, **cap)


def main():
    fig = plt.figure(figsize=(10.6, 3.3), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.1], wspace=0.35,
                          left=0.02, right=0.99, top=0.88, bottom=0.2)
    axA = fig.add_subplot(gs[0]); protocol(axA)
    axB = fig.add_subplot(gs[1]); observation(axB)
    axC = fig.add_subplot(gs[2]); explanation(axC)
    for ax, lab, x in ((axA, "a.", 0.0), (axB, "b.", -0.2), (axC, "c.", 0.0)):
        ax.text(x, 1.04, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig1.{e}"), bbox_inches="tight")
    print("wrote plots/fig1")


if __name__ == "__main__":
    main()
