"""The sweeps in the (B, A) plane: old-fact accuracy against new-fact accuracy, one curve per
knob value, time running along the curve.

    plots/toy_min_sweepAB_{beta,nb,d,dose}.{png,pdf}     and the 2x2 composite toy_min_sweepAB

Why this plane: the laws say the phases sit at fixed points of B's acquisition. Then the
learning rate should drop out entirely (the dose curves lie on top of one another), and
beta, n_B and d should change the depth of the valley without moving it along the axis.
Steps after B reaches 0.99 are drawn as the short vertical tail at B = 1 (erosion). Curves
are the mean over seeds of (B(t), A(t)) at each checkpoint, so the same step grid applies.
Normalized arm only.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAMP = LinearSegmentedColormap.from_list("mint_navy",
                                         ["#B9E3CB", "#63BB96", "#2F8C7D", "#1E5E6B", "#1B2A4E"])
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})
SPECS = {
    "beta": ("sw_beta.json", "shared fraction", lambda v: f"β = {v:g}"),
    "nb": ("sw_nb.json", "number injected", lambda v: f"$n_B$ = {int(v)}"),
    "d": ("sw_d.json", "key dimension", lambda v: f"d = {int(v)}"),
    "dose": ("sw_dose.json", "injection rate", lambda v: f"rate ×{v:g}"),
}


def load(fname):
    d = json.load(open(os.path.join(HERE, fname)))
    out = {}
    for r in d.values():
        if r["norm"] != 1:
            continue
        # B is monotone up to eval jitter; its running maximum is the clock
        out.setdefault(float(r["value"]), []).append(
            (np.maximum.accumulate(np.array([x["B"] for x in r["rows"]])), np.array([x["A"] for x in r["rows"]])))
    return out


SMOOTH = 3     # light moving average over checkpoints (the grid is 5 steps early, log later)


def smooth(y, k=SMOOTH):
    """Centred moving average over k checkpoints, the first point pinned (the crash can be
    two checkpoints long, and averaging into step 0 would move the start off 1.0)."""
    if k <= 1:
        return y
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k - 1 - k // 2, y[-1])])
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")
    sm[0] = y[0]
    return sm


def draw(ax, data, fmt, legend_loc="lower left"):
    vals = sorted(data)
    cmap = RAMP(np.linspace(0, 1, len(vals)))
    for i, v in enumerate(vals):
        B = np.array([b for b, _ in data[v]]).mean(0); A = np.array([a for _, a in data[v]]).mean(0)
        B, A = np.maximum.accumulate(smooth(B)), smooth(A)
        ax.plot(B, A, color=cmap[i], lw=1.8, label=fmt(v))
        ax.plot(B[0], A[0], "o", color=cmap[i], ms=3.5)                     # start
        k = int(np.argmin(A[:np.argmax(B >= 0.99) + 1] if (B >= 0.99).any() else A))
        ax.plot(B[k], A[k], "s", color=cmap[i], ms=3.5)                     # trough
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.03)
    ax.set_xlabel("new-fact accuracy"); ax.set_ylabel("old-fact accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.6, columnspacing=1.0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), dpi=200)
    for ax, (k, (f, label, fmt)) in zip(axes.flat, SPECS.items()):
        data = load(f)
        draw(ax, data, fmt)
        f1, a1 = plt.subplots(figsize=(4.6, 3.5), dpi=200)
        draw(a1, data, fmt)
        f1.tight_layout()
        for e in ("png", "pdf"):
            f1.savefig(os.path.join(ROOT, "plots", f"toy_min_sweepAB_{k}.{e}"), bbox_inches="tight")
        plt.close(f1)
        # where is the trough on B's axis, per value?
        print(f"{k}: trough at B =", ", ".join(
            f"{fmt(v)}: {np.array([b for b, _ in data[v]]).mean(0)[int(np.argmin(np.array([a for _, a in data[v]]).mean(0)[:60]))]:.2f}"
            for v in sorted(data)))
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_sweepAB.{e}"), bbox_inches="tight")
    print("wrote plots/toy_min_sweepAB and the four panels")


if __name__ == "__main__":
    main()
