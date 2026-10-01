"""Figure 3 (b, c): the mean-state length s and the two margins that drive it, on the minimal model.

    plots/toy_min_shift_equation.{png,pdf}   the mean-state length s = ||mean old-fact state||, measured
                                            (solid) and integrated from the scalar equation
                                            (dashed), with and without the normalization
    plots/toy_min_margins.{png,pdf}         the new facts' mean margins with the normalization:
                                            M (their individual parts) and G (the shared
                                            direction); the dotted line marks the peak of s

The equation is integrated along each run with the second-order term of a finite step included
(|D_perp|^2 / 2s; it vanishes in gradient flow and accounts for the whole gap at this rate --
see reduced_equation.py). Mean over seeds, band +/- 1 sd. Data: reduced_equation.json.

Run: JAX_PLATFORMS=cpu uv run python plot_reduced_equation.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, TEAL, DEEP, OLIVE, GREY = "#1B2A4E", "#2F8C7D", "#1E5E6B", "#8A8C30", "0.55"
XMAX = 1000


def series(d, norm, key):
    rs = [r for r in d.values() if r["norm"] == norm]
    st = np.array([x["step"] for x in rs[0]["rows"]], float)
    Y = np.array([[x[key] for x in r["rows"]] for r in rs]); keep = st <= XMAX
    return st[keep], Y[:, keep]


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})


def band(ax, st, Y, col, ls="-", lw=2.3, label=None, fill=True):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    if fill:
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
    ax.plot(st, m, color=col, ls=ls, lw=lw, label=label, zorder=3)
    return m


def main():
    d = json.load(open(os.path.join(HERE, "reduced_equation.json")))
    style()
    # --- the shift
    fig, ax = plt.subplots(figsize=(5.0, 3.55), dpi=200)
    st, S1 = series(d, 1, "s"); _, P1 = series(d, 1, "s_pred2")
    _, S0 = series(d, 0, "s"); _, P0 = series(d, 0, "s_pred2")
    m1 = band(ax, st, S1, TEAL, label="normalized readout")
    band(ax, st, P1, "0.15", ls=(0, (4, 3)), lw=1.3, fill=False)
    band(ax, st, S0, OLIVE, label="no normalization")
    band(ax, st, P0, "0.15", ls=(0, (4, 3)), lw=1.3, fill=False, label="equation")
    pk = st[int(m1.argmax())]
    ax.axvline(pk, color=GREY, ls=":", lw=1.2, zorder=1)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, None)
    ax.set_xlabel("Fine-tuning step"); ax.set_ylabel("Mean-state length $s$")
    fig.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=3,
               handlelength=1.8, columnspacing=1.4)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_shift_equation.{e}"), bbox_inches="tight")
    plt.close(fig)
    # --- the margins
    fig, ax = plt.subplots(figsize=(5.0, 3.55), dpi=200)
    _, M = series(d, 1, "M"); _, G = series(d, 1, "G")
    mM = band(ax, st, M, DEEP, label=r"$\bar M$, individual parts")
    band(ax, st, G, GREY, lw=1.8, label=r"$\bar G$, shared direction")
    ax.axhline(0, color="0.35", lw=0.8, zorder=1)
    ax.axvline(pk, color=GREY, ls=":", lw=1.2, zorder=1)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("Fine-tuning step"); ax.set_ylabel("New facts' margin (logits)")
    fig.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=2,
               handlelength=1.8, columnspacing=1.4)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_margins.{e}"), bbox_inches="tight")
    zc = st[int(np.argmax(mM > 0))]
    err = [np.abs(P - S).max(1) / (S.max(1) - S.min(1)) for S, P in ((S1, P1), (S0, P0))]
    print(f"{S1.shape[0]} seeds | normalized: s {S1.mean(0)[0]:.3f} -> peak {m1.max():.3f}@{pk:.0f} -> "
          f"{S1.mean(0)[-1]:.3f} | M crosses 0 at {zc:.0f} | no norm: {S0.mean(0)[0]:.3f} -> "
          f"{S0.mean(0)[-1]:.3f} | equation error {100 * err[0].mean():.1f}% / {100 * err[1].mean():.1f}% of range")
    print("wrote plots/toy_min_shift_equation, plots/toy_min_margins")


if __name__ == "__main__":
    main()
