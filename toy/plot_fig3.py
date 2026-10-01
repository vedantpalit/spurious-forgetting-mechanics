"""Figure 3: why the shared shift reverses, on the minimal model.

    plots/fig3.{png,pdf}

(a) The old facts' hidden states h_a (before the normalization) at steps 0, 65 and 400, relative
    to their pretrained mean (for the picture only: pretraining leaves s(0) ~ 0.4, not 0). x along
    the mean's move at step 65, y the leading direction of the spread orthogonal to
    the mean's moves at steps 65 and 400, so all three means sit on the x axis. The arrow is the mean's
    move. The cloud moves out as one block and comes back.
(b) The mean hidden state norm s = ||mean old-fact state||, with and without the normalization.
(c) The new facts' mean margins with the normalization: the individual margin crosses zero at the
    peak of s (dotted); the shared margin is ~0 after the first steps.

(a): seed 0, base configuration, gate-matched rate (the same as (b), (c)). (b), (c): ten seeds,
reduced_equation.json. Style: a shared set_default_style, with
the serif font, the colors and no minor ticks.

Run: JAX_PLATFORMS=cpu uv run python plot_fig3.py
"""
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import kmin as m
from sweeps import match_gate
from plot_reduced_equation import series

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
FIG_WIDTH = 397 / 72                                  # the paper's text width; saved at exactly this size
NAVY, TEAL, DEEP, OLIVE, GREY = "#1B2A4E", "#2F8C7D", "#1E5E6B", "#8A8C30", "0.55"
SEED, MOMENTS, XMAX = 0, (0, 65, 400), 500


def set_default_style():
    matplotlib.rcParams["lines.linewidth"] = 1
    matplotlib.rcParams["lines.markersize"] = 3
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
    matplotlib.rcParams["legend.fontsize"] = 5
    matplotlib.rcParams["axes.labelsize"] = 6
    matplotlib.rcParams["xtick.labelsize"] = 5
    matplotlib.rcParams["ytick.labelsize"] = 5
    matplotlib.rcParams["legend.title_fontsize"] = 6
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["figure.dpi"] = 250
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams["xtick.major.size"] = 2
    matplotlib.rcParams["xtick.major.width"] = 0.5
    matplotlib.rcParams["ytick.major.size"] = 2
    matplotlib.rcParams["ytick.major.width"] = 0.5
    matplotlib.rcParams["axes.titlesize"] = 7
    matplotlib.rcParams["axes.titlepad"] = 10
    matplotlib.rcParams["axes.linewidth"] = 0.5
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def trajectory():
    """Old facts' hidden states h = k W (before the normalization) at MOMENTS."""
    p0, lr, gate, mu, A, Dd, B = m.pretrain(SEED, 0.5, 1)
    ilr, g, ok = match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, SEED, 200)
    (kA, vA), _, (kB, vB) = A, Dd, B
    rng = np.random.default_rng(SEED + 5); p = p0; H = {}
    for t in range(max(MOMENTS) + 1):
        if t in MOMENTS:
            H[t] = np.asarray(m.fwd(p, jnp.asarray(kA), 1)[1])
        i = rng.integers(0, m.NB, 32)
        p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, 1)
    return H


def band(ax, x, Y, color, ls="-", label=None):
    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
    ax.plot(x, mu, color=color, ls=ls, label=label)
    return mu


def main():
    set_default_style()
    H = trajectory()
    # for the picture only: states relative to the pretrained mean, so the mean starts at the
    # origin; x along the mean's move at step 65, y the leading direction of the spread
    # orthogonal to the mean's move at steps 65 and 400, so the means have no y component
    ref = H[MOMENTS[0]].mean(0)
    X = {t: H[t] - ref for t in MOMENTS}
    d1, d2 = X[MOMENTS[1]].mean(0), X[MOMENTS[2]].mean(0)
    Q = np.linalg.qr(np.stack([d1, d2], 1))[0]                  # span of the two mean moves
    e1 = d1 / np.linalg.norm(d1)
    E = np.concatenate([X[t] - X[t].mean(0) for t in MOMENTS]); E -= (E @ Q) @ Q.T
    e2 = np.linalg.svd(E, full_matrices=False)[2][0]
    proj = lambda h: np.stack([h @ e1, h @ e2], -1)
    allP = np.concatenate([proj(X[t]) for t in MOMENTS])
    lo, hi = allP.min(0), allP.max(0)
    pad = 0.04 * (hi - lo); lo[0], hi[0] = lo[0] - pad[0], hi[0] + pad[0]
    hi[0] += 0.14 * (hi[0] - lo[0])                     # room for the step labels on the right

    fig = plt.figure(figsize=(FIG_WIDTH, 1.6))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.0], wspace=0.36,
                             left=0.03, right=0.985, top=0.979, bottom=0.195)
    # the three rows fill the left block from the legend down into the band the tick labels
    # take on the right (the rows have none), with a clear gap under the legend
    blk = outer[0].get_position(fig)
    y_bot, y_top, gap = blk.y0 - 0.07, blk.y1 - 0.14, 0.02
    hr = (y_top - y_bot - 2 * gap) / 3
    axes_a = [fig.add_axes([blk.x0, y_top - (k + 1) * hr - k * gap, blk.width, hr]) for k in range(3)]
    for k, (ax, t) in enumerate(zip(axes_a, MOMENTS)):
        P = proj(X[t]); c = P.mean(0)
        ax.axvline(0, color="0.85", lw=0.4, zorder=0)
        ax.plot([lo[0], P[:, 0].max()], [0, 0], color="0.85", lw=0.4, zorder=0)   # stops at the cloud
        ax.scatter(P[:, 0], P[:, 1], s=1.2, color=NAVY, alpha=0.4, edgecolors="none", zorder=2, clip_on=False)
        if np.linalg.norm(c) > 0:
            ax.annotate("", xy=c, xytext=(0, 0), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=5))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(1.0, 0.5, f"step {t}", transform=ax.transAxes, ha="right", va="center",
                fontsize=matplotlib.rcParams["legend.fontsize"])
        print(f"step {t:3d}  mean move {np.linalg.norm(X[t].mean(0)):.3f}  in plane "
              f"{np.linalg.norm(proj(X[t].mean(0))) / max(np.linalg.norm(X[t].mean(0)), 1e-12):.3f}  mean (x, y) ({c[0]:.3f}, {c[1]:.1e})")
    axes_a[1].set_ylabel("orthogonal")
    hs = [Line2D([], [], ls="", marker="o", ms=1.5, mew=0, color=NAVY, alpha=0.6, label="old hidden states"),
          Line2D([], [], color=TEAL, lw=0.9, marker=">", ms=2.5, mew=0, label="mean hidden state")]
    fig.legend(handles=hs, frameon=False, loc="upper center", bbox_to_anchor=((blk.x0 + blk.x1) / 2, blk.y1),
               ncol=2, handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)

    d = json.load(open(os.path.join(HERE, "reduced_equation.json")))
    axb, axc = fig.add_subplot(outer[1]), fig.add_subplot(outer[2])
    st, S1 = series(d, 1, "s"); _, S0 = series(d, 0, "s")
    keep = st <= XMAX; st = st[keep]
    s1 = band(axb, st, S1[:, keep], TEAL, label="normalized")
    band(axb, st, S0[:, keep], OLIVE, label="not normalized")
    pk = st[int(s1.argmax())]
    _, Mm = series(d, 1, "M"); _, Gm = series(d, 1, "G")
    band(axc, st, Mm[:, keep], DEEP, label=r"individual $\bar\gamma_{\mathrm{ind}}$")
    band(axc, st, Gm[:, keep], GREY, label=r"shared $\bar\gamma_{\mathrm{sh}}$")
    axc.axhline(0, color="0.6", lw=0.5, zorder=1)
    for ax in (axb, axc):
        ax.axvline(pk, color="0.6", ls=":", lw=0.8, zorder=1)
        ax.set_xlim(0, XMAX); ax.set_xlabel("finetuning step")
    axb.set_ylim(0, None); axb.set_ylabel("mean hidden state norm s")
    axc.set_ylabel("margin")
    axb.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.56), handlelength=1.5, handletextpad=0.4)
    axc.legend(frameon=False, loc="lower right", handlelength=1.5)
    # the left x label at the height of the right ones (no tick labels on the left)
    fig.canvas.draw()
    yl = axb.xaxis.label.get_window_extent().y0 / fig.bbox.height
    b2 = axes_a[2].get_position()
    fig.text((b2.x0 + b2.x1) / 2, yl, "common shift direction", ha="center", va="bottom",
             fontsize=matplotlib.rcParams["axes.labelsize"])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig3.{e}"))
    print(f"peak of s at step {pk:.0f}; wrote plots/fig3")


if __name__ == "__main__":
    main()
