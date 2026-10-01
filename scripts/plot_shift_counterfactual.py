"""The causal test of the delta + eps split in the transformer: A's accuracy under three
readouts at every dense checkpoint,

    trained                  z_a(t)                what happened
    common shift only        z_a(0) + c(t)         h_a(0) + delta_t
    individual only          z_a(0) + v_a(t)       h_a(0) + eps_a(t)  =  h_a(t) - delta_t

from shift_decomposition/{standard,mlp_free}-p16000-disjoint-t1200-seed*.npz, per-fact
accuracy averaged over facts and attributes, mean and seed range where more than one seed
exists. Two figures. The main one is the attention-only arm, all three readouts. The appendix one
is both arms side by side; with the MLPs the "common shift only" counterfactual stops at the
decision boundary instead of crossing it (the two parts are not additive at the argmax). Log x-axis: the three phases live
inside the first 200 steps of a 1200-step run.

    plots/tf_counterfactual.{png,pdf}          main text, attention-only arm, three curves
    plots/tf_counterfactual_arms.{png,pdf}     appendix, both arms, three curves

Run: uv run python scripts/plot_shift_counterfactual.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

NAVY, TEAL, OLIVE = "#1B2A4E", "#2F8C7D", "#8A8C30"
# the main figure uses plot_delta_eps's colours and phase designations so the two read as one
C_DELTA, C_EPS, MARK = "#5B3A7A", "#8A8C30", "#9A9A9A"
XTICKS = [10, 25, 50, 100, 200, 400, 1200]
TROUGH_STEP, RECOVERY_END = 60, 190          # from mlpfree_p16000_a_vs_b
ARMS = (("standard", "standard (with MLPs)"), ("mlp_free", "attention-only"))
CURVES = (("acc_trained", NAVY, "-", "trained"),
          ("acc_const", TEAL, "--", "common shift only  $h_a^{(0)}+\\delta_t$"),
          ("acc_vary", OLIVE, "-.", "individual only  $h_a(t)-\\delta_t$"))


def load(arm):
    fs = sorted(glob.glob(f"shift_decomposition/{arm}-p16000-disjoint-t1200-seed*.npz"))
    ds = [np.load(f) for f in fs]
    st = ds[0]["steps"].astype(float)
    out = {k: np.stack([d[k].astype(np.float64).mean((1, 2)) for d in ds]) for k, *_ in CURVES}
    return st, out, len(ds)


def draw(ax, arm, curves, n_label=True):
    st, out, n = load(arm)
    for k, col, ls, lab in curves:
        Y = out[k]; m = Y.mean(0)
        if n > 1:
            ax.fill_between(st, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=2.0, marker="o", ms=3.5, label=lab)
    tr = int(out["acc_trained"].mean(0).argmin())
    ax.axvline(st[tr], color="0.6", ls=":", lw=1.0)
    ax.text(st[tr], 1.03, "trough", color="0.45", style="italic", fontsize=9, ha="center", va="bottom")
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 1200); ax.set_ylim(0, 1.05)
    ax.set_xlabel("injection step")
    return st, out, n


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
                         "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})
    # main figure: the attention-only arm, all three readouts, styled like delta_eps_left
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "axes.spines.top": True,
                         "axes.spines.right": True, "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    st, out, n = load("mlp_free")
    keep = st > 0
    for k, col, ls, lab in ((CURVES[0][0], NAVY, "-", "trained"),
                            (CURVES[1][0], C_DELTA, "--", "common shift only"),
                            (CURVES[2][0], C_EPS, "-.", "individual only")):
        Y = out[k][:, keep]; m = Y.mean(0)
        ax.fill_between(st[keep], Y.min(0), Y.max(0), color=col, alpha=0.20, lw=0)
        ax.plot(st[keep], m, color=col, ls=ls, lw=2.6, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(XTICKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(st[keep].min(), st[keep].max()); ax.set_ylim(0, 1.05)
    bounds = [st[keep].min(), TROUGH_STEP, RECOVERY_END, st[keep].max()]
    for x in bounds[1:-1]:
        ax.axvline(x, color=MARK, ls=":", lw=1.4, zorder=1)
    for (x0, x1), lab in zip(zip(bounds, bounds[1:]), ("(i)", "(ii)", "(iii)")):
        ax.text(np.sqrt(x0 * x1), 1.05 * 1.01, lab, color=MARK, style="italic", fontsize=12,
                va="bottom", ha="center", clip_on=False)
    ax.set_xlabel("Injection step"); ax.set_ylabel("Accuracy on the old facts")
    ax.legend(frameon=False, fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.64), handlelength=1.6)
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/tf_counterfactual.{e}", bbox_inches="tight")
    plt.close(fig)

    # appendix figure: both arms, all three readouts
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200, sharey=True)
    for ax, (arm, title) in zip(axes, ARMS):
        st, out, n = draw(ax, arm, CURVES)
        ax.set_title(f"{title}, {n} seeds", fontsize=10, color="0.25", pad=14)
        m = {k: out[k].mean(0) for k, *_ in CURVES}; tr = int(m["acc_trained"].argmin())
        print(f"{arm} ({n} seeds): trough step {int(st[tr])}  trained {m['acc_trained'][tr]:.3f}  "
              f"const-only {m['acc_const'][tr]:.3f}  vary-only {m['acc_vary'][tr]:.3f};  "
              f"end trained {m['acc_trained'][-1]:.3f}  const {m['acc_const'][-1]:.3f}  vary {m['acc_vary'][-1]:.3f}")
    axes[0].set_ylabel("accuracy on the old facts")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", handlelength=2.2)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"plots/tf_counterfactual_arms.{e}", bbox_inches="tight")
    print("wrote plots/tf_counterfactual{,_arms}.png/.pdf")


if __name__ == "__main__":
    main()
