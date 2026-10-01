"""Figure 1, right panel: the two processes measured in the 8-layer transformer we pretrain.

    ||delta||       the common part of A's displacement at the readout state (mean over A)
    rms ||eps_a||   the individual remainder

from delta_structure/standard-p16000-disjoint-t1200-seed*.npz (last layer = the readout
state; the second axis is averaged), three seeds,
mean and one sd. A's trough is read from weight_patch/standard-head-acc-*.npz and marked.
Same figure size, type size and linear x-axis style as the OLMo panel (llm/plot_seeds.py
--paper); colours match the schematic strip (teal = common, olive = individual).

Run: uv run python scripts/plot_fig1_transformer.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, OLIVE, MARK = "#2F8C7D", "#8A8C30", "#9A9A9A"
XMAX = 400


def main():
    ds = [np.load(f) for f in sorted(glob.glob("delta_structure/standard-p16000-disjoint-t1200-seed*.npz"))]
    acc = [np.load(f) for f in sorted(glob.glob("weight_patch/standard-head-acc-p16000-disjoint-seed*.npz"))]
    st = ds[0]["steps"].astype(float)
    D = np.stack([np.linalg.norm(d["delta_layer"][:, :, -1, :].astype(np.float64), axis=-1).mean(1) for d in ds])
    E = np.stack([d["eps_layer_rms"][:, :, -1].astype(np.float64).mean(1) for d in ds])
    ast = acc[0]["steps"].astype(float)
    A = np.stack([a["acc_t"].mean(-1) for a in acc]).mean(0)
    tr = int(ast[int(A.argmin())])
    keep = st <= XMAX
    print(f"{len(ds)} seeds; A trough at step {tr} ({A.min():.3f}); "
          f"||delta|| peak {D.mean(0).max():.2f} @{int(st[D.mean(0).argmax()])}, at 400 {D.mean(0)[keep][-1]:.2f}; "
          f"rms||eps|| at 400 {E.mean(0)[keep][-1]:.2f}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    for Y, c, lab in ((D, TEAL, "common shift  $\\|\\delta\\|$"),
                      (E, OLIVE, "individual displacement  rms$\\,\\|\\varepsilon_a\\|$")):
        m, sd = Y.mean(0)[keep], Y.std(0, ddof=1)[keep]
        ax.fill_between(st[keep], m - sd, m + sd, color=c, alpha=0.18, lw=0)
        ax.plot(st[keep], m, color=c, lw=2.4, marker="o", ms=4.5, label=lab, zorder=3)
    ax.axvline(tr, color=MARK, ls=":", lw=1.4, zorder=1)
    ax.text(tr, ax.get_ylim()[1] * 0.0 + max(D.max(), E.max()) * 1.08, "trough", color=MARK,
            style="italic", fontsize=11, ha="center", va="bottom", clip_on=False)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, max(D.max(), E.max()) * 1.14)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("Displacement of the readout state")
    ax.legend(frameon=False, fontsize=11, loc="lower right", handlelength=1.6)
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/fig1_transformer.{e}", bbox_inches="tight")
    print("wrote plots/fig1_transformer.png/.pdf")


if __name__ == "__main__":
    main()
