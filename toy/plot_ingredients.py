"""The minimal ingredients: one panel per ingredient, the model against the ablation.

    plots/toy_min_ingredients.{png,pdf}

Old facts solid, new facts dotted. Every arm is gate-matched, so the new facts are learned in
all of them -- which is what makes the ablations readable: an arm where the new facts never
arrived would show no recovery for a trivial reason. Data: ingredients.json (ten seeds).

    (a) readout    the reversal needs the normalization: with ReLU or none, the old facts
                   crash and never return
    (b) keys       the crash needs a shared component: with beta = 0 there is no collapse
    (c) values     the crash needs a competing answer region: new answers spread over the
                   whole vocabulary produce no collapse
    (d) the write  removing the shared row of the store's update at every training step
                   prevents the collapse while the new facts are still learned

Run: JAX_PLATFORMS=cpu uv run python plot_ingredients.py
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, OLIVE = "#1B2A4E", "#C4245F", "#8A8C30"
XMAX = 1000
PANELS = (
    ("readout", (("model", NAVY, "the model"), ("relu", CRIM, "ReLU readout"),
                 ("linear", OLIVE, "no normalization"))),
    ("keys", "beta"),   # drawn from sw_beta.json: the whole beta range
    ("values", (("model", NAVY, "the model"), ("allvalues", CRIM, "answers over all values"))),
    ("the write", (("model", NAVY, "the model"), ("projected", CRIM, "shared row removed"))),
)


def series(d, tag, key):
    rs = [v for v in d.values() if v["tag"] == tag]
    st = np.array([x["step"] for x in rs[0]["rows"]], float)
    Y = np.array([[x[key] for x in r["rows"]] for r in rs])
    keep = st <= XMAX
    return st[keep], Y[:, keep]


def beta_panel(ax, key):
    """Old facts for every beta in the gate-matched sweep (rms readout, three seeds each)."""
    d = json.load(open(os.path.join(HERE, "sw_beta.json")))
    betas = sorted({r["beta"] for r in d.values() if r["norm"] == 1})
    cmap = plt.get_cmap("Blues")
    for i, b in enumerate(betas):
        rs = [r for r in d.values() if r["norm"] == 1 and r["beta"] == b]
        st = np.array([x["step"] for x in rs[0]["rows"]], float)
        A = np.array([[x[key] for x in r["rows"]] for r in rs]); keep = st <= XMAX
        col = NAVY if b == 0.5 else cmap(0.3 + 0.55 * i / max(len(betas) - 1, 1))
        ax.plot(st[keep], A.mean(0)[keep], color=col, lw=2.2 if b == 0.5 else 1.6,
                label=(r"$\beta = %g$" % b) + (" (the model)" if b == 0.5 else ""), zorder=3)
        print(f"keys      beta={b:<5g} trough {A.mean(0).min():.3f}@{st[A.mean(0).argmin()]:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="prob", choices=["prob", "acc"],
                    help="prob: mean probability on the correct value (smooth); acc: argmax")
    a = ap.parse_args()
    KA, KB = ("A_p", "B_p") if a.metric == "prob" else ("A", "B")
    ylab = "P(correct value)" if a.metric == "prob" else "First-token accuracy"
    suffix = "" if a.metric == "prob" else "_acc"
    d = json.load(open(os.path.join(HERE, "ingredients.json")))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), dpi=200, sharey=True)
    for ax, (title, arms) in zip(axes, PANELS):
        if arms == "beta":
            beta_panel(ax, KA)
            ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0); ax.set_xlabel("Injection step")
            ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=2, handlelength=1.4, labelspacing=0.2, columnspacing=1.0,
                      title=title, title_fontsize=10.5)
            continue
        for tag, col, lab in arms:
            st, A = series(d, tag, KA); _, B = series(d, tag, KB)
            m, sd = A.mean(0), A.std(0, ddof=1)
            ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
            ax.plot(st, m, color=col, lw=2.2, label=lab, zorder=3)
            ax.plot(st, B.mean(0), color=col, ls=":", lw=1.6, zorder=3)
            q = (m.argmin(), )
            print(f"{title:9s} {tag:10s} trough {m.min():.3f}@{st[int(m.argmin())]:.0f} "
                  f"-> after {m[int(m.argmin()):].max():.3f}  B end {B.mean(0)[-1]:.2f}")
        ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
        ax.set_xlabel("Injection step")
        ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=1, handlelength=1.6, labelspacing=0.25,
                  title=title, title_fontsize=10.5)
    axes[0].set_ylabel(ylab)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_ingredients{suffix}.{e}"), bbox_inches="tight")
    print(f"wrote plots/toy_min_ingredients{suffix}")


if __name__ == "__main__":
    main()
