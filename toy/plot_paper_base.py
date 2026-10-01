"""Paper figures for the minimal toy: two standalone accuracy curves, one per readout.

    plots/toy_min_linear_acc.{png,pdf}   z = h U        (keys -> one matrix -> softmax)
    plots/toy_min_norm_acc.{png,pdf}     z = rms(h) U   (one normalizer added)

Same store, same data, same optimizer, same seeds; the readout is the only difference.
Data: paper_base.json (paper_base_runs.py): beta = 0.5, n = 128, one learning rate, TEN
seeds, injection rate gate-matched per seed and arm so B reaches 0.99 at step 200. Style
follows the transformer figures: navy A, crimson B, mean and one sd, symlog step, dpi 200.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM = "#1B2A4E", "#C4245F"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def curves(norm, fname="paper_base.json"):
    d = json.load(open(os.path.join(HERE, fname)))
    runs = [r["rows"] for r in d.values() if r["norm"] == norm]
    st = np.array([x["step"] for x in runs[0]], float)
    A = np.array([[x["A"] for x in r] for r in runs])
    B = np.array([[x["B"] for x in r] for r in runs])
    return st, A, B


def one(norm, out, fname="paper_base.json"):
    st, A, B = curves(norm, fname)
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    for Y, col, lab in ((A, NAVY, "A"), (B, CRIM, "B")):
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.plot(st, mu, color=col, lw=2.0, label=lab)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("injection step")
    ax.set_ylabel("first-token accuracy")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{out}.{e}"), bbox_inches="tight")
    mu = A.mean(0); tr = int(mu.argmin())
    print(f"{out} ({len(A)} seeds): A trough {mu[tr]:.3f} at step {int(st[tr])}, max after "
          f"{mu[tr:].max():.3f}, end {mu[-1]:.3f}; B reaches 0.99 at step "
          f"{int(st[np.argmax(B.mean(0) >= 0.99)]) if (B.mean(0) >= 0.99).any() else -1}")


if __name__ == "__main__":
    one(0, "toy_min_linear_acc")
    one(1, "toy_min_norm_acc")
