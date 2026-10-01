"""Paper figures: the mechanism on the minimal model, four separate panels for one row.

    plots/toy_min_mechanism_a   A and B accuracy
    plots/toy_min_mechanism_b   the shared write's between-half content <s, w>
    plots/toy_min_mechanism_c   B's mean confidence margin
    plots/toy_min_mechanism_d   growth rate of the write along its own direction:
                                the normalizer's term vs. the total

Normalized readout only. Data: track_m{0..9}.json from
    verify_math.py --seed S --match 200 --norms 1 --save track_mS.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
NAVY, TEAL, DEEP, OLIVE, GREY, CRIM = "#1B2A4E", "#2F8C7D", "#1E5E6B", "#8A8C30", "0.62", "#C4245F"
SEEDS = tuple(range(10))
XMAX = 400
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def band(ax, x, Y, color, ls="-", lw=1.9, label=None):
    Y = np.asarray(Y); mu = Y.mean(0); sd = Y.std(0, ddof=1)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, mu, color=color, ls=ls, lw=lw, label=label)
    return mu


def save(fig, name):
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{name}.{e}"), bbox_inches="tight")
    print(f"wrote {name}")


def panel():
    return plt.subplots(figsize=(3.6, 3.1), dpi=200)


def main():
    tr = {s: json.load(open(os.path.join(HERE, f"track_m{s}.json")))[f"n1-s{s}"] for s in SEEDS}
    st = np.array([x["step"] for x in tr[0]["rows"]], float)
    keep = st <= XMAX
    x = st[keep]

    def grab(key):
        return np.array([[r[key] for r in tr[s]["rows"]] for s in SEEDS])[:, keep]

    mM = grab("margin").mean(0)
    cm = next((x[i] for i in range(1, len(x)) if mM[i - 1] < 0 <= mM[i]), None)

    def mark(ax, label=False):
        if cm is not None:
            ax.axvline(cm, color="0.35", lw=1.0, ls="--", alpha=0.85, zorder=0)
            if label:
                ax.text(cm + 6, 0.03, "new facts\nbecome confident", fontsize=7, color="0.35", va="bottom")
        ax.set_xlim(0, XMAX)
        ax.set_xlabel("injection step")

    fig, ax = panel()
    band(ax, x, grab("A"), NAVY, label="old facts")
    band(ax, x, grab("B"), CRIM, ls=":", lw=1.7, label="new facts")
    ax.set_ylim(0, 1.04); ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6)
    mark(ax, label=True); save(fig, "toy_min_mechanism_a")

    fig, ax = panel()
    band(ax, x, grab("s_on_w"), TEAL)
    ax.set_ylabel("shift toward the new facts' answers")
    ax.set_ylim(0, None)
    mark(ax); save(fig, "toy_min_mechanism_b")

    fig, ax = panel()
    band(ax, x, grab("margin"), DEEP)
    ax.axhline(0, color="0.45", lw=0.9)
    yl = ax.get_ylim()
    ax.text(XMAX * 0.97, 0.25, "right", fontsize=8, color="0.35", ha="right", va="bottom")
    ax.text(XMAX * 0.97, -0.6, "wrong", fontsize=8, color="0.35", ha="right", va="top")
    ax.set_ylabel("confidence of the new facts\n(mean margin)")
    mark(ax); save(fig, "toy_min_mechanism_c")

    fig, ax = panel()
    P, Pa = grab("proj") * 1e3, grab("proj_anti") * 1e3
    band(ax, x, P, TEAL, label="total")
    band(ax, x, Pa, OLIVE, lw=1.5, label="normalization's term")
    ax.axhline(0, color="0.45", lw=0.9)
    # the first steps (+70) are off-scale on purpose: the sign change is the point
    lo, hi = -14.0, 24.0
    ax.set_ylim(lo, hi)
    ax.axhspan(0, hi, color=CRIM, alpha=0.05, lw=0); ax.axhspan(lo, 0, color=TEAL, alpha=0.07, lw=0)
    ax.text(XMAX * 0.97, hi * 0.94, "shift being written", fontsize=7.5, color="0.35", ha="right", va="top")
    ax.text(XMAX * 0.97, lo * 0.94, "shift being withdrawn", fontsize=7.5, color="0.35", ha="right", va="bottom")
    ax.set_ylabel("rate of change of the shift")
    ax.set_yticks([-10, 0, 10, 20])
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.8)
    mark(ax); save(fig, "toy_min_mechanism_d")

    print(f"margin crosses zero at step {cm}")
    print("per seed: margin cross / normalizer's term flips / total turns:",
          [(tr[s]["margin_cross"], tr[s]["anti_cross"], tr[s]["proj_cross"]) for s in SEEDS])


if __name__ == "__main__":
    main()
