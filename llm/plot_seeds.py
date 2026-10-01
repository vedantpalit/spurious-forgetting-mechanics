"""A vs B for the OLMo 2 1B injection, one figure per learning rate, mean +/- sd over seeds.

Style follows `plot_curve.py` and the 8-layer figures: navy A, crimson B, serif type, boxed
spines, no grid, dpi 200, and vertical dotted rules dividing (i) the crash, (ii) the recovery,
(iii) the erosion, with the numerals in the headroom above y=1.

A IS THE NON-COPY STRATUM. 62% of the gated set is answerable by copying the answer out of the
prompt; plotting the undifferentiated set dilutes a 0.87 -> 0.18 crash into a much
shallower one. Copying is an in-context operation, not parametric recall, so it is a different
behaviour rather than a weaker version of the same one. --stratum ALL overrides.

THE BAND IS THE SAMPLE SD ACROSS SEEDS, not the per-item standard error `plot_curve.py` uses
for a single run. With three seeds it is a wide, honest band rather than a decoration: at
lr 1e-5 the peak ranges 0.282 to 0.344 across seeds, so a figure drawn from one seed would put
the recovery destination 11% too high.

WINDOW. Ends at twice B's ceiling step by default, per learning rate, because the three arms run
on different timescales and a common axis squashes the fast one. Pass --xmax for a shared axis
when the three are shown side by side.

All nine runs share one step grid, so seeds are averaged directly; this is asserted, not assumed.

Run:
  uv run python -m llm.plot_seeds
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots"
A_COLOR = "#1B2A4E"
B_COLOR = "#C4245F"
MARK = "#9A9A9A"
SPAN_Y = 1.045


def phases(acc):
    """(trough, peak) -- the crashed point with the largest subsequent rebound.

    Not the global minimum: the curve crashes, recovers, then erodes for thousands of steps, so
    the global minimum is the last step. Candidates must be a real crash (0.05 below the running
    maximum), because every arm rises ~0.05 before it falls and at low LR that rise outsizes the
    recovery.
    """
    cand = [k for k in range(len(acc)) if acc[k] <= max(acc[:k + 1]) - 0.05]
    if not cand:
        return None, None
    i = max(cand, key=lambda k: max(acc[k:]) - acc[k])
    return i, i + int(np.argmax(acc[i:]))


def load(lr_tag, stratum):
    files = sorted(glob.glob(f"llm/out/inject_lr{lr_tag}_seed*.json"))
    if not files:
        raise SystemExit(f"no runs matching llm/out/inject_lr{lr_tag}_seed*.json")
    steps, A, B, seeds = None, [], [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if not d.get("tokens_per_step"):
            print(f"  skipping {os.path.basename(f)}: predates fp32 master weights, voided")
            continue
        c = d["curve"]
        s = [x["step"] for x in c]
        if steps is None:
            steps = s
        elif s != steps:
            raise SystemExit(f"{f} has a different step grid; seeds cannot be averaged directly")
        A.append([x[f"A/{stratum}/acc"] for x in c])
        B.append([x["B/ALL/acc"] for x in c])
        seeds.append(d["seed"])
    return np.array(steps, float), np.array(A), np.array(B), seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", default="3e-06,5e-06,1e-05")
    ap.add_argument("--stratum", default="noncopy", choices=["noncopy", "ALL", "copy"])
    ap.add_argument("--xmax", type=int, default=None,
                    help="shared x limit; default is 2x B's ceiling step, per learning rate")
    ap.add_argument("--no_markers", action="store_true")
    ap.add_argument("--paper", action="store_true",
                    help="Figure 1 left panel: legend 'old facts' / 'new facts', the spans "
                         "labelled suppression / recovery / erosion, written to plots/fig1_olmo")
    a = ap.parse_args()
    LAB = ("old facts", "new facts") if a.paper else ("A", "B")
    SPANS = ("suppression", "recovery", "erosion") if a.paper else ("(i)", "(ii)", "(iii)")

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })

    for tag in a.lrs.split(","):
        steps, A, B, seeds = load(tag, a.stratum)
        n = A.shape[0]
        Am, Asd, Bm, Bsd = A.mean(0), A.std(0, ddof=1), B.mean(0), B.std(0, ddof=1)

        ceil_i = next(k for k, v in enumerate(Bm) if v >= 0.95 * Bm[-1])
        xmax = a.xmax or int(min(2 * steps[ceil_i], steps[-1]))
        keep = steps <= xmax
        x = steps[keep]

        fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
        for m, sd, c, lab in ((Am, Asd, A_COLOR, LAB[0]), (Bm, Bsd, B_COLOR, LAB[1])):
            ax.fill_between(x, (m - sd)[keep], (m + sd)[keep], color=c, alpha=0.18, lw=0)
            ax.plot(x, m[keep], color=c, lw=2.4, label=lab, zorder=3)

        tr, pk = phases(list(Am[keep]))
        if not a.no_markers and tr is not None:
            bounds = [0, x[tr], x[pk], xmax]
            for v in bounds[1:-1]:
                ax.axvline(v, color=MARK, ls=":", lw=1.4, zorder=1)
            for k, ((x0, x1), lab) in enumerate(zip(zip(bounds, bounds[1:]), SPANS)):
                # the first two spans are narrow (100 steps each); with word labels the
                # second one is raised so the two do not collide
                y = SPAN_Y + (0.055 if (a.paper and k == 1) else 0.0)
                ax.text((x0 + x1) / 2, y, lab, color=MARK, style="italic",
                        fontsize=11 if a.paper else 12, va="bottom", ha="center", clip_on=False)

        ax.set_xlabel("Injection step")
        ax.set_ylabel("First-token accuracy")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.14)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62),
                  fontsize=13, handlelength=1.6)
        fig.tight_layout()

        stem = "fig1_olmo" if a.paper else f"llm_a_vs_b_lr{tag}_{a.stratum}"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), bbox_inches="tight")
        plt.close(fig)

        print(f"  lr {tag}  n={n} seeds {seeds}  window 0-{xmax}  B ceiling @{steps[ceil_i]:.0f}")
        if tr is not None:
            print(f"    trough {Am[keep][tr]:.3f} +/- {Asd[keep][tr]:.3f} @{x[tr]:.0f}   "
                  f"peak {Am[keep][pk]:.3f} +/- {Asd[keep][pk]:.3f} @{x[pk]:.0f}   "
                  f"B at trough {Bm[keep][tr]:.4f}")
        print(f"    wrote {OUT_DIR}/{stem}.png and .pdf")


if __name__ == "__main__":
    main()
