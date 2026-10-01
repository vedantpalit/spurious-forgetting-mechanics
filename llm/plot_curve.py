"""A-vs-B curve for an LLM injection run, in the small model's figure style.

Matches `a_vs_b_zucchet_style.png`: navy A, crimson B, a standard-error band on A, dotted
phase markers with rotated italic labels, boxed spines, no grid.

A IS THE NON-COPY STRATUM. 62% of the gated set is answerable by copying the answer out of the
prompt, and plotting the undifferentiated set would dilute a 0.87 -> 0.19 crash into
a much shallower one. The copy stratum is a different behaviour, not a weaker version of the
same one, so it is not averaged in.

THE WINDOW ENDS AT B'S CEILING BY DEFAULT, not at the last step run. Everything after that is
continued optimisation against a saturated task -- A does keep eroding there, which is itself
worth knowing, but it is not part of the injection event and letting it set the axis squashes
the event into the first tenth of the plot. `--xmax` overrides.

Run:
  .venv/Scripts/python -m llm.plot_curve --run llm/out/inject_lr1e-05_seed0.json
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_COLOR = "#1B2A4E"
B_COLOR = "#C4245F"
MARK = "#9A9A9A"


def phases(steps, acc):
    """(trough index, peak index) -- the crashed point with the largest subsequent rebound.

    Not the global minimum: the curve crashes, recovers, then erodes for thousands of steps, so
    the global minimum is the last step. And candidates must be a real crash (0.05 below the
    running maximum), because every arm rises ~0.05 before it falls and at low LR that rise
    outsizes the recovery.
    """
    cand = [k for k in range(len(acc)) if acc[k] <= max(acc[:k + 1]) - 0.05]
    if not cand:
        return None, None
    i = max(cand, key=lambda k: max(acc[k:]) - acc[k])
    return i, i + acc[i:].index(max(acc[i:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="llm/out/inject_lr1e-05_seed0.json")
    ap.add_argument("--out", default="llm_a_vs_b_zucchet_style.png")
    ap.add_argument("--xmax", type=int, default=None, help="default: 2x B's ceiling step")
    a = ap.parse_args()

    d = json.load(open(a.run, encoding="utf-8"))
    if not d.get("tokens_per_step"):
        raise SystemExit(f"{a.run} predates fp32 master weights and is voided")
    c = d["curve"]
    step = [x["step"] for x in c]
    acc = [x["A/noncopy/acc"] for x in c]
    se = [x["A/noncopy/acc_se"] for x in c]
    b = [x["B/ALL/acc"] for x in c]

    ceil_i = next(k for k, v in enumerate(b) if v >= 0.95 * b[-1])
    xmax = a.xmax or min(2 * step[ceil_i], step[-1])
    n = sum(1 for s in step if s <= xmax)
    step, acc, se, b = step[:n], acc[:n], se[:n], b[:n]
    tr, pk = phases(step, acc)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
        "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)

    ax.fill_between(step, [v - s for v, s in zip(acc, se)],
                    [v + s for v, s in zip(acc, se)], color=A_COLOR, alpha=0.18, linewidth=0)
    ax.plot(step, acc, color=A_COLOR, linewidth=2.4, label="A (non-copy)", zorder=3)
    ax.plot(step, b, color=B_COLOR, linewidth=2.4, label="B", zorder=3)

    for idx, text in ((tr, "crash"), (pk, "erosion")):
        if idx is None:
            continue
        ax.axvline(step[idx], color=MARK, linestyle=":", linewidth=1.3, zorder=1)
        ax.text(step[idx] - xmax * 0.016, 0.53, text, rotation=90, color=MARK,
                style="italic", fontsize=12, va="center", ha="center")

    ax.set_xlabel("Injection step")
    ax.set_ylabel("First-token accuracy")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62),
          fontsize=13, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"lr={d['lr']:g}  baseline {acc[0]:.3f}  trough {acc[tr]:.3f}@{step[tr]}  "
          f"peak {acc[pk]:.3f}@{step[pk]}  "
          f"recovery {(acc[pk]-acc[tr])/(acc[0]-acc[tr]):.3f}  "
          f"B ceiling @{step[ceil_i]}  window 0-{xmax}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
