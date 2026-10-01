"""With ballast vs without, 8-layer standard arm: A and B during injection.

Parsed from logs_scale8/. Both arms are 5 seeds, same pretrain config, same LR, same batch,
same 50/50 partition -- ballast presence is the only difference.

THE X-AXIS IS B'S ACCURACY, NOT THE STEP, and that is the whole point. Without ballast, half Y
is never trained during pretraining, and in `disjoint` B draws from half Y -- so B's value
tokens start cold and B is acquired 3.6x more slowly (ceiling at step 360 against 100). Plotting
against step would compare the arms at points where different amounts of B have been learned,
which is a confound. The damage tracks B's acquisition rather than the step count, so B's own
accuracy is the axis that matches the arms.

`--xaxis step` gives the step version anyway, because the timing difference is itself a result
and hiding it would be its own distortion.

THE ARM IS CONFOUNDED BY CONSTRUCTION AND THAT IS WHY IT EXISTS. "No ballast" is not a clean
manipulation of one variable: it necessarily also means B's value tokens are cold. Ballast was
introduced precisely to remove that confound. This arm shows what the confound does when it is
left in.

Run: uv run python -m scripts.plot_scale8_ballast_contrast
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = "logs_scale8"
OUT_DIR = "plots"
A_COLOR = "#1B2A4E"
B_COLOR = "#C4245F"

LINE = re.compile(
    r"\[step (\d+)\].*?dataA: first_acc=([\d.]+).*?dataB: first_acc=([\d.]+)")

ARMS = {
    "with ballast": "scale8_injection.p{p}.{c}.seed*.out",
    "without ballast": "scale8_noballast.p{p}.{c}.t1200.seed*.out",
}


def load(pattern, pretrain_step):
    runs = []
    for f in sorted(glob.glob(os.path.join(IN_DIR, pattern))):
        r = {}
        for line in open(f, encoding="utf-8", errors="replace"):
            m = LINE.search(line)
            if m:
                r[int(m.group(1)) - pretrain_step] = (float(m.group(2)), float(m.group(3)))
        if r:
            runs.append(r)
    if not runs:
        raise SystemExit(f"no logs matching {pattern}")
    st = sorted(set.intersection(*(set(r) for r in runs)))
    A = np.array([[r[s][0] for s in st] for r in runs])
    B = np.array([[r[s][1] for s in st] for r in runs])
    return np.array(st, float), A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--xaxis", default="b_acc", choices=["b_acc", "step"])
    ap.add_argument("--xmax", type=int, default=1200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = {}
    for name, pat in ARMS.items():
        st, A, B = load(pat.format(p=a.pretrain_step, c=a.condition), a.pretrain_step)
        data[name] = (st, A, B)
        ceil = st[int(np.argmax(B.mean(0) >= 0.99))]
        print(f"  {name:>16}: {A.shape[0]} seeds, B reaches 0.99 at step {ceil:.0f}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)

    for name, ls in (("with ballast", "-"), ("without ballast", "--")):
        st, A, B = data[name]
        ns = A.shape[0]
        Am, Bm = A.mean(0), B.mean(0)
        Ase = A.std(0, ddof=1) / np.sqrt(ns)
        if a.xaxis == "step":
            keep = st <= a.xmax
            x = st[keep]
            ax.plot(x, Bm[keep], color=B_COLOR, ls=ls, lw=2.2, zorder=2)
        else:
            # Only the strictly-increasing part of B: once B saturates, many steps share one
            # x and the curve would fold back on itself.
            keep = np.zeros(len(st), bool)
            best = -1.0
            for k, v in enumerate(Bm):
                if v > best + 1e-9:
                    keep[k] = True
                    best = v
            x = Bm[keep]
        ax.fill_between(x, (Am - Ase)[keep], (Am + Ase)[keep], color=A_COLOR,
                        alpha=0.15, lw=0)
        ax.plot(x, Am[keep], color=A_COLOR, ls=ls, lw=2.4, label=f"A, {name}", zorder=3)

    if a.xaxis == "step":
        ax.set_xlabel("Injection step")
        ax.set_xlim(0, a.xmax)
        ax.plot([], [], color=B_COLOR, lw=2.2, label="B")
    else:
        ax.set_xlabel("B first-token accuracy (acquisition progress)")
        ax.set_xlim(0, 1.0)
    ax.set_ylabel("A first-token accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=12, handlelength=2.2)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = a.out or f"scale8_p{a.pretrain_step}_ballast_contrast_{a.xaxis}"
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
