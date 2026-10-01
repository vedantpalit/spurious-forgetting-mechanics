"""Paper figures: the normalized minimal toy's accuracy curve across the knobs, overlaid.

    plots/toy_min_sweep_d.{png,pdf}      key dimension d             (sw_d.json)
    plots/toy_min_sweep_beta.{png,pdf}   shared key fraction beta    (sw_beta.json)
    plots/toy_min_sweep_nb.{png,pdf}     number injected n_B         (sw_nb.json)
    plots/toy_min_sweep_dose.{png,pdf}   injection rate              (sw_dose.json)
    plots/toy_min_sweep_load.{png,pdf}   n_A = n_D                   (sw_load.json)   supplementary
    plots/toy_min_sweep_vocab.{png,pdf}  |V|                         (sw_vocab.json)  supplementary

A's curves are drawn as a mint-to-navy gradient, dark at the largest value of the knob. B is
drawn once, in grey, as the mean over every cell of the sweep: the matched sweeps bisect the
injection rate so B reaches 0.99 at step 200 in every cell. The dose sweep varies the rate
on purpose, so there B is drawn once per rate, thin, with the reference rate bold.
Normalized arm only (z = rms(h) U).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAMP = LinearSegmentedColormap.from_list("mint_navy",
                                         ["#B9E3CB", "#63BB96", "#2F8C7D", "#1E5E6B", "#1B2A4E"])
B_GREY = "0.62"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})
GRID = np.array(sorted(set(range(0, 201, 5)) | {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(5000), 60)))}), float)


def onto(st, y):
    st = np.asarray(st, float); y = np.asarray(y, float)
    return y if len(st) == len(GRID) and np.allclose(st, GRID) else np.interp(GRID, st, y)


def load_sw(fname, key="value"):
    d = json.load(open(os.path.join(HERE, fname)))
    out = {}
    for r in d.values():
        if r["norm"] != 1:
            continue
        st = [x["step"] for x in r["rows"]]
        out.setdefault(float(r[key]), []).append((onto(st, [x["A"] for x in r["rows"]]),
                                                  onto(st, [x["B"] for x in r["rows"]])))
    return out


def figure(data, out, label, fmt, dose=False, legend_loc="lower left"):
    vals = sorted(data)
    cmap = RAMP(np.linspace(0, 1, len(vals)))
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    summary = []
    for i, v in enumerate(vals):
        A = np.array([a for a, _ in data[v]]); mu = A.mean(0)
        ax.plot(GRID, mu, color=cmap[i], lw=1.8, label=f"A ({fmt(v)})")
        tr = int(mu.argmin())
        summary.append((v, len(A), mu[tr], int(GRID[tr]), mu[tr:].max(), mu[-1]))
        if dose:
            B = np.array([b for _, b in data[v]]).mean(0)
            ax.plot(GRID, B, color=B_GREY, ls=":", lw=2.0 if v == 1.0 else 0.9,
                    alpha=1.0 if v == 1.0 else 0.45)
    if not dose:
        B = np.array([b for _, b in sum(data.values(), [])]).mean(0)
        ax.plot(GRID, B, color=B_GREY, ls=":", lw=2.0, label="B")
    else:
        ax.plot([], [], color=B_GREY, ls=":", lw=2.0, label="B (bold: ×1)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=7, loc=legend_loc, handlelength=1.8, labelspacing=0.25)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{out}.{e}"), bbox_inches="tight")
    print(f"\n{out}   ({label})")
    print(f"  {'value':>8} {'seeds':>5} {'A trough':>9} {'@step':>6} {'max after':>10} {'A end':>6}")
    for v, n, tr, ts, mx, en in summary:
        print(f"  {v:>8g} {n:>5} {tr:>9.3f} {ts:>6} {mx:>10.3f} {en:>6.3f}")


SPECS = {
    "d": ("sw_d.json", "key dimension", lambda v: f"d = {int(v)}", False, "center right"),
    "beta": ("sw_beta.json", "shared fraction", lambda v: f"β = {v:g}", False, "center right"),
    "nb": ("sw_nb.json", "number injected", lambda v: f"$n_B$ = {int(v)}", False, "center right"),
    "dose": ("sw_dose.json", "injection rate", lambda v: f"rate ×{v:g}", True, "lower right"),
    "load": ("sw_load.json", "load", lambda v: f"$n_A$ = {int(v)}", False, "center right"),
    "vocab": ("sw_vocab.json", "vocabulary", lambda v: f"|V| = {int(v)}", False, "center right"),
}

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or list(SPECS)
    for k in which:
        f, label, fmt, dose, loc = SPECS[k]
        if os.path.exists(os.path.join(HERE, f)):
            figure(load_sw(f), f"toy_min_sweep_{k}", label, fmt, dose=dose, legend_loc=loc)
        else:
            print(f"skip {k}: {f} not there yet")
