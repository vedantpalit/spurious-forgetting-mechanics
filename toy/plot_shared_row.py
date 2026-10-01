"""The toy companion of the OLMo weight-patch figure: old and new facts, as they are and with
the shared row of the store's update removed at every checkpoint.

    plots/toy_min_shared_row_removed.{png,pdf}

Data: patch_weights.json (patch_weights.py; ten seeds, base configuration, gate-matched).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, OLIVE = "#1B2A4E", "#C4245F", "#8A8C30"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.grid": False})


def main():
    data = json.load(open(os.path.join(HERE, "patch_weights.json")))
    st = np.array([x["step"] for x in data["0"]], float)
    get = lambda key, fb: np.array([[x.get(key, x[fb]) for x in data[s]] for s in data])
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(5.0, 3.55), dpi=200)
    AS_IS, REMOVED = "#1B2A4E", "#8A8C30"
    for Y, col, ls, lw, lab in ((get("A", "A"), AS_IS, "-", 2.4, "as trained"),
                                (get("B", "B"), AS_IS, ":", 2.0, None),
                                (get("murow/A", "A"), REMOVED, "-", 2.4, "shared row of the update removed"),
                                (get("murow/B", "B"), REMOVED, ":", 2.0, None)):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 5000); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("First-token accuracy")
    fig.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=2,
               handlelength=1.8, columnspacing=1.6)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_shared_row_removed.{e}"), bbox_inches="tight")
    for k, fb in (("A", "A"), ("murow/A", "A"), ("B", "B"), ("murow/B", "B")):
        v = get(k, fb).mean(0)
        print(f"{k:>8}: trough {v.min():.3f}@{int(st[v.argmin()])}  at 200 {v[st == 200][0]:.3f}  at 1000 {v[np.argmin(abs(st - 1000))]:.3f}  end {v[-1]:.3f}")
    print("wrote plots/toy_min_shared_row_removed")


if __name__ == "__main__":
    main()
