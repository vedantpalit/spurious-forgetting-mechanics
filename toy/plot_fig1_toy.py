"""Figure 1 in the style of the sparse-attention paper: the task on the left, the phenomenon,
the mechanism and a scaling law on the right, all from the minimal model.

    plots/fig1_toy.{png,pdf}

Left: the task -- keys that share a component, one matrix, one normalization, one readout;
three populations; pretrain on A and D, then train on B alone and watch A.
Right: a. old- and new-fact accuracy; b. old- and new-fact loss (paper_base.json, ten seeds).
panel_b (the mechanism) and panel_c (the timing law) are kept in the file for the record;
they are the four-panel mechanism figure and the laws figure of Section 3.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE, DEEP, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "#1E5E6B", "0.55"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9.5,
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, xy, w, h, text, fc, ec="none", fs=9, tc="white", bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color="0.3", lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, color=color, lw=lw, arrowstyle=style, mutation_scale=9))


def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    # row 1: keys share a component
    ox, oy = 0.55, 5.35
    arrow(ax, (ox, oy), (ox + 1.5, oy), color=GREY, lw=2.0)
    ax.text(ox + 1.62, oy, r"$\mu$", fontsize=9, color="0.35", va="center")
    for ang, col in ((30, NAVY), (-24, OLIVE), (12, CRIM)):
        a = np.deg2rad(ang)
        arrow(ax, (ox, oy), (ox + 1.45 * np.cos(a), oy + 1.45 * np.sin(a)), color=col, lw=1.3)
    ax.text(3.1, 5.6, r"$k_i = \sqrt{\beta}\,\mu + \sqrt{1-\beta}\,g_i$", fontsize=9.5, va="center")
    ax.text(3.1, 5.05, r"every key shares $\mu$; $\beta$ sets how related the facts are",
            fontsize=7.5, color="0.35", va="center")
    # row 2: the model
    y = 3.35; hh = 0.78
    box(ax, (0.1, y), 1.0, hh, r"key $k$", "0.75", fs=8)
    box(ax, (1.85, y), 1.0, hh, r"$W$", NAVY, fs=9.5, bold=True)
    box(ax, (3.6, y), 1.35, hh, r"rms$(\cdot)$", TEAL, fs=9, bold=True)
    box(ax, (5.7, y), 1.0, hh, r"$U$", NAVY, fs=9.5, bold=True)
    box(ax, (7.45, y), 1.9, hh, "softmax", "0.75", fs=8)
    for x0, x1 in ((1.1, 1.85), (2.85, 3.6), (4.95, 5.7), (6.7, 7.45)):
        arrow(ax, (x0, y + hh / 2), (x1, y + hh / 2))
    ax.text(2.35, y - 0.22, "store", fontsize=7, color="0.35", ha="center", va="top")
    ax.text(4.27, y - 0.22, "normalization", fontsize=7, color="0.35", ha="center", va="top")
    ax.text(6.2, y - 0.22, "readout", fontsize=7, color="0.35", ha="center", va="top")
    # row 3: populations
    y2 = 0.95; hh2 = 1.05
    box(ax, (0.1, y2), 2.85, hh2, "$A$: old facts\nvalues in half X", NAVY, fs=7.5)
    box(ax, (3.35, y2), 2.85, hh2, "$D$: old facts\nvalues in half Y", OLIVE, fs=7.5)
    box(ax, (6.6, y2), 2.85, hh2, "$B$: new facts\nvalues in half Y", CRIM, fs=7.5)
    ax.text(0.1, 0.3, r"pretrain on $A \cup D$; then train on $B$ alone and watch $A$",
            fontsize=8, color="0.2", va="center")


def panel_a(ax):
    d = json.load(open(os.path.join(HERE, "paper_base.json")))
    runs = [r["rows"] for r in d.values() if r["norm"] == 1]
    st = np.array([x["step"] for x in runs[0]], float)
    A = np.array([[x["A"] for x in r] for r in runs]); B = np.array([[x["B"] for x in r] for r in runs])
    for Y, col, ls, lab in ((A, NAVY, "-", "old facts"), (B, CRIM, ":", "new facts")):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 5000); ax.set_ylim(0, 1.03)
    ax.set_xlabel("injection step"); ax.set_ylabel("accuracy")
    m = A.mean(0); tr = int(m.argmin())
    ax.annotate("suppression", (st[tr], m[tr]), xytext=(2.2, 0.06), fontsize=7.5, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.annotate("recovery", (250, 0.5), xytext=(60, 0.72), fontsize=7.5, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.annotate("erosion", (3000, m[st == 3000][0] if (st == 3000).any() else 0.53), xytext=(700, 0.30),
                fontsize=7.5, color="0.3", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.78), handlelength=1.6)


def panel_b(ax):
    tr = {s: json.load(open(os.path.join(HERE, f"track_m{s}.json")))[f"n1-s{s}"] for s in range(10)}
    st = np.array([x["step"] for x in tr[0]["rows"]], float); keep = st <= 400; x = st[keep]
    grab = lambda k: np.array([[r[k] for r in tr[s]["rows"]] for s in tr])[:, keep]
    S, M = grab("s_on_w"), grab("margin")
    mM = M.mean(0); cm = next((x[i] for i in range(1, len(x)) if mM[i - 1] < 0 <= mM[i]), None)
    ax.plot(x, S.mean(0), color=TEAL, lw=1.8, label="the shift, $S$")
    ax.fill_between(x, S.mean(0) - S.std(0, ddof=1), S.mean(0) + S.std(0, ddof=1), color=TEAL, alpha=0.15, lw=0)
    ax.set_ylabel("shift $S$ toward the new facts", color=TEAL)
    ax.set_ylim(0, S.mean(0).max() * 1.25); ax.set_xlim(0, 400); ax.set_xlabel("injection step")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(x, mM, color=CRIM, lw=1.6, ls="--", label="new facts' confidence, $\\bar M$")
    ax2.axhline(0, color="0.6", lw=0.7)
    ax2.set_ylabel("new facts' confidence $\\bar M$", color=CRIM); ax2.set_ylim(-9.5, 3)
    if cm is not None:
        ax.axvline(cm, color="0.35", lw=0.9, ls=":")
        ax.text(cm + 8, ax.get_ylim()[1] * 0.93, "new facts\nbecome confident", fontsize=7, color="0.35", va="top")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="lower right", handlelength=1.6)


def panel_loss(ax):
    d = json.load(open(os.path.join(HERE, "paper_base.json")))
    runs = [r["rows"] for r in d.values() if r["norm"] == 1]
    st = np.array([x["step"] for x in runs[0]], float)
    A = np.array([[x["A_loss"] for x in r] for r in runs]); B = np.array([[x["B_loss"] for x in r] for r in runs])
    for Y, col, ls, lab in ((A, NAVY, "-", "old facts"), (B, CRIM, ":", "new facts")):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 5000)
    ax.set_xlabel("injection step"); ax.set_ylabel("loss (nats)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.6)


def panel_c(ax):
    d = json.load(open(os.path.join(HERE, "sw_dose.json")))
    pts = {}
    for r in d.values():
        if r["norm"] != 1:
            continue
        st = np.array([z["step"] for z in r["rows"]], float); A = np.array([z["A"] for z in r["rows"]])
        tr = int(A.argmin()); mx = A[tr:].max()
        rec = tr + int(np.argmax(A[tr:] >= A[tr] + 0.9 * (mx - A[tr])))
        pts.setdefault(r["value"], []).append((st[tr], st[rec]))
    xs = np.array(sorted(pts))
    for j, (col, lab, mk) in enumerate(((NAVY, "trough", "o"), (TEAL, "recovery", "s"))):
        ys = np.array([np.mean([p[j] for p in pts[v]]) for v in xs])
        a, b = np.polyfit(np.log(xs), np.log(ys), 1)
        xx = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 30)
        ax.plot(xx, np.exp(b) * xx ** a, color=col, lw=1.0, ls="--")
        for v in xs:
            ax.scatter([v] * len(pts[v]), [p[j] for p in pts[v]], color=col, s=14, marker=mk, zorder=3)
        ax.scatter([], [], color=col, s=14, marker=mk, label=f"{lab}: $t \\propto \\eta^{{{a:.2f}}}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("injection rate $\\eta$ (relative)"); ax.set_ylabel("step")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.4)


def main():
    fig = plt.figure(figsize=(11.2, 3.3), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.38,
                          left=0.02, right=0.99, top=0.9, bottom=0.2)
    axL = fig.add_subplot(gs[0]); schematic(axL)
    axA = fig.add_subplot(gs[1]); panel_a(axA)
    axB = fig.add_subplot(gs[2]); panel_loss(axB)
    for ax, lab in ((axA, "a."), (axB, "b.")):
        ax.text(-0.2, 1.04, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig1_toy.{e}"), bbox_inches="tight")
    print("wrote plots/fig1_toy")


if __name__ == "__main__":
    main()
