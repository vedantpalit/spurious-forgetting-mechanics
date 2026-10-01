"""Figure 2: the minimal model and its three ingredients.

    plots/fig2.{png,pdf}

Left: the model, with its three ingredients numbered -- (1) a component shared by all keys,
(2) the normalization before the readout, (3) the output tokens split into two regions.
Then one panel per ingredient, the model against its removal, in the schematic's order:
    (1) key sharing       alpha from 0 to 0.5 (gate-matched sweep, sw_beta.json)
    (2) normalization     rms (the model) against ReLU and none (ingredients.json)
    (3) output tokens     new answers in one half of the tokens (the model) against all (ingredients.json)
Old facts only; mean probability of the correct value; mean over
seeds, band +/- 1 sd. The new facts are learned in every arm (gate-matched).

Notation as in paper.tex (alpha for the shared share of the key, e_i, W_1, W_2); the data files
keep the name beta. Style as plot_fig3.py: exactly 397 pt wide, serif, no minor ticks.

Run: JAX_PLATFORMS=cpu uv run python plot_fig2.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, OLIVE, TEAL = "#1B2A4E", "#C4245F", "#8A8C30", "#2F8C7D"
XMAX = 500
KEY_Y, GAP, DROP = 9.3, 4.8, 0.35                     # key formula height (schematic units); title-legend gap,
                                                      # titles' drop below the formula's center (pt)
FIG_WIDTH = 397 / 72                                  # the paper's text width; saved at exactly this size


def style():
    r = matplotlib.rcParams
    r["lines.linewidth"] = 1; r["lines.markersize"] = 3
    r["font.family"] = "serif"; r["font.serif"] = ["DejaVu Serif"]; r["mathtext.fontset"] = "dejavuserif"
    r["legend.fontsize"] = 5; r["legend.title_fontsize"] = 6; r["axes.labelsize"] = 6
    r["xtick.labelsize"] = 5; r["ytick.labelsize"] = 5
    r["axes.spines.top"] = False; r["axes.spines.right"] = False
    r["figure.dpi"] = 250; r["axes.linewidth"] = 0.5
    r["xtick.major.size"] = 2; r["xtick.major.width"] = 0.5
    r["ytick.major.size"] = 2; r["ytick.major.width"] = 0.5
    r["pdf.fonttype"] = 42; r["ps.fonttype"] = 42


def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    num = dict(fontsize=5, fontweight="bold", color="white", ha="center", va="center", zorder=6)

    def badge(x, y, n):
        ax.plot(x, y, "o", ms=6.5, color="0.3", zorder=5); ax.text(x, y, n, **num)

    def box(x, y, w, h, text, fc="0.9", ec="none", tc="0.15", fs=6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=0.5, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

    def trapeze(y_top, w_top, y_bot, w_bot, text, x=5.6, fc="0.9", ec="none", tc="0.15", fs=6):
        ax.add_patch(Polygon([(x - w_top / 2, y_top), (x + w_top / 2, y_top), (x + w_bot / 2, y_bot),
                              (x - w_bot / 2, y_bot)], closed=True, fc=fc, ec=ec, lw=0.5, zorder=3))
        ax.text(x, (y_top + y_bot) / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    # keys: shared + own
    # "shared" and "own" centered under their terms, measured on the rendered formula
    parts = (r"key  $k_i = $", r"$\sqrt{\alpha}\,\mu$", r"$\;+\;$", r"$\sqrt{1-\alpha}\,e_i$")
    kw = dict(va="center", fontsize=6)
    r = ax.figure.canvas.get_renderer(); inv = ax.transData.inverted()
    widths = []
    for p in parts:
        t = ax.text(0, 0, p, **kw); bb = t.get_window_extent(r); t.remove()
        widths.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
    x = 5.6 - sum(widths) / 2; centers = []
    for p, w in zip(parts, widths):
        ax.text(x, KEY_Y, p, ha="left", **kw); centers.append(x + w / 2); x += w
    ax.text(centers[1], 8.55, "shared", ha="center", va="center", fontsize=5, color="0.35")
    ax.text(centers[3], 8.55, "own", ha="center", va="center", fontsize=5, color="0.35")
    badge(0.55, KEY_Y, "1")
    # the network: W_1 narrows onto the RMS block, W_2 widens onto the answer values
    trapeze(8.05, 5.0, 6.0, 3.0, "$W_1$")
    ax.text(7.45, 5.8, "$h_i$", ha="left", va="center", fontsize=6, color="0.3")
    box(5.6, 5.2, 3.0, 0.9, "RMS")
    badge(0.55, 5.2, "2")
    trapeze(4.4, 3.0, 2.45, 5.0, "$W_2$")
    # answer values: one dot per value, the two regions in two colors, as wide as W_2's base
    xs = np.linspace(5.6 - 2.3, 5.6 + 2.3, 12); y = 1.85
    for k, xx in enumerate(xs):
        ax.plot(xx + (0.15 if k >= 6 else -0.15), y, "o", ms=3.2, mew=0,
                color="#9AA3BD" if k < 6 else "#8FC4BA", zorder=3)
    ax.text(5.6, 1.15, "output tokens, two regions", ha="center", va="center", fontsize=5, color="0.35")
    badge(0.55, y, "3")


def curves(ax, d, tag, key, col, lab, lw=1.0, fill=True, ls="-"):
    rs = [v for v in d.values() if v["tag"] == tag]
    st = np.array([x["step"] for x in rs[0]["rows"]], float); keep = st <= st[st >= XMAX].min()   # one point past XMAX, clipped by the axis
    Y = np.array([[x[key] for x in r["rows"]] for r in rs])[:, keep]; st = st[keep]
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    if fill:
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
    ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
    return st, m


def main():
    style()
    ing = json.load(open(os.path.join(HERE, "ingredients.json")))
    swb = json.load(open(os.path.join(HERE, "sw_beta.json")))
    fig = plt.figure(figsize=(FIG_WIDTH, 1.75))
    gs = fig.add_gridspec(1, 3, wspace=0.16, left=0.32, right=0.985, top=0.8, bottom=0.178)
    ax0 = fig.add_axes([0.0, -0.06, 0.25, 1.0]); schematic(ax0)     # the full height, on the left
    axk, axn, axv = (fig.add_subplot(gs[k]) for k in (0, 1, 2))
    axn.sharey(axk); axv.sharey(axk)
    # (1) shared keys: the beta range
    betas = sorted(b for b in {r["beta"] for r in swb.values() if r["norm"] == 1} if b <= 0.5)
    cmap = plt.get_cmap("Blues")
    for i, b in enumerate(betas):
        rs = [r for r in swb.values() if r["norm"] == 1 and r["beta"] == b]
        st = np.array([x["step"] for x in rs[0]["rows"]], float); keep = st <= st[st >= XMAX].min()   # one point past XMAX, clipped by the axis
        A = np.array([[x["A_p"] for x in r["rows"]] for r in rs]).mean(0)
        col = cmap(0.45 + 0.5 * i / max(len(betas) - 1, 1))     # one gradient; the model is its end
        axk.plot(st[keep], A[keep], color=col, zorder=3,
                 label=r"$\alpha = %g$" % b if i == 0 else "%g" % b)
    # (2) normalization
    for tag, col, lab in (("model", NAVY, "RMS"), ("relu", CRIM, "ReLU"),
                          ("linear", OLIVE, "none")):
        curves(axn, ing, tag, "A_p", col, lab)
    # (3) answer regions
    for tag, col, lab in (("model", NAVY, "half"), ("allvalues", CRIM, "all")):
        curves(axv, ing, tag, "A_p", col, lab)
    panels = ((axk, "1", "key sharing", 4), (axn, "2", "normalization", 3), (axv, "3", "output tokens", 2))
    for ax, n, title, ncol in panels:
        ax.set_xlim(0, XMAX); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("finetuning step")
        ax.set_xticks(range(0, XMAX + 1, 100))
    # titles at the height of the key formula, legends GAP under them, plots 1 pt under the legends
    pt = 1 / (fig.get_figheight() * 72)
    y_key = fig.transFigure.inverted().transform(ax0.transData.transform((0, KEY_Y)))[1] - DROP * pt
    titles = []
    for ax, n, title, ncol in panels:
        b = ax.get_position()
        titles.append(fig.text((b.x0 + b.x1) / 2, y_key, f"({n})  {title}", ha="center", va="center",
                               fontsize=matplotlib.rcParams["legend.title_fontsize"]))
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    y_leg = min(t.get_window_extent(r).y0 for t in titles) / fig.bbox.height - GAP * pt
    for ax, n, title, ncol in panels:
        b = ax.get_position()
        h, l = ax.get_legend_handles_labels()
        order = [i for c in range(ncol) for i in range(c, len(h), ncol)]   # read row by row
        ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc="upper center",
                  bbox_to_anchor=((b.x0 + b.x1) / 2, y_leg), bbox_transform=fig.transFigure, ncol=ncol,
                  handlelength=1.0, handletextpad=0.4, labelspacing=0.25, columnspacing=0.8, borderaxespad=0)
    fig.canvas.draw()
    gs.update(top=min(ax.get_legend().get_window_extent(r).y0 for ax, *_ in panels) / fig.bbox.height - pt)
    axk.set_ylabel("old facts accuracy")
    for ax in (axn, axv):
        plt.setp(ax.get_yticklabels(), visible=False)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig2.{e}"))
    print("wrote plots/fig2")


if __name__ == "__main__":
    main()
