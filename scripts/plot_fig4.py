"""Figure 4: the mechanism in a trained transformer (attention-only, the model of Figure 1).

    plots/fig4.{png,pdf}

Left: the old facts' readout-state displacement split into its common part delta (their mean,
within attribute) and the individual remainder eps_a -- ||delta|| rises and recedes, rms ||eps_a||
keeps growing. Right: old-fact accuracy through three readouts at the same checkpoints --
trained; the common shift only (h_a(0) + delta_t); the common shift removed (h_a(t) - delta_t).
Removing delta removes suppression; delta alone produces the suppression-recovery trajectory.

The 10-step-resolution run (FIG_SCHEDULE, `-fig` files) when present, else the coarse grid.
Same data as scripts/plot_delta_eps.py (residual_split/) and scripts/plot_shift_counterfactual.py
(shift_decomposition/); three seeds, mean and range. Linear steps, Figure 3's colours.

Run: uv run python -m scripts.plot_fig4
"""
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, OLIVE = "#1B2A4E", "#2F8C7D", "#8A8C30"
XMAX = 500


def split():
    fs = (sorted(glob.glob("residual_split/mlp_free-fig-p16000-disjoint-t1200-seed*.npz"))
          or sorted(glob.glob("residual_split/mlp_free-p16000-disjoint-t1200-seed*.npz")))
    st = np.load(fs[0], allow_pickle=True)["steps"].astype(float)
    D = np.stack([np.linalg.norm(np.load(f, allow_pickle=True)["delta"], axis=-1).mean(-1) for f in fs])
    E = np.stack([np.load(f, allow_pickle=True)["eps_rms"].mean(-1) for f in fs])
    return st, D, E


def counterfactual():
    fs = (sorted(glob.glob("shift_decomposition/mlp_free-fig-p16000-disjoint-t1200-seed*.npz"))
          or sorted(glob.glob("shift_decomposition/mlp_free-p16000-disjoint-t1200-seed*.npz")))
    ds = [np.load(f) for f in fs]
    st = ds[0]["steps"].astype(float)
    get = lambda k: np.stack([d[k].astype(np.float64).mean((1, 2)) for d in ds])
    return st, get("acc_trained"), get("acc_const"), get("acc_vary")


FIG_WIDTH = 397 / 72                                  # the paper's text width, as Figures 1-3


def set_default_style():
    """Figures 1-3's style (toy/plot_fig3.py)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "lines.linewidth": 1, "lines.markersize": 3, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "legend.fontsize": 5, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.title_fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 250,
        "mathtext.fontset": "dejavuserif", "xtick.major.size": 2, "xtick.major.width": 0.5,
        "ytick.major.size": 2, "ytick.major.width": 0.5, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42})


CRIM = "#C4245F"


def _light(c, a):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c); return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


# Real individuals of the Figure 1 population (build() of the mlp_free t1200 disjoint run, seed 0):
# A = first old set (answers in region 1), B = second old set (region 2), B' = first new person
# (answers in region 2). Same birthplace template for all three, as in the data.
ROWS = [("old", "Sarah Hossein Esposito", "Boston", NAVY),
        ("old", "Emeka Elena Choi", "Antwerp", CRIM),
        ("new", "Pablo Irene Carvalho", "Nagoya", _light(CRIM, 0.55))]
TEMPLATE = " originally hails from the city of "


def sentence(ax, x, y, parts, fs):
    """One line of mixed-style text, packed without gaps, anchored at (x, y) in axes coords."""
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    kids = [TextArea(t, textprops=dict(color=c, fontsize=fs, fontweight=w, fontstyle=st))
            for t, c, w, st in parts]
    box = HPacker(children=kids, align="baseline", pad=0, sep=0)
    ax.add_artist(AnchoredOffsetbox(loc="center left", child=box, pad=0, borderpad=0, frameon=False,
                                    bbox_to_anchor=(x, y), bbox_transform=ax.transAxes))


def schematic(ax):
    """The data behind the transformer as a grid: columns are the two answer regions, rows are
    pretraining and finetuning. The new individual (B') sits in B's column: a different person,
    an answer from the same region. Set labels as in Figure 1."""
    from matplotlib.patches import FancyBboxPatch
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fs = 4.0
    X = (0.0, 0.515); W = 0.485

    def box(x, y, w, h, ec, fc="white", lw=0.6, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    ec=ec, fc=fc, lw=lw, ls=ls, transform=ax.transAxes, zorder=1))

    def card(x, y, tag, name, val, col):
        box(x, y, W, 0.27, col, lw=0.7)
        ax.text(x + W - 0.025, y + 0.235, tag, color=col, fontsize=4.8, fontweight="bold",
                ha="right", va="center", transform=ax.transAxes)
        sentence(ax, x + 0.025, y + 0.18, [("The hometown of", "0.45", "normal", "normal")], fs)
        sentence(ax, x + 0.025, y + 0.118, [(name, "0.15", "normal", "italic")], 3.8)
        sentence(ax, x + 0.025, y + 0.055, [("is ", "0.45", "normal", "normal"), (val, col, "bold", "normal"),
                                           (".", "0.45", "normal", "normal")], fs)

    (_, n1, v1, c1), (_, n2, v2, c2), (_, n3, v3, c3) = ROWS
    # column headers: the two answer regions (real values of the birthplace split)
    for x, col, t, ex in ((X[0], NAVY, "answer region 1", "Boston, Lagos, …"),
                          (X[1], CRIM, "answer region 2", "Antwerp, Nagoya, …")):
        box(x, 0.855, W, 0.14, col, fc=_light(col, 0.1), lw=0.5)
        ax.text(x + W / 2, 0.955, t, color=col, fontsize=4.6, ha="center", va="center", transform=ax.transAxes)
        ax.text(x + W / 2, 0.89, ex, color=col, fontsize=4.0, ha="center", va="center", transform=ax.transAxes)
    # rows: pretraining (one old set per region), finetuning (new individuals, region 2 only)
    lab = dict(fontsize=4.8, color="0.35", va="center", transform=ax.transAxes)
    # the same spacing above each row label (from the header, from the pretraining cards)
    # and the same spacing between each label and its cards
    G, L, H = 0.07, 0.04, 0.27
    y_pre = 0.855 - G; y_pre_card = y_pre - L - H
    y_ft = y_pre_card - G; y_ft_card = y_ft - L - H
    ax.text(X[0], y_pre, "pretraining", **lab)
    card(X[0], y_pre_card, "A", n1, v1, c1)
    card(X[1], y_pre_card, "B", n2, v2, c2)
    ax.text(X[0], y_ft, "finetuning", **lab)
    box(X[0], y_ft_card, W, H, "0.75", fc="white", lw=0.5, ls=(0, (2, 2)))
    ax.text(X[0] + W / 2, y_ft_card + H / 2, "no new answers\nin this region", color="0.55", fontsize=4.0,
            ha="center", va="center", transform=ax.transAxes, linespacing=1.3)
    card(X[1], y_ft_card, "B′", n3, v3, c3)
    ax.text(X[1] + W / 2, y_ft_card - 0.055, "new individual", color="0.45", fontsize=4.0, style="italic",
            ha="center", va="center", transform=ax.transAxes)


def line(ax, st, Y, col, ls, lab):
    keep = st <= st[st >= XMAX].min()                  # one point past the edge, clipped by the axis
    st, Y = st[keep], Y[:, keep]
    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
    ax.plot(st, mu, color=col, ls=ls, label=lab, zorder=3)


def main():
    set_default_style()
    fig = plt.figure(figsize=(FIG_WIDTH, 115.2 / 72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.40)
    axs = fig.add_subplot(gs[0]); axl = fig.add_subplot(gs[1]); axr = fig.add_subplot(gs[2])
    schematic(axs)
    st, D, E = split()
    line(axl, st, D, TEAL, "-", r"common shift $\|\delta\|$")
    line(axl, st, E, OLIVE, "-", r"individual drift $\|\varepsilon_a\|$ (rms)")
    axl.set_ylim(0, None); axl.set_ylabel("displacement of old-fact states")
    s2, T, C, V = counterfactual()
    line(axr, s2, T, NAVY, "-", "trained")
    line(axr, s2, C, TEAL, "--", "common shift only")
    line(axr, s2, V, OLIVE, "-", "common shift removed")
    axr.set_ylim(-0.03, 1.03); axr.set_ylabel("old-fact accuracy")
    tr = s2[int(T.mean(0).argmin())]
    for ax in (axl, axr):
        ax.axvline(tr, color="0.55", ls=":", lw=0.8, zorder=1)
        ax.set_xlim(0, XMAX); ax.set_xlabel("finetuning step")
    axl.legend(frameon=False, loc="lower right")
    axr.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.01, right=0.985, bottom=0.2, top=0.98)
    # the data panel has no x-axis labels: let it use the space below the plots' axes, so it is
    # not top-heavy, while keeping its top level with the other panels
    p = axs.get_position(); axs.set_position([p.x0, 0.09, p.width, p.y1 - 0.09])
    for e in ("png", "pdf"):
        fig.savefig(f"plots/fig4.{e}")          # exactly 397 pt wide, no tight crop
    d, e = D.mean(0), E.mean(0); k = int(d.argmax())
    cross = st[int(np.argmax(e > d))] if (e > d).any() else None
    t, c, v = T.mean(0), C.mean(0), V.mean(0); i = int(t.argmin())
    print(f"split: ||delta|| peak {d[k]:.1f}@{st[k]:.0f}, end {d[-1]:.1f}; eps {e[0]:.1f} -> {e[-1]:.1f}; first eps > delta at {cross}")
    print(f"readouts at the trough (step {s2[i]:.0f}): trained {t[i]:.3f}, shift only {c[i]:.3f}, removed {v[i]:.3f}; "
          f"at {s2[-1]:.0f}: {t[-1]:.3f} / {c[-1]:.3f} / {v[-1]:.3f}")
    print("wrote plots/fig4")


if __name__ == "__main__":
    main()
