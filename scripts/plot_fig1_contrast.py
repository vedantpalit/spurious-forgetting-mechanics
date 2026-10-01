"""Figure 1's two data panels: ordinary forgetting against the structured case.

    plots/tf_allvalues.{png,pdf}   new facts' answers drawn from the whole pool: the old facts
                                   decline slowly, both populations alike, no trough
    plots/tf_oneregion.{png,pdf}   new facts' answers in one half of the pool: the old facts
                                   whose answers sit in the OTHER half crash and return; those
                                   whose answers the new facts use are untouched. The three
                                   phases are marked on the crashing curve.
    plots/fig1_setup.{png,pdf}     the middle panel: one attribute's answer values split into
                                   two regions; old facts point into each; the new facts'
                                   answers either anywhere (-> left panel) or in one region
                                   (-> right panel)

Same model (8 layers, 512 wide), same pretrained checkpoint (16,000 steps), same injection
rate and steps; only where the new facts' answers sit differs. Old facts solid, new facts
dotted; mean and +/-1 sd over seeds (3 all-values, 5 one-region).

Run: uv run python -m scripts.plot_fig1_contrast
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, TEAL, MARK = "#1B2A4E", "#C4245F", "#2F8C7D", "#7A7A7A"
STEP = re.compile(r"\[step (\d+)\]"); FIELD = re.compile(r"(\w+)=([-\d.]+)")
PRETRAIN, XMAX = 16000, 1000


def parse(path):
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = STEP.search(line)
        if not m:
            continue
        rec = {}
        for pop, tag in (("A", "dataA:"), ("B", "dataB:"), ("Bal", "ballast:")):
            i = line.find(tag)
            if i >= 0:
                rec[pop] = {k: float(v) for k, v in FIELD.findall(line[i + len(tag):].split(",")[0])}
        rows[int(m.group(1)) - PRETRAIN] = rec
    return rows


def load(cond):
    """Both data panels come from the same model, so the only difference between them is where
    the new facts' answers lie. SOURCE selects which model: the minimal associative memory
    (probability of the correct value, ten seeds) or the 8-layer transformer (accuracy)."""
    if SOURCE == "toy":
        import json
        tag = {"all_values": "allvalues", "disjoint": "model"}[cond]
        rs = [v for v in json.load(open("toy/fig1_toy.json")).values() if v["tag"] == tag]
        steps = np.array([x["step"] for x in rs[0]["rows"]]); keep = steps <= XMAX
        get = lambda k: np.array([[x[k] for x in r["rows"]] for r in rs])[:, keep]
        return steps[keep], get("A_p"), get("D_p"), get("B_p"), len(rs)
    d, pre = {"transformer": ("logs_scale8", "scale8"), "attention": ("logs_mlpfree", "mlpfree")}[SOURCE]
    files = sorted(glob.glob(f"{d}/{pre}_injection.p{PRETRAIN}.{cond}.seed*.out"))
    if not files:    # the 1200-step runs carry the run length in the name (constant LR: same curve)
        files = sorted(glob.glob(f"{d}/{pre}_injection.p{PRETRAIN}.{cond}.t1200.seed*.out"))
    if not files:
        return None
    runs = [parse(f) for f in files]
    steps = np.array(sorted(set.intersection(*[set(r) for r in runs])))
    keep = steps <= XMAX; steps = steps[keep]
    get = lambda pop: np.array([[r[s][pop]["first_acc"] for s in steps] for r in runs])
    return steps, get("A"), get("Bal"), get("B"), len(runs)


SOURCE = "attention"    # "toy" | "transformer" (8 layers, MLPs) | "attention" (attention-only)


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})


def panel(name, cond, lab_a, lab_bal, phases=False, pool=False, ax=None):
    got = load(cond)
    if got is None:
        print(f"{name}: no {SOURCE} logs for condition {cond} -- not drawn (left as it was)")
        return
    st, A, Bal, B, n = got
    if pool:     # both old populations, one curve: nothing separates them in this condition
        A = (A + Bal) / 2
    own = ax is None
    if own:
        style()
        fig, ax = plt.subplots(figsize=(5.0, 3.55), dpi=200)
    curves = ((Bal, TEAL, "-", 2.4, lab_bal, 3), (A, NAVY, "-", 2.4, lab_a, 4),
              (B, CRIM, ":", 2.0, "new facts", 3))
    if pool:
        curves = curves[1:]
    for Y, col, ls, lw, lab, z in curves:
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=z)
    h, l = ax.get_legend_handles_labels(); order = [0, 1] if pool else [1, 0, 2]
    h, l = [h[i] for i in order], [l[i] for i in order]
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Fine-tuning step")
    ax.set_ylabel("P(correct answer)" if SOURCE == "toy" else "First-token accuracy")
    if phases:
        a = A.mean(0); i = int(a.argmin()); j = i + int(a[i:].argmax())
        kw = dict(color=MARK, style="italic", fontsize=10.5,
                  arrowprops=dict(arrowstyle="-", color=MARK, lw=0.9, shrinkA=2, shrinkB=3))
        ax.annotate("suppression", xy=(st[i], a[i]), xytext=(st[i] + 120, a[i] - 0.07),
                    ha="left", va="center", **kw)
        k = i + int(np.argmin(abs(a[i:j] - (a[i] + a[j]) / 2)))
        ax.annotate("recovery", xy=(st[k], a[k]), xytext=(st[k] + 130, a[k] - 0.09),
                    ha="left", va="center", **kw)
        e = int(np.argmin(abs(st - 650)))
        if a[j] - a[-1] > 0.05:          # label erosion only if the curve visibly declines
            ax.annotate("erosion", xy=(st[e], a[e]), xytext=(st[e] + 60, a[e] + 0.15),
                        ha="center", va="center", **kw)
    if own:
        fig.legend(h, l, frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.92),
                   ncol=3, handlelength=1.8, columnspacing=1.4)
        for e in ("png", "pdf"):
            fig.savefig(f"plots/{name}.{e}", bbox_inches="tight")
    else:
        ax.legend(h, l, frameon=False, fontsize=9.5, loc="lower right", handlelength=1.6,
                  borderaxespad=0.3, labelspacing=0.3)
    a, bal, b = A.mean(0), Bal.mean(0), B.mean(0); i = a.argmin()
    print(f"{name}: {n} seeds | old facts min {a[i]:.3f}@{st[i]} -> max after {a[i:].max():.3f}, "
          f"at 1000 {a[-1]:.3f} | other population at that step {bal[i]:.3f}, at 1000 {bal[-1]:.3f} "
          f"| new facts at 50 {b[np.argmin(abs(st - 50))]:.3f}, at 200 {b[np.argmin(abs(st - 200))]:.3f}")


def main():
    os.makedirs("plots", exist_ok=True)
    panel("tf_allvalues", "all_values", "old facts", None, pool=True)
    panel("tf_oneregion", "disjoint", "old facts, other region", "old facts, same region",
          phases=True)
    setup()
    print("wrote plots/tf_allvalues, plots/tf_oneregion, plots/fig1_setup")
    combined()


def setup(ax=None, k=1.0):
    """The middle panel, in two parts. Top: the protocol -- the model already knows old facts
    (two groups, navy and teal) and is fine-tuned on new facts only. Bottom: one attribute's
    answer values in two regions; the old facts' answers lie in both; the new facts' answers
    come either from both regions (dashed, -> left panel) or all from one (solid, -> right)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    CL = np.array([[0, 0], [0.42, 0.18], [-0.38, 0.3], [0.1, -0.4], [-0.3, -0.25], [0.35, -0.3]])
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.3, 3.6), dpi=200)
    ax.set_xlim(-0.4, 10.4); ax.set_ylim(0.3, 10); ax.axis("off")
    note = dict(fontsize=7.2 * k, color="0.4", linespacing=1.15)

    def cluster(c, col, scale=0.75):
        ax.plot(CL[:, 0] * scale + c[0], CL[:, 1] * scale + c[1], "o", color=col, ms=5.0 * k, zorder=3)

    def arrow(p, q, col, lw=1.3, alpha=1.0):
        ax.add_patch(FancyArrowPatch(p, q, color=col, lw=lw, alpha=alpha, arrowstyle="-|>",
                                     mutation_scale=9 * k, zorder=2, shrinkA=2, shrinkB=2))
    # --- the protocol ------------------------------------------------------------------------------
    mx, my, mw, mh = 6.4, 7.2, 2.4, 1.0
    ax.add_patch(FancyBboxPatch((mx - mw / 2, my - mh / 2), mw, mh,
                                boxstyle="round,pad=0.02,rounding_size=0.1", fc="0.45", ec="none", zorder=3))
    ax.text(mx, my, "model", ha="center", va="center", fontsize=9.5 * k, color="white", zorder=4)
    cluster((mx - 0.65, 9.35), NAVY); cluster((mx + 0.65, 9.35), TEAL)
    ax.text(mx - 1.35, 9.35, "old facts", ha="right", va="center", fontsize=9.5 * k, color=NAVY)
    arrow((mx, 8.85), (mx, my + mh / 2), "0.45", lw=1.0, alpha=0.7)
    ax.text(mx + 0.2, 8.25, "already known", ha="left", va="center", **note)
    cluster((1.9, my), CRIM)
    ax.text(1.9, my + 0.6, "new facts", ha="center", va="bottom", fontsize=9.5 * k, color=CRIM)
    arrow((2.6, my), (mx - mw / 2, my), CRIM, lw=1.5)
    ax.text(1.9, my - 0.6, "fine-tuned on these only,\nno replay of old facts", ha="center", va="top", **note)
    # --- where the answers lie -----------------------------------------------------------------------
    x0, x1, xm, sy, sh = 0.9, 9.5, 5.2, 3.75, 0.72
    ax.text((x0 + x1) / 2, sy + sh + 0.15, "answer values of one attribute (e.g. cities)",
            ha="center", va="bottom", **note)
    ax.add_patch(Rectangle((x0, sy), xm - x0, sh, fc="#E6E8EF", ec="none", zorder=1))
    ax.add_patch(Rectangle((xm, sy), x1 - xm, sh, fc="#E1EFEB", ec="none", zorder=1))
    ax.add_patch(Rectangle((x0, sy), x1 - x0, sh, fc="none", ec="0.55", lw=0.8, zorder=2))
    ax.plot([xm, xm], [sy, sy + sh], color="0.55", lw=0.8, zorder=2)
    for xs, col in ((np.linspace(x0 + 0.5, xm - 0.5, 5), NAVY), (np.linspace(xm + 0.5, x1 - 0.5, 5), TEAL)):
        ax.plot(xs, np.full_like(xs, sy + sh / 2), "o", color=col, ms=4.5 * k, zorder=3)

    def bracket(a, b, y, ls):
        ax.plot([a, b], [y, y], color=CRIM, lw=1.6, ls=ls, zorder=3)
        for x in (a, b):
            ax.plot([x, x], [y, y + 0.25], color=CRIM, lw=1.6, zorder=3)
    bracket(x0, x1, 3.1, (0, (3, 2)))
    ax.text((x0 + x1) / 2, 2.75, "new facts: answers from both regions", ha="center", va="top",
            fontsize=8.2 * k, color=CRIM)
    bracket(xm, x1, 1.55, "-")
    ax.text((xm + x1) / 2, 1.2, "new facts: answers\nall from one region", ha="center", va="top",
            fontsize=8.2 * k, color=CRIM, linespacing=1.05)
    for (xa, xb), y in (((x0 - 0.1, -0.35), 3.1), ((x1 + 0.1, 10.35), 1.55)):
        ax.annotate("", xy=(xb, y), xytext=(xa, y), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.3, mutation_scale=10 * k))
    if own:
        for e in ("png", "pdf"):
            fig.savefig(f"plots/fig1_setup.{e}", bbox_inches="tight")


FIG_WIDTH = 397 / 72                                  # the paper's text width; saved at exactly this size
# the sets of the middle panel: A and B pretrained (non-overlapping), then finetuned on C (answers
# anywhere, left panel) or B' (answers in B's region, right panel). In the logs A is dataA, B the
# ballast and C / B' the injected population.
# the paper's palette: navy (A), crimson (B, as in Figure 2), grey (C, as the shared margin of
# Figure 3); B' is a lighter crimson, the same family as the set B it overlaps
_light = lambda c, t: matplotlib.colors.to_hex(t * np.array(matplotlib.colors.to_rgb(c)) + (1 - t))
COL = {"A": NAVY, "B": CRIM, "C": "0.55", "B'": _light(CRIM, 0.55)}
FIG1_XMAX = 500
SETS_ARROW_END = 0.9                                  # where the finetuning arrows end (middle panel units)


def paper_style():
    r = matplotlib.rcParams
    r["lines.linewidth"] = 1; r["lines.markersize"] = 3
    r["font.family"] = "serif"; r["font.serif"] = ["DejaVu Serif"]; r["mathtext.fontset"] = "dejavuserif"
    r["legend.fontsize"] = 5; r["axes.labelsize"] = 6; r["xtick.labelsize"] = 5; r["ytick.labelsize"] = 5
    r["axes.spines.top"] = False; r["axes.spines.right"] = False
    r["figure.dpi"] = 250; r["axes.linewidth"] = 0.5
    r["xtick.major.size"] = 2; r["xtick.major.width"] = 0.5
    r["ytick.major.size"] = 2; r["ytick.major.width"] = 0.5
    r["pdf.fonttype"] = 42; r["ps.fonttype"] = 42


def sets_panel(ax):
    """The middle panel, after the hand sketch: A and B pretrained, non-overlapping; finetuning
    goes left to C (overlaps both) or right to B' (overlaps B only)."""
    from matplotlib.patches import FancyArrowPatch
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    rng = np.random.default_rng(3)

    def cluster(x, y, name):
        dots = np.array([[-0.3, 0.28], [0.3, 0.28], [0, 0], [-0.38, -0.3], [0.38, -0.3]])
        dots = dots + rng.uniform(-0.13, 0.13, dots.shape)          # a little irregular, fixed seed
        ax.plot(dots[:, 0] + x, dots[:, 1] * 1.3 + y, "o", ms=3.4, mew=0, color=COL[name], zorder=3)
        ax.text(x, y + 0.95, name.replace("'", "\u2032"), ha="center", va="bottom", fontsize=6, color=COL[name])
    # symmetric about x = 5: pretraining on top, the two finetuning branches either side
    ax.text(5, 9.55, "pretraining", ha="center", va="center", fontsize=6)
    cluster(3.8, 7.1, "A"); cluster(6.2, 7.1, "B")
    ax.text(5, 5.35, "non-overlapping sets", ha="center", va="center", fontsize=5, color="0.55")
    for x0, x1, rad in ((4.5, 0.2, -0.25), (5.5, 9.8, 0.25)):
        ax.add_patch(FancyArrowPatch((x0, 4.7), (x1, SETS_ARROW_END), connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=5, color="0.25", lw=0.6, zorder=2))
    cluster(1.5, 3.2, "C"); cluster(8.5, 3.2, "B'")
    ax.text(5, 1.9, "finetuning", ha="center", va="center", fontsize=6)


def data_panel(ax, cond, new):
    st, A, Bal, B, n = load(cond)
    keep = st <= st[st >= FIG1_XMAX].min()             # one point past the edge, clipped by the axis
    st, A, Bal, B = st[keep], A[:, keep], Bal[:, keep], B[:, keep]
    for Y, name in ((A, "A"), (Bal, "B"), (B, new)):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=COL[name], alpha=0.15, lw=0)
        ax.plot(st, m, color=COL[name], label=name.replace("'", "\u2032"), zorder=3)
    ax.set_xlim(0, FIG1_XMAX); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("finetuning step")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, loc="lower right", handlelength=1.0, handletextpad=0.4, labelspacing=0.25,
              borderaxespad=0.2)
    return st, A


def combined():
    """Figure 1: answers from both sets' regions (C) | the sets | answers in B's region only (B')."""
    paper_style()
    fig = plt.figure(figsize=(FIG_WIDTH, 1.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.0], wspace=0.2,
                          left=0.06, right=0.985, top=0.875, bottom=0.215)
    axa, axb, axc = (fig.add_subplot(gs[k]) for k in range(3))
    data_panel(axa, "all_values", "C")
    sets_panel(axb)
    st, A = data_panel(axc, "disjoint", "B'")
    axa.set_title("catastrophic forgetting", fontsize=6, pad=4)
    axc.set_title("spurious forgetting", fontsize=6, pad=4)
    # stretch the middle panel upward (its height set by the titles)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = axa.title.get_window_extent(r); y_t = (tb.y0 + tb.y1) / 2 / fig.bbox.height
    p = axb.get_position()
    # and center it between what is drawn on either side: the left plot's rightmost tick label
    # and the right panel's y label
    x_l = max(axa.get_position().x1 * fig.bbox.width,
              max(t.get_window_extent(r).x1 for t in axa.get_xticklabels() if t.get_text())) / fig.bbox.width
    x_r = axc.yaxis.label.get_window_extent(r).x0 / fig.bbox.width
    axb.set_position([(x_l + x_r) / 2 - p.width / 2, p.y0, p.width, (y_t - p.y0) * 10 / 9.55])
    # then center it vertically: its drawing midway between the titles' top and the x labels' bottom
    fig.canvas.draw()
    top = max(ax.title.get_window_extent(r).y1 for ax in (axa, axc))
    bot = min(ax.xaxis.label.get_window_extent(r).y0 for ax in (axa, axc))
    # (the drawing's extent from its texts, dots and arrow tips: a curved arrow's bbox includes its
    # control point, which is not drawn)
    ext = [t.get_window_extent(r) for t in axb.texts] + [l.get_window_extent(r) for l in axb.lines]
    y_hi = max(e.y1 for e in ext)
    y_lo = min(min(e.y0 for e in ext), axb.transData.transform((0, SETS_ARROW_END))[1])
    p = axb.get_position()
    axb.set_position([p.x0, p.y0 + ((top + bot) / 2 - (y_hi + y_lo) / 2) / fig.bbox.height, p.width, p.height])
    # the phases, on A's curve (collapse, recovery, erosion)
    a = A.mean(0); i = int(a.argmin()); j = i + int(a[i:].argmax())
    k = i + int(np.argmin(abs(a[i:j] - (a[i] + a[j]) / 2))); e = int(np.argmin(abs(st - 400)))
    kw = dict(color="0.45", style="italic", fontsize=5, ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, shrinkA=1, shrinkB=0))   # ends on the curve
    axc.annotate("collapse", xy=(st[i], a[i]), xytext=(st[i] + 55, a[i] - 0.06), **kw)
    axc.annotate("recovery", xy=(st[k], a[k]), xytext=(st[k] + 60, a[k] - 0.08), **kw)
    axc.annotate("erosion", xy=(st[e], a[e]), xytext=(st[e] - 20, a[e] + 0.17), **kw)
    for e_ in ("png", "pdf"):
        fig.savefig(f"plots/fig1.{e_}")
    print("wrote plots/fig1")


if __name__ == "__main__":
    main()
