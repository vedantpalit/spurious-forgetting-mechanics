"""The three appendix figures of the minimal-model analysis, in the style of Figure 3.

    plots/app_shift_validation.{png,pdf}  (Appendix, empirical validation)
        left: s = |mean old-fact state|, measured (solid) and predicted by the shift equation
              integrated along the run (dashed), with and without normalization;
        right: the new facts' mean shared and individual margins, with normalization.
        Data: reduced_equation.json (ten seeds; the prediction is s_pred2: the equation evaluated on
        each step's minibatch plus the finite-step term |D_perp|^2 / 2s, integrated from step 10).
    plots/app_erosion.{png,pdf}  (Appendix, fact-specific displacements)
        left: the spread eps_rms of the old hidden states against sqrt((1-a)/d) |dW|_F and
              sqrt((1-a)/d) |dW_perp|_F over finetuning; right: measured against predicted.
        Data: paper_base.json, normalized runs (n1), the same runs as Figures 2-3.
    plots/app_readout_frozen.{png,pdf}  (Appendix, why recovery is incomplete)
        old- and new-fact accuracy with the readout trained (paper_base.json, n1) and frozen
        (paper_base_u0.json, n1; its rate re-matched so the new facts are learned by the same step).

Run: uv run --frozen python plot_app_figs.py   (from toy/)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
FIG_WIDTH, FIG_H = 397 / 72, 115.2 / 72
NAVY, CRIM, TEAL, DEEP, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#1E5E6B", "#8A8C30", "0.55"
ALPHA, D = 0.5, 128


def style():
    matplotlib.rcParams.update({
        "lines.linewidth": 1, "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "legend.fontsize": 5, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 250,
        "mathtext.fontset": "dejavuserif", "xtick.major.size": 2, "xtick.major.width": 0.5,
        "ytick.major.size": 2, "ytick.major.width": 0.5, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def band(ax, st, Y, col, ls="-", lab=None, lw=1.0):
    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
    ax.plot(st, mu, color=col, ls=ls, lw=lw, label=lab, zorder=3)
    return mu


def runs(path, pred):
    d = json.load(open(os.path.join(HERE, path)))
    return [v for k, v in sorted(d.items()) if pred(k, v)]


def stack(rs, key, xmax=None):
    st = np.array([r["step"] for r in rs[0]["rows"]], float)
    keep = st <= (xmax if xmax else st.max())
    return st[keep], np.array([[r[key] for r in v["rows"]] for v in rs])[:, keep]


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{name}.{e}"))
    plt.close(fig)
    print(f"wrote plots/{name}")


def shift_validation():
    XMAX = 500
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_H), gridspec_kw={"wspace": 0.3})
    for norm, col, lab in ((1, TEAL, "normalized"), (0, OLIVE, "not normalized")):
        rs = runs("reduced_equation.json", lambda k, v: v["norm"] == norm)
        st, S = stack(rs, "s", XMAX); _, P = stack(rs, "s_pred2", XMAX)
        band(a1, st, S, col, lab=lab)
        a1.plot(st, P.mean(0), color="0.2", ls=(0, (2, 1.5)), lw=0.8, zorder=4)
        rng = S.mean(0).max() - S.mean(0).min()
        err = np.abs(P.mean(0) - S.mean(0)).max() / rng
        print(f"  s, {lab}: max |predicted - measured| = {100 * err:.1f}% of the range")
        if norm == 1:
            peak = st[int(S.mean(0).argmax())]
            _, G = stack(rs, "G", XMAX); _, M = stack(rs, "M", XMAX)
            band(a2, st, M, DEEP, lab=r"individual $\bar\gamma_{\mathrm{ind}}$")
            band(a2, st, G, GREY, lab=r"shared $\bar\gamma_{\mathrm{sh}}$")
    for ax in (a1, a2):
        ax.axvline(peak, color="0.55", ls=":", lw=0.8, zorder=1)
        ax.set_xlim(0, XMAX); ax.set_xlabel("finetuning step")
    h, l = a1.get_legend_handles_labels()
    h.append(Line2D([], [], color="0.2", ls=(0, (2, 1.5)), lw=0.8)); l.append("predicted")
    a1.legend(h, l, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
    a1.set_ylabel("mean hidden state norm $s$"); a1.set_ylim(0, None)
    a2.axhline(0, color="0.8", lw=0.5, zorder=0)
    a2.legend(frameon=False, loc="lower right"); a2.set_ylabel("margin")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.97)
    save(fig, "app_shift_validation")


def erosion():
    rs = runs("paper_base.json", lambda k, v: v["norm"] == 1)
    st, E = stack(rs, "eps_h"); _, W = stack(rs, "dW"); _, Wp = stack(rs, "dW_perp")
    c = np.sqrt((1 - ALPHA) / D)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_H), gridspec_kw={"wspace": 0.3})
    band(a1, st, E, NAVY, lab=r"measured $\varepsilon_{\mathrm{rms}}$")
    band(a1, st, c * W, OLIVE, "--", lab=r"$\sqrt{(1-\alpha)/d}\,\|\Delta W_1\|_F$", lw=0.9)
    band(a1, st, c * Wp, TEAL, ":", lab=r"$\sqrt{(1-\alpha)/d}\,\|\Delta W_\perp\|_F$", lw=0.9)
    a1.set_xlim(0, st.max()); a1.set_xlabel("finetuning step"); a1.set_ylabel("spread of old hidden states")
    a1.legend(frameon=False, loc="lower right")
    x, y = (c * Wp).ravel(), E.ravel()
    lim = max(x.max(), y.max()) * 1.05
    a2.plot([0, lim], [0, lim], color="0.8", lw=0.6, zorder=0)
    a2.scatter(x, y, s=2, color=NAVY, alpha=0.5, lw=0)
    a2.set_xlim(0, lim); a2.set_ylim(0, lim)
    a2.set_xlabel(r"$\sqrt{(1-\alpha)/d}\,\|\Delta W_\perp\|_F$"); a2.set_ylabel(r"measured $\varepsilon_{\mathrm{rms}}$")
    ratio = y[x > 0.2 * x.max()] / x[x > 0.2 * x.max()]
    print(f"  erosion: measured / predicted (dW_perp) = {ratio.mean():.3f} +- {ratio.std():.3f}")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.97)
    save(fig, "app_erosion")


def readout_frozen():
    XMAX = 1000
    tr = runs("paper_base.json", lambda k, v: v["norm"] == 1)
    fr = runs("paper_base_u0.json", lambda k, v: v["norm"] == 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_H), gridspec_kw={"wspace": 0.3}, sharey=True)
    for ax, key, name in ((a1, "A", "old facts"), (a2, "B", "new facts")):
        st, Yt = stack(tr, key, XMAX); sf, Yf = stack(fr, key, XMAX)
        band(ax, st, Yt, NAVY if key == "A" else CRIM, lab="readout trained")
        band(ax, sf, Yf, NAVY if key == "A" else CRIM, (0, (3, 2)), lab="readout frozen")
        ax.set_xlim(0, XMAX); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("finetuning step")
        ax.text(0.97, 0.95, name, transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
        if key == "A":
            i = int(Yt.mean(0).argmin()); j = int(Yf.mean(0).argmin())
            print(f"  readout: old facts, recovered peak after the trough: trained {Yt.mean(0)[i:].max():.2f}, "
                  f"frozen {Yf.mean(0)[j:].max():.2f}")
    a1.set_ylabel("accuracy")
    a1.legend([Line2D([], [], color="0.3", lw=1), Line2D([], [], color="0.3", lw=1, ls=(0, (3, 2)))],
              ["readout trained", "readout frozen"], frameon=False, loc="lower right")
    plt.setp(a2.get_yticklabels(), visible=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.97)
    save(fig, "app_readout_frozen")


def replay():
    """Never-replayed old facts when a fixed subset of old facts is replayed in every batch."""
    d = json.load(open(os.path.join(HERE, "replay_ps16.json")))
    NS = (0, 4, 16, 64)
    cols = {0: NAVY, 4: "#4A78B5", 16: TEAL, 64: OLIVE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_H), gridspec_kw={"wspace": 0.3})
    for n in NS:
        rs = [v for v in d.values() if v["n_rep"] == n]
        st, Y = stack(rs, "A_rest", 3000)
        band(a1, st, Y, cols[n], lab=f"{n} replayed" if n else "no replay")
    a1.set_xlim(0, 3000); a1.set_ylim(-0.03, 1.03); a1.set_xlabel("finetuning step")
    a1.set_ylabel("never-replayed old facts"); a1.legend(frameon=False, loc="lower right")
    for ps, mk in ((16, "o"), (4, "s")):
        dd = json.load(open(os.path.join(HERE, f"replay_ps{ps}.json")))
        allN = sorted({v["n_rep"] for v in dd.values()})
        def at(n, which):
            vals = []
            for v in dd.values():
                if v["n_rep"] != n:
                    continue
                a = np.array([r["A_rest"] for r in v["rows"]])
                vals.append(a.min() if which == "trough" else a[-1])
            return np.mean(vals)
        for which, col in (("trough", TEAL), ("end", NAVY)):
            gain = [at(n, which) - at(0, which) for n in allN]
            a2.plot(allN, gain, color=col, marker=mk, ms=2.5, lw=0.9,
                    ls="-" if ps == 16 else (0, (3, 2)))
            print(f"  replay, {ps}/step, {which}: gain over no replay " +
                  ", ".join(f"{n}:{g:+.2f}" for n, g in zip(allN, gain)))
    a2.set_xscale("symlog", linthresh=1); a2.set_xticks([0, 1, 4, 16, 64]); a2.set_xticklabels(["0", "1", "4", "16", "64"])
    a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.set_xlabel("old facts replayed"); a2.set_ylabel("gain over no replay")
    a2.axhline(0, color="0.8", lw=0.5, zorder=0)
    a2.legend([Line2D([], [], color=TEAL, lw=1), Line2D([], [], color=NAVY, lw=1),
               Line2D([], [], color="0.3", lw=0.9, marker="o", ms=2.5),
               Line2D([], [], color="0.3", lw=0.9, marker="s", ms=2.5, ls=(0, (3, 2)))],
              ["at the trough", "at the end", "16 replayed per step", "4 replayed per step"],
              frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.97)
    save(fig, "app_replay")


def shared_row():
    """Old and new facts as trained and with the part of the store's update that acts on the shared
    key direction removed, W(t) - mu mu^T dW (the paper's dW_1 (I - mu mu^T)); readout as trained.
    Data: patch_weights.json (patch_weights.py; the same ten normalized runs as paper_base.json)."""
    XMAX = 1000
    d = json.load(open(os.path.join(HERE, "patch_weights.json")))
    st = np.array([x["step"] for x in d["0"]], float); keep = st <= XMAX; st = st[keep]
    get = lambda key, fb: np.array([[x.get(key, x[fb]) for x in d[s]] for s in d])[:, keep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_H), gridspec_kw={"wspace": 0.3}, sharey=True)
    for ax, key, col, name in ((a1, "A", NAVY, "old facts"), (a2, "B", CRIM, "new facts")):
        T = get(key, key); R = get(f"murow/{key}", key)
        band(ax, st, T, col, lab=name)
        band(ax, st, R, OLIVE, lab=f"{name}, shared part of the update removed")
        ax.set_xlim(0, XMAX); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("finetuning step")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.05))
        i = int(T.mean(0).argmin()) if key == "A" else int(np.argmin(abs(st - 200)))
        print(f"  shared row, {name}: step {st[i]:.0f}: trained {T.mean(0)[i]:.3f}, removed {R.mean(0)[i]:.3f}; "
              f"at {st[-1]:.0f}: {T.mean(0)[-1]:.3f} / {R.mean(0)[-1]:.3f}")
    tr = st[int(get("A", "A").mean(0).argmin())]
    for ax in (a1, a2):
        ax.axvline(tr, color=GREY, ls=":", lw=0.8, zorder=1)
    a1.set_ylabel("accuracy")
    plt.setp(a2.get_yticklabels(), visible=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.97)
    save(fig, "app_shared_row")


if __name__ == "__main__":
    style()
    shift_validation()
    erosion()
    readout_frozen()
    replay()
    shared_row()
