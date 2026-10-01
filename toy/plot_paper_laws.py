"""One figure per law, minimal model (data: sw_*.json, erosion_long.json, paper_base.json).

    plots/toy_min_law_timing.{png,pdf}     trough and recovery step against the rate (log-log)
    plots/toy_min_law_depth.{png,pdf}      the shift's peak against sqrt(beta) n_B (log-log)
    plots/toy_min_law_erosion_identity     ||eps||_h measured against sqrt((1-beta)/d) ||dW_perp||_F
    plots/toy_min_law_erosion_time         eps against t, log x, with the c ln t + k fit
    plots/toy_min_law_recovery             reversal fraction measured against 1 - K/x_peak

The (B, A)-plane sweeps (plot_paper_sweeps_AB.py) are the companion figures per knob.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.grid": False})


def save(fig, name):
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig); print(f"wrote {name}")


def loglog_fit(x, y):
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b; r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return a, b, r2


def series(r, k):
    return np.array([x[k] for x in r["rows"]], float)


def timing():
    d = json.load(open(os.path.join(HERE, "sw_dose.json")))
    pts = {}
    for r in d.values():
        if r["norm"] != 1:
            continue
        st, A = series(r, "step"), series(r, "A")
        tr = int(A.argmin()); mx = A[tr:].max()
        rec = tr + int(np.argmax(A[tr:] >= A[tr] + 0.9 * (mx - A[tr])))
        pts.setdefault(r["value"], []).append((st[tr], st[rec]))
    x = np.array(sorted(pts)); T = np.array([np.mean([p[0] for p in pts[v]]) for v in x]); R = np.array([np.mean([p[1] for p in pts[v]]) for v in x])
    a1, b1, r1 = loglog_fit(x, T); a2, b2, r2 = loglog_fit(x, R)
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    ax.plot(xx, np.exp(b1) * xx ** a1, color=NAVY, lw=1.2, ls="--"); ax.plot(xx, np.exp(b2) * xx ** a2, color=TEAL, lw=1.2, ls="--")
    for v in x:
        ax.scatter([v] * len(pts[v]), [p[0] for p in pts[v]], color=NAVY, s=18, zorder=3)
        ax.scatter([v] * len(pts[v]), [p[1] for p in pts[v]], color=TEAL, s=18, marker="s", zorder=3)
    ax.scatter([], [], color=NAVY, s=18, label=f"trough step $\\propto$ rate$^{{{a1:.2f}}}$")
    ax.scatter([], [], color=TEAL, s=18, marker="s", label=f"recovery step $\\propto$ rate$^{{{a2:.2f}}}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("injection rate (relative)"); ax.set_ylabel("step")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "toy_min_law_timing")
    print(f"  trough ~ rate^{a1:.2f} (R2 {r1:.3f}); recovery ~ rate^{a2:.2f} (R2 {r2:.3f}); ratio {np.mean(R / T):.2f}")


def depth():
    xs, ys, cols = [], [], []
    for f, col in (("sw_beta.json", NAVY), ("sw_nb.json", TEAL)):
        for r in json.load(open(os.path.join(HERE, f))).values():
            if r["norm"] != 1 or not r["matched"] or r["beta"] == 0:
                continue
            xs.append(np.sqrt(r["beta"]) * r["nb"]); ys.append(series(r, "s_on_w").max()); cols.append(col)
    xs, ys = np.array(xs), np.array(ys)
    a, b, r2 = loglog_fit(xs, ys)
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    ax.scatter(xs, ys, c=cols, s=18, zorder=3)
    xx = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
    ax.plot(xx, np.exp(b) * xx ** a, color=GREY, lw=1.2, ls="--", label=f"$\\propto (\\sqrt{{\\beta}}\\,n_B)^{{{a:.2f}}}$, $R^2$ = {r2:.2f}")
    ax.scatter([], [], color=NAVY, s=18, label="$\\beta$ sweep ($n_B$ = 128)"); ax.scatter([], [], color=TEAL, s=18, label="$n_B$ sweep ($\\beta$ = 0.5)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\sqrt{\\beta}\\; n_B$"); ax.set_ylabel("peak of the shift $\\langle s, \\hat w\\rangle$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "toy_min_law_depth")
    print(f"  peak ~ (sqrt(b) n_B)^{a:.2f}, R2 {r2:.3f}, n={len(xs)}")


def depth_beta():
    """S_peak against beta alone (n_B = 128): the derived sqrt(beta) line, anchored at the
    geometric mean of the data, next to the free-exponent fit. Three seeds per beta."""
    xs, ys = [], []
    for r in json.load(open(os.path.join(HERE, "sw_beta.json"))).values():
        if r["norm"] != 1 or not r["matched"] or r["beta"] == 0:
            continue
        xs.append(r["beta"]); ys.append(series(r, "s_on_w").max())
    xs, ys = np.array(xs), np.array(ys)
    a, b, r2 = loglog_fit(xs, ys)
    c = np.exp(np.mean(np.log(ys) - 0.5 * np.log(xs)))          # sqrt(beta) line through the data
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    xx = np.logspace(np.log10(xs.min()) - 0.05, np.log10(xs.max()) + 0.02, 50)
    ax.plot(xx, c * np.sqrt(xx), color=CRIM, lw=1.6, ls="-", label="predicted $\\propto \\sqrt{\\beta}$")
    ax.plot(xx, np.exp(b) * xx ** a, color=GREY, lw=1.2, ls="--", label=f"fit $\\propto \\beta^{{{a:.2f}}}$")
    ax.scatter(xs, ys, color=NAVY, s=22, zorder=3, label="measured, three seeds")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sorted(set(xs))); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("key relatedness $\\beta$"); ax.set_ylabel("peak of the shift $S_{\\mathrm{peak}}$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "toy_min_law_depth_beta")
    print(f"  peak ~ beta^{a:.2f}, R2 {r2:.3f}, n={len(xs)}; sqrt(beta) line coefficient {c:.3f}")


def erosion_identity():
    xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "sw_*.json"))):
        if "_u" in os.path.basename(f):
            continue
        for r in json.load(open(f)).values():
            if not r["matched"] and r["sweep"] != "dose":
                continue
            x = r["rows"][-1]
            xs.append(np.sqrt((1 - r["beta"]) / r["d"]) * x["dW_perp"]); ys.append(x["eps_h"])
    xs, ys = np.array(xs), np.array(ys); ratio = ys / xs
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    lim = [min(xs.min(), ys.min()) * 0.8, max(xs.max(), ys.max()) * 1.2]
    ax.plot(lim, lim, color=GREY, lw=1.0, ls="--", label="identity")
    ax.scatter(xs, ys, color=NAVY, s=14, alpha=0.8, zorder=3, label=f"{len(xs)} runs, six sweeps, both arms")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    from matplotlib.ticker import LogLocator, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=6)); axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$\\sqrt{(1-\\beta)/d}\\;\\|\\Delta W_\\perp\\|_F$"); ax.set_ylabel("$\\|\\varepsilon_a\\|$ measured")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "toy_min_law_erosion_identity")
    print(f"  identity: measured/predicted {ratio.mean():.3f} +/- {ratio.std():.3f}, n={len(xs)}")


def erosion_time():
    d = json.load(open(os.path.join(HERE, "erosion_long.json")))
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    r2s = []
    for i, (k, r) in enumerate(d.items()):
        st, eps = series(r, "step"), series(r, "eps"); keep = st >= 1000
        c, kk = np.polyfit(np.log(st[keep]), eps[keep], 1)
        pred = c * np.log(st[keep]) + kk; r2s.append(1 - ((eps[keep] - pred) ** 2).sum() / ((eps[keep] - eps[keep].mean()) ** 2).sum())
        m = st > 0
        ax.plot(st[m], eps[m], color=NAVY, lw=1.4, alpha=0.8, label="rms $\\|\\varepsilon_a\\|$, three seeds" if i == 0 else None)
        xx = np.logspace(3, np.log10(st[-1]), 50)
        ax.plot(xx, c * np.log(xx) + kk, color=CRIM, lw=1.2, ls="--", label="$c\\ln t + k$ on $t \\geq 1000$" if i == 0 else None)
    ax.set_xscale("log"); ax.set_xlabel("injection step"); ax.set_ylabel("individual displacement")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "toy_min_law_erosion_time")
    print(f"  eps = c ln t + k: R2 {min(r2s):.4f}-{max(r2s):.4f}")


def recovery():
    rows = []
    for f, tag in (("sw_beta.json", "beta"), ("sw_nb.json", "nb"), ("sw_dose.json", "dose"), ("paper_base.json", "base")):
        for r in json.load(open(os.path.join(HERE, f))).values():
            if r["norm"] != 1 or (not r.get("matched", True) and tag != "dose"):
                continue
            sw = series(r, "s_on_w"); pk = sw.argmax()
            if sw[pk] < 0.3:
                continue
            rows.append((tag, sw[pk], sw[-1]))
    peak = np.array([r[1] for r in rows]); end = np.array([r[2] for r in rows]); K = end.mean()
    fm = 1 - end / peak; fp = 1 - K / peak
    r2 = 1 - ((fm - fp) ** 2).sum() / ((fm - fm.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    cols = {"beta": NAVY, "nb": TEAL, "dose": OLIVE, "base": CRIM}
    for tag, col, lab in (("beta", NAVY, "$\\beta$ sweep"), ("nb", TEAL, "$n_B$ sweep"), ("dose", OLIVE, "rate sweep"), ("base", CRIM, "base, ten seeds")):
        m = np.array([r[0] == tag for r in rows])
        ax.scatter(fp[m], fm[m], color=col, s=18, zorder=3, label=lab)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"predicted  $1 - S_\\infty / S_{{\\mathrm{{peak}}}}$   ($S_\\infty$ = {K:.2f})"); ax.set_ylabel("fraction of the shift withdrawn, measured")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", title=f"$R^2$ = {r2:.2f}", title_fontsize=8.5)
    save(fig, "toy_min_law_recovery")
    print(f"  f = 1 - K/x_peak: K {K:.3f}, R2 {r2:.3f}, n={len(rows)}")


if __name__ == "__main__":
    timing(); depth(); depth_beta(); erosion_identity(); erosion_time(); recovery()
