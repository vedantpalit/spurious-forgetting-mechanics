"""The laws, read off the sweep data of the minimal model. Nothing is asserted; every line is
a fit or a ratio with its spread.

    python laws.py            # sw_*.json, erosion_long.json (readout at the single rate)
    python laws.py --suffix _u0   # the same files with the readout frozen during injection

L1  Erosion identity. On the raw state, eps_a = sqrt(1-b) (g_a dW) + O(1/sqrt d), and
    E||g M|| = ||M||_F / sqrt(d) for g uniform on the sphere, so
        ||eps_a||_h  =  sqrt((1-b)/d) ||dW_perp||_F .
    Reported: measured / predicted at the end of every matched run, across every sweep.
L2  Erosion in time: eps = c ln t + k on the tail of the long runs (t >= 1000), with R^2.
L3  Depth of suppression: the peak of the write's between-half content <s,w> against
    sqrt(b) n_B across the beta and n_B sweeps (correlation and a log-log slope for n_B).
L4  Timing: on the dose sweep, log-log slope of the trough step and of the recovery step
    against the rate; the ratio t_rec / t_trough per dose.
L5  Dose invariance: trough depth and the post-trough maximum per dose.
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, suffix):
    f = os.path.join(HERE, f"{name}{suffix}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def fit_loglog(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([a, b]); r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, r2


def series(r, key):
    return np.array([x[key] for x in r["rows"]], float)


def timing(r):
    st, A = series(r, "step"), series(r, "A")
    tr = int(A.argmin()); mx = A[tr:].max()
    rec = tr + int(np.argmax(A[tr:] >= A[tr] + 0.9 * (mx - A[tr])))
    return st[tr], st[rec], A[tr], mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    S = a.suffix

    # L1 -- erosion identity, end of every matched normalized-and-linear run in every sweep
    print("L1  erosion identity  ||eps||_h / ( sqrt((1-b)/d) ||dW_perp||_F )  at the end of each run")
    ratios = []
    for f in sorted(glob.glob(os.path.join(HERE, f"sw_*{S}.json"))):
        if S == "" and "_u" in os.path.basename(f):
            continue
        d = json.load(open(f)); rr = []
        for r in d.values():
            if not r["matched"] and r["sweep"] != "dose":
                continue
            x = r["rows"][-1]
            pred = np.sqrt((1 - r["beta"]) / r["d"]) * x["dW_perp"]
            rr.append(x["eps_h"] / pred)
        rr = np.array(rr); ratios += list(rr)
        print(f"    {os.path.basename(f):>18}: {rr.mean():.3f} +/- {rr.std():.3f}  (n={len(rr)})")
    ratios = np.array(ratios)
    print(f"    all: {ratios.mean():.3f} +/- {ratios.std():.3f}  (n={len(ratios)})")

    # L2 -- erosion in time
    d = load("erosion_long", S)
    if d:
        print("\nL2  eps = c ln t + k on t >= 1000 (long runs, normalized arm)")
        for k, r in d.items():
            st, eps = series(r, "step"), series(r, "eps")
            keep = st >= 1000
            A = np.vstack([np.log(st[keep]), np.ones(keep.sum())]).T
            (c, kk), *_ = np.linalg.lstsq(A, eps[keep], rcond=None)
            pred = A @ np.array([c, kk]); r2 = 1 - ((eps[keep] - pred) ** 2).sum() / ((eps[keep] - eps[keep].mean()) ** 2).sum()
            # alternative: power law
            a_, r2p = fit_loglog(st[keep], eps[keep])
            print(f"    {k}: c={c:.4f} k={kk:.3f} R^2={r2:.4f} | power-law exponent {a_:.3f} R^2={r2p:.4f} "
                  f"| eps {eps[keep][0]:.3f} -> {eps[-1]:.3f} over {int(st[keep][0])} -> {int(st[-1])}; "
                  f"A end {series(r, 'A')[-1]:.2f}, s.w end {series(r, 's_on_w')[-1]:.3f}")

    # L3 -- depth of the write vs sqrt(b) n_B
    print("\nL3  peak of <s,w> vs sqrt(beta) n_B  (normalized arm, matched cells)")
    xs, ys, ax = [], [], []
    for name in ("sw_beta", "sw_nb"):
        d = load(name, S)
        if not d:
            continue
        for r in d.values():
            if r["norm"] != 1 or not r["matched"]:
                continue
            xs.append(np.sqrt(r["beta"]) * r["nb"]); ys.append(series(r, "s_on_w").max())
            ax.append((name, r["value"], series(r, "A").min()))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs):
        print(f"    corr(<s,w>_peak, sqrt(b) n_B) = {np.corrcoef(xs, ys)[0, 1]:.3f}  (n={len(xs)})")
    d = load("sw_nb", S)
    if d:
        v = [(r["value"], series(r, "s_on_w").max()) for r in d.values() if r["norm"] == 1 and r["matched"]]
        sl, r2 = fit_loglog([x for x, _ in v], [y for _, y in v])
        print(f"    n_B alone: <s,w>_peak ~ n_B^{sl:.2f}  (R^2 {r2:.3f})")
    d = load("sw_beta", S)
    if d:
        v = [(r["value"], series(r, "s_on_w").max()) for r in d.values() if r["norm"] == 1 and r["matched"] and r["value"] > 0]
        sl, r2 = fit_loglog([x for x, _ in v], [y for _, y in v])
        print(f"    beta alone: <s,w>_peak ~ beta^{sl:.2f}  (R^2 {r2:.3f}; the derivation says 0.5)")

    # L4, L5 -- timing and dose invariance
    d = load("sw_dose", S)
    if d:
        print("\nL4/L5  dose sweep (normalized arm): trough step, recovery step, depth, destination")
        rows = {}
        for r in d.values():
            if r["norm"] != 1:
                continue
            rows.setdefault(r["value"], []).append(timing(r))
        doses = sorted(rows)
        tt, trc = [], []
        for v in doses:
            arr = np.array(rows[v]); tt.append(arr[:, 0].mean()); trc.append(arr[:, 1].mean())
            print(f"    x{v:<5g} trough @{arr[:, 0].mean():7.1f}  rec @{arr[:, 1].mean():7.1f}  "
                  f"ratio {np.mean(arr[:, 1] / np.maximum(arr[:, 0], 1)):5.2f}  depth {arr[:, 2].mean():.3f}"
                  f" +/- {arr[:, 2].std():.3f}  destination {arr[:, 3].mean():.3f}")
        s1, r1 = fit_loglog(doses, tt); s2, r2 = fit_loglog(doses, trc)
        print(f"    trough step ~ rate^{s1:.2f} (R^2 {r1:.3f});  recovery step ~ rate^{s2:.2f} (R^2 {r2:.3f})")
        depths = [np.mean([x[2] for x in rows[v]]) for v in doses]
        print(f"    depth across {doses[-1] / doses[0]:g}x in rate: {min(depths):.3f} to {max(depths):.3f}")


if __name__ == "__main__":
    main()
