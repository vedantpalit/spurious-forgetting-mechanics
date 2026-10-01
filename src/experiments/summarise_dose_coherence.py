"""Does de-coherence track B's clock or the step clock?

Reads `dose_coherence/*.npz`. Two panels, the same shape as the dose figure:

  panel 1   mean pairwise cosine vs STEP        -- the step-count alternative predicts the
                                                   conditions lying on top of each other
  panel 2   mean pairwise cosine vs B'S ACCURACY -- the account predicts the conditions lying
                                                   on top of each other HERE instead

Only one of the two can collapse the conditions onto a single curve, so the comparison is the
test. It is quantified as the spread of the four conditions at matched x, averaged over the
range they share -- smaller spread on B's clock than on the step clock is the result, and the
ratio of the two is the effect size.

Also reports the step at which coherence crosses a threshold in each condition, against B's
onset, as the direct analogue of the dose runs' turnover-vs-onset regression.

Run: uv run python -m src.experiments.summarise_dose_coherence
"""
import argparse
import glob
import os

import numpy as np

OUT_DIR = "dose_coherence"
# From the earlier dose runs: the step at which B's accuracy first exceeds 0.5. Used only for
# the regression panel; the vs-B-accuracy comparison uses each run's OWN measured B accuracy.
B_ONSET = {"lr2.0-bs256": 40, "lr1.0-bs512": 60, "lr1.0-bs128": 80, "lr0.5-bs256": 120}


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        lr, bs = float(d["lr_mult"]), int(d["inject_batch"])
        tag = "baseline" if (lr == 1.0 and bs == 256) else f"lr{lr}-bs{bs}"
        runs.setdefault(tag, []).append(
            dict(steps=d["steps"], b_acc=d["b_acc"],
                 cos=d["pairwise_cos"].mean(-1), base=d["base_norm"].mean(-1),
                 seed=int(d["seed"]), file=os.path.basename(f)))
    if not runs:
        raise SystemExit(f"no npz in {OUT_DIR}/")
    return runs


def agg(rs):
    """Mean over seeds. Every seed of a condition shares the same step grid."""
    st = rs[0]["steps"]
    for r in rs:
        if not np.array_equal(r["steps"], st):
            raise SystemExit(f"step grids differ within a condition: {r['file']}")
    return st, (np.mean([r["b_acc"] for r in rs], 0), np.mean([r["cos"] for r in rs], 0),
                np.mean([r["base"] for r in rs], 0), len(rs))


def spread_at(xs, ys, grid):
    """Mean over `grid` of the max-min across conditions, on the shared x range only.

    STRICTLY INCREASING X IS REQUIRED, and this is not pedantry. B's accuracy saturates at
    1.000 several checkpoints before the end, so on the B-accuracy axis many steps share one x
    and `np.interp` with non-increasing `xp` returns whatever it likes. Worse, the degenerate
    region is where much of the de-coherence happens -- 60-92% of the remaining fall occurs
    after B reaches 0.99 -- so interpolating through it silently compresses the tail and makes
    the B axis look better than it is. Callers restrict to the increasing region instead.
    """
    for x in xs:
        if np.any(np.diff(x) <= 0):
            raise SystemExit(f"x is not strictly increasing: {x}. Restrict to the region where "
                             f"it is; see the note above.")
    lo = max(x.min() for x in xs)
    hi = min(x.max() for x in xs)
    g = grid[(grid >= lo) & (grid <= hi)]
    if len(g) < 3:
        return np.nan, 0, (lo, hi)
    v = np.stack([np.interp(g, x, y) for x, y in zip(xs, ys)])
    return float((v.max(0) - v.min(0)).mean()), len(g), (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b_cap", type=float, default=0.95,
                    help="collapse test uses only points below this B accuracy, where B acc "
                         "is strictly increasing")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="pairwise-cosine level whose crossing step is regressed on B's onset")
    a = ap.parse_args()
    runs = load()

    print(f"{len(runs)} conditions: " + ", ".join(f"{k} (n={len(v)})" for k, v in runs.items()))
    print(f"\n{'condition':>14} {'step':>6} {'B acc':>7} {'||delta||':>10} {'pairwise cos':>13}")
    data = {}
    for tag in sorted(runs):
        st, (b, c, bn, ns) = agg(runs[tag])
        data[tag] = (st, b, c, bn)
        for i in range(len(st)):
            print(f"{tag if i == 0 else '':>14} {st[i]:>6} {b[i]:>7.3f} {bn[i]:>10.3f} "
                  f"{c[i]:>13.4f}")
        print()

    tags = sorted(data)
    print(f"COLLAPSE TEST -- which x-axis puts the conditions on one curve?")
    print(f"  restricted to B acc < {a.b_cap}, where B's accuracy is strictly increasing; both "
          f"axes use the SAME points, so the comparison is not between different subsets")
    sel = {t: (data[t][1] < a.b_cap) for t in tags}
    for t in tags:
        st, b, c, _ = data[t]
        k = sel[t]
        print(f"  {t:>14} keeps {k.sum()}/{len(b)}  steps {st[k].min():.0f}-{st[k].max():.0f}  "
              f"B acc {b[k].min():.3f}-{b[k].max():.3f}")
    s_step, n1, r1 = spread_at([data[t][0].astype(float)[sel[t]] for t in tags],
                               [data[t][2][sel[t]] for t in tags], np.arange(10, 401, 1.0))
    s_bacc, n2, r2 = spread_at([data[t][1][sel[t]] for t in tags],
                               [data[t][2][sel[t]] for t in tags], np.linspace(0, 1, 201))
    print(f"  vs STEP    : mean spread {s_step:.4f} over {n1} points in [{r1[0]:.0f}, {r1[1]:.0f}]")
    print(f"  vs B ACC   : mean spread {s_bacc:.4f} over {n2} points in [{r2[0]:.2f}, {r2[1]:.2f}]")
    if np.isfinite(s_step) and np.isfinite(s_bacc) and s_bacc > 0:
        print(f"  ratio step/B = {s_step/s_bacc:.2f}  "
              f"({'B clock' if s_bacc < s_step else 'STEP clock'} collapses them better)")

    print(f"\nAFTER B SATURATES -- de-coherence does not stop, which the account does not predict")
    print(f"  {'condition':>14} {'step B>=.99':>12} {'cos there':>10} {'cos at end':>11} "
          f"{'further fall':>13}")
    for t in tags:
        st, b, c, _ = data[t]
        i = np.where(b >= 0.99)[0]
        if not len(i):
            print(f"  {t:>14} {'never reaches .99':>12}")
            continue
        i = i[0]
        print(f"  {t:>14} {st[i]:>12.0f} {c[i]:>10.3f} {c[-1]:>11.3f} "
              f"{(c[i]-c[-1])/max(c[i], 1e-9):>12.1%}")

    print(f"\nCROSSING STEP at pairwise cos = {a.threshold}, against B's onset (dose runs)")
    print(f"  {'condition':>14} {'B onset':>8} {'crossing':>9}")
    xs, ys = [], []
    for tag in tags:
        st, b, c, _ = data[tag]
        below = np.where(c <= a.threshold)[0]
        if not len(below) or below[0] == 0:
            print(f"  {tag:>14} {B_ONSET.get(tag, float('nan')):>8} "
                  f"{'not bracketed':>9}")
            continue
        i = below[0]
        # linear interpolation between the bracketing checkpoints
        x0, x1, y0, y1 = st[i-1], st[i], c[i-1], c[i]
        cross = x0 + (y0 - a.threshold) * (x1 - x0) / max(y0 - y1, 1e-12)
        print(f"  {tag:>14} {B_ONSET.get(tag, float('nan')):>8} {cross:>9.1f}")
        if tag in B_ONSET:
            xs.append(B_ONSET[tag]); ys.append(cross)
    if len(xs) >= 3:
        xs, ys = np.array(xs, float), np.array(ys, float)
        sl, ic = np.polyfit(xs, ys, 1)
        r = np.corrcoef(xs, ys)[0, 1]
        print(f"  fit: slope {sl:.3f}  intercept {ic:.1f}  r = {r:.4f}   (the dose runs' "
              f"turnover fit was slope 0.800, intercept -5.0, r = 0.9978)")
        print(f"  the step-count alternative predicts a FLAT crossing; observed range "
              f"{ys.min():.0f} to {ys.max():.0f}")
    else:
        print("  fewer than 3 conditions bracketed the threshold; try --threshold")


if __name__ == "__main__":
    main()
