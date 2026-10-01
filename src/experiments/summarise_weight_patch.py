"""Cross-checkpoint readout for the weight patch (step 2).

Ranks blocks by ||P_j(50)|| - ||P_j(200)||, the change in what block j's patched weights
account for between the trough and the end of the recovery window, with cos(P_j(50), P_j(200))
ALONGSIDE. The cosine is not decoration: a piece that ROTATES rather than shrinks shows a norm
difference near zero and would be invisible without it. P_j is the difference vector
`delta_none - delta_patched`, never the shrinkage of ||delta|| -- reporting the ratio of totals
already understated one component by more than half.

Also reports whether the per-block contribution is UNIFORM across attributes. Attributes differ
from 37% to 63% in pattern share while the survivor falls 49-61% in all six; if the OV
contributions come out uniform too, that is a stronger statement than any pooled number.
Birthdate is labelled as the LOW-PATTERN case (37.6% cumulative at step 50), not the clean one.

Run:
  uv run python -m src.experiments.summarise_weight_patch --patch ov
"""
import argparse
import os
import glob

import numpy as np

OUT_DIR = "weight_patch"
ATTRS = ["birthdate", "birthplace", "university", "major", "company", "work_location"]
LOW_PATTERN = 0  # birthdate -- the low-pattern case, reported separately, not the clean case.


def load(patch, step, arm, pretrain_step, condition):
    pat = os.path.join(OUT_DIR, f"{arm}-{patch}-p{pretrain_step}-{condition}-"
                                f"step{step}-seed*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no files matching {pat}")
    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        runs.append({"names": [str(x) for x in d["names"]],
                     "removed": d["removed"].astype(np.float64),   # (C, A, E)
                     "base": d["base_norm"].astype(np.float64),    # (A,)
                     "file": os.path.basename(f)})
    n = runs[0]["names"]
    for r in runs:
        if r["names"] != n:
            raise SystemExit(f"config lists differ: {r['file']}")
    return n, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="ov")
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--steps", default="50,200")
    a = ap.parse_args()
    s_early, s_late = [int(s) for s in a.steps.split(",")]

    names, early = load(a.patch, s_early, a.arm, a.pretrain_step, a.condition)
    _, late = load(a.patch, s_late, a.arm, a.pretrain_step, a.condition)
    print(f"patch={a.patch}  step {s_early}: {len(early)} seeds   "
          f"step {s_late}: {len(late)} seeds")

    def stack(runs, key):
        return np.stack([r[key] for r in runs])            # (S, C, A, E) or (S, A)

    Pe, Pl = stack(early, "removed"), stack(late, "removed")
    be, bl = stack(early, "base"), stack(late, "base")
    print(f"  unpatched ||delta||  step {s_early}: {be.mean():.3f}   "
          f"step {s_late}: {bl.mean():.3f}")

    # Norms per seed, then averaged: keeps seed spread visible instead of averaging vectors
    # first, which would hide a seed that disagreed in direction.
    ne, nl = np.linalg.norm(Pe, axis=-1), np.linalg.norm(Pl, axis=-1)   # (S, C, A)
    cos = ((Pe * Pl).sum(-1)
           / np.clip(np.linalg.norm(Pe, axis=-1) * np.linalg.norm(Pl, axis=-1), 1e-12, None))

    rows = []
    for i, nm in enumerate(names):
        if nm == "none":
            continue
        rows.append((nm, ne[:, i].mean(), nl[:, i].mean(),
                     ne[:, i].mean() - nl[:, i].mean(), cos[:, i].mean(),
                     ne[:, i].mean(axis=0), nl[:, i].mean(axis=0), cos[:, i].mean(axis=0)))
    rows.sort(key=lambda r: -r[3])

    print(f"\n  ranked by ||P_j({s_early})|| - ||P_j({s_late})||; cosine alongside, so a piece "
          f"that rotates rather than shrinks is visible")
    print(f"  {'config':>12} {'|P|@'+str(s_early):>9} {'|P|@'+str(s_late):>9} "
          f"{'diff':>8} {'cos':>7} | interpretation")
    for nm, e, l, d, c, *_ in rows:
        if c < 0.8 and max(e, l) > 0.5:
            note = "ROTATES -- different direction at the two checkpoints"
        elif d > 0.5:
            note = "shrinks"
        elif d < -0.5:
            note = "grows"
        else:
            note = "flat"
        print(f"  {nm:>12} {e:>9.3f} {l:>9.3f} {d:>8.3f} {c:>7.3f} | {note}")

    print(f"\n  UNIFORMITY ACROSS ATTRIBUTES: ||P_j|| as a fraction of that attribute's own "
          f"||delta||, at step {s_early}")
    print(f"  {'config':>12} " + " ".join(f"{x[:9]:>9}" for x in ATTRS)
          + f" | {'spread':>7}")
    for nm, e, l, d, c, pa_e, pa_l, pa_c in rows:
        frac = pa_e / be.mean(axis=0)
        print(f"  {nm:>12} " + " ".join(f"{v:>9.1%}" for v in frac)
              + f" | {frac.max()-frac.min():>6.1%}")

    print(f"\n  birthdate ({ATTRS[LOW_PATTERN]}) is the LOW-PATTERN case (37.6% of its delta "
          f"is pattern-produced cumulatively at step 50, against 49-63% for the rest).")
    print(f"  It is reported separately for that reason, not as the clean case.")
    for nm, e, l, d, c, pa_e, pa_l, pa_c in rows:
        if nm in ("blk7", f"blk0-7"):
            print(f"    {nm:>8}  birthdate |P|@{s_early}={pa_e[LOW_PATTERN]:.3f} "
                  f"@{s_late}={pa_l[LOW_PATTERN]:.3f} cos={pa_c[LOW_PATTERN]:.3f} | "
                  f"other five mean @{s_early}={np.delete(pa_e, LOW_PATTERN).mean():.3f} "
                  f"@{s_late}={np.delete(pa_l, LOW_PATTERN).mean():.3f}")

    print(f"\n  seed spread (max-min of ||P_j|| over seeds, step {s_early}): "
          f"{(ne.max(0) - ne.min(0)).max():.4f} worst case over all configs and attributes")


if __name__ == "__main__":
    main()
