"""Replay of a few old facts during fine-tuning (after Hinton & Plaut 1987, "Using fast weights to
deblur old memories": rehearsing a subset of old associations restores the unrehearsed ones).

THE PREDICTION. The collapse is a shift common to all old states, written through the shared key
direction mu. A replayed old fact's gradient writes along the same mu, against the shift, so
replaying a FEW old facts should prevent the collapse for ALL old facts, including the many that
are never replayed. Erosion is fact-specific (each fact's own key), so replay should protect only
the replayed facts: the never-replayed ones should still erode.

WHAT IS RUN. The base configuration of ingredients.py (beta = 0.5, RMS normalization, B's values
in half Y), gate-matched injection rate, ten seeds by default. A fixed random subset R of n_rep old
facts (A) is replayed: every step's batch is 32 B facts plus `per_step` facts drawn from R
(n_rep = 0 is the no-replay baseline). Logged per checkpoint: accuracy and probability-correct on
the never-replayed old facts (A \\ R), on R, on B and on the ballast D.

  JAX_PLATFORMS=cpu uv run --frozen python replay.py --reps 0,1,4,16 --seeds 0,1,2,3,4 --steps 3000
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

import kmin as m
from sweeps import match_gate


def inject_replay(p0, ilr, A, Dd, B, R, per_step, steps, seed, norm=1):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    rest = np.setdiff1d(np.arange(len(vA)), R)
    rng = np.random.default_rng(seed + 5)
    gi = set(m.grid(steps)); rows = []; p = p0
    for t in range(steps + 1):
        if t in gi:
            row = dict(step=t,
                       A_rest=m.accuracy(p, kA[rest], vA[rest], norm),
                       A_rest_p=m.prob_correct(p, kA[rest], vA[rest], norm),
                       A_rest_own=m.accuracy(p, kA[rest], vA[rest], norm, restrict=m.X),
                       B=m.accuracy(p, kB, vB, norm), D=m.accuracy(p, kD, vD, norm))
            if len(R):
                row.update(A_rep=m.accuracy(p, kA[R], vA[R], norm))
            rows.append(row)
        i = rng.integers(0, len(vB), 32)
        k, v = kB[i], vB[i]
        if len(R):
            j = R[rng.integers(0, len(R), per_step)]
            k, v = np.concatenate([k, kA[j]]), np.concatenate([v, vA[j]])
        p = m.step(p, jnp.asarray(k), jnp.asarray(v), ilr, 1.0, norm)
    return rows


def summarise(rows, key="A_rest"):
    st = np.array([r["step"] for r in rows]); a = np.array([r[key] for r in rows])
    i = int(a.argmin())
    return dict(trough=float(a[i]), trough_step=int(st[i]), after=float(a[i:].max()), end=float(a[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0,1,4,16", help="sizes of the replayed subset of old facts")
    ap.add_argument("--per_step", type=int, default=4, help="replayed facts per step (batch 32 + this)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", default="replay.json")
    a = ap.parse_args()
    out = {}
    for seed in [int(x) for x in a.seeds.split(",")]:
        p0, lr, pg, mu, A, Dd, B = m.pretrain(seed, a.beta, 1)
        ilr, g, ok = match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, seed, 200)
        for n in [int(x) for x in a.reps.split(",")]:
            R = np.random.default_rng(seed + 77).choice(len(A[1]), n, replace=False) if n else np.array([], int)
            rows = inject_replay(p0, ilr, A, Dd, B, R, a.per_step, a.steps, seed)
            q = summarise(rows)
            out[f"rep{n}-s{seed}"] = dict(n_rep=n, seed=seed, inject_lr=ilr, matched=bool(ok), rows=rows)
            print(f"seed {seed} replay {n:3d} | never-replayed old: trough {q['trough']:.3f}@{q['trough_step']} "
                  f"after {q['after']:.3f} end {q['end']:.3f} | replayed end "
                  f"{rows[-1].get('A_rep', float('nan')):.2f} | B end {rows[-1]['B']:.2f}", flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"\nwrote {a.out}\n\n{'replayed':>9} {'trough':>7} {'after':>7} {'end':>7} {'own@trough':>11} {'B end':>6}")
    for n in [int(x) for x in a.reps.split(",")]:
        rs = [v for v in out.values() if v["n_rep"] == n]
        qs = [summarise(r["rows"]) for r in rs]
        own = [r["rows"][[x["step"] for x in r["rows"]].index(q["trough_step"])]["A_rest_own"] for r, q in zip(rs, qs)]
        f = lambda k: np.mean([q[k] for q in qs])
        print(f"{n:9d} {f('trough'):7.3f} {f('after'):7.3f} {f('end'):7.3f} {np.mean(own):11.3f} "
              f"{np.mean([r['rows'][-1]['B'] for r in rs]):6.3f}")


if __name__ == "__main__":
    main()
