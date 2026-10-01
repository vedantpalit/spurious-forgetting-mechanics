"""Hyperparameter sweeps over the minimal toy, with B's clock matched across columns.

WHY GATE MATCHING. Every axis here (d, n_B, load, vocabulary, beta) changes how fast B is
learned at a fixed nominal rate. The crash is caused by B's acquisition, so a column that
learns B faster gets a different dose, not just a different geometry, and crash depths are
then not comparable. For every configuration the injection rate is bisected until B reaches
0.99 at a fixed target step. The dose sweep is the deliberate exception: it varies the rate
on purpose, to test whether the curve collapses on B's accuracy axis.

AXES
  d       key/residual dimension
  nb      number of injected individuals
  load    n_A = n_D, at fixed n_B
  vocab   |V|
  beta    shared key fraction
  dose    injection-rate multiplier       (NOT gate matched, by design)

Each cell: pretrain once (learning-rate probe), bisect, then one full injection run.
Same driver as toy_final/k1/sweeps.py with the model swapped for kmin (no gain argument).
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

import kmin as m


def gate_of(p0, A, Dd, B, norm, u_scale, ilr, seed, cap=4000, every=5):
    """Steps until B first reaches the gate, or `cap` if it never does."""
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    p = p0
    rng = np.random.default_rng(seed + 5)
    for t in range(cap + 1):
        if t % every == 0 and m.accuracy(p, kB, vB, norm) >= m.GATE:
            return t
        i = rng.integers(0, m.NB, 32)
        p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm)
    return cap


def match_gate(p0, A, Dd, B, norm, u_scale, base_ilr, seed, target, iters=11, tol=0.15):
    """Bisect the injection rate in log space so B's gate lands on `target`.
    Returns (rate, achieved gate, matched?)."""
    lo, hi = -6.0, 6.0
    best = (base_ilr, gate_of(p0, A, Dd, B, norm, u_scale, base_ilr, seed), False)
    if abs(best[1] - target) <= tol * target:
        return best[0], best[1], True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ilr = base_ilr * 2.0 ** mid
        g = gate_of(p0, A, Dd, B, norm, u_scale, ilr, seed)
        if abs(g - target) < abs(best[1] - target):
            best = (ilr, g, abs(g - target) <= tol * target)
        if best[2]:
            break
        if g > target:
            lo = mid
        else:
            hi = mid
    return best


AXES = {"d": "d", "nb": "n_B", "load": "n_A = n_D", "vocab": "|V|",
        "beta": "beta", "dose": "injection-rate multiplier"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=sorted(AXES))
    ap.add_argument("--values", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--na", type=int, default=128)
    ap.add_argument("--nb", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--norms", default="1,0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--match_gate", type=int, default=200, help="0 disables matching")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vals = [float(x) for x in a.values.split(",")]
    out = {}
    for val in vals:
        cfg = dict(d=a.d, v=a.vocab, na=a.na, nd=a.na, nb=a.nb)
        beta, dose = a.beta, 1.0
        if a.sweep == "d":
            cfg["d"] = int(val)
        elif a.sweep == "nb":
            cfg["nb"] = int(val)
        elif a.sweep == "load":
            cfg["na"] = cfg["nd"] = int(val)
        elif a.sweep == "vocab":
            cfg["v"] = int(val)
        elif a.sweep == "beta":
            beta = val
        elif a.sweep == "dose":
            dose = val
        m.configure(**cfg)

        for norm in [int(x) for x in a.norms.split(",")]:
            for seed in [int(x) for x in a.seeds.split(",")]:
                try:
                    p0, lr, pg, mu, A, Dd, B = m.pretrain(seed, beta, norm)
                except SystemExit as e:
                    print(f"{a.sweep}={val} norm={norm} seed={seed}  PRETRAIN FAILED: {e}", flush=True)
                    continue
                base = lr / m.INJECT_RATIO
                if a.match_gate:
                    ilr, g, ok = match_gate(p0, A, Dd, B, norm, a.u_scale, base, seed, a.match_gate)
                else:
                    ilr, g, ok = base * dose, -1, True
                rows, _ = m.inject(p0, lr, mu, A, Dd, B, norm, a.u_scale, a.steps, seed, ilr=ilr)
                r = dict(sweep=a.sweep, value=val, norm=norm, seed=seed, beta=beta,
                         d=m.D, v=m.V, na=m.NA, nd=m.ND, nb=m.NB, u_scale=a.u_scale,
                         pretrain_lr=lr, pretrain_gate=pg, inject_lr=ilr, matched=bool(ok),
                         probe_gate=g, dose=dose, rows=rows)
                q = m.summarise(r)
                out[f"{a.sweep}{val}-n{norm}-s{seed}"] = r
                flag = "" if ok else "  [GATE NOT MATCHED]"
                print(f"{a.sweep}={val:g} norm={norm} seed={seed} | ilr {ilr:.2e} Bgate {q['B_gate']} "
                      f"| A trough {q['trough']:.2f}@{q['trough_step']} -> after {q['after']:.2f} "
                      f"(rec {q['recovery']:+.2f}) end {q['end']:.2f} | s.w {q['sw_peak']:.3f}->"
                      f"{q['sw_end']:.3f} ({-100 * q['sw_drop']:+.0f}%) | eps {q['eps_end']:.2f} "
                      f"| a {q['a_peak']:.2f} | dU/U {q['dU_rel']:.3f} | Aown {q['A_own_end']:.2f}{flag}",
                      flush=True)
        json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out}  ({len(out)} runs)")


if __name__ == "__main__":
    main()
