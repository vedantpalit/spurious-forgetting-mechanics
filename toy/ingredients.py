"""The minimal ingredients: which parts of the model the phenomenon needs.

Three ablations, each removing one ingredient and leaving everything else, plus the
training-time removal of the shared write. Every arm is gate-matched -- the injection rate is
bisected so the new facts reach 0.99 at the same step (200) in every arm -- because an arm in
which the new facts are never learned says nothing about recovery (the withdrawal needs their
confidence to turn positive). The matched rate and the achieved gate are recorded per run.

    readout     rms (the model) | ReLU | none              does the reversal need normalization?
    keys        beta = 0.5 (the model) | beta = 0           does it need a shared component?
    values      B in half Y (the model) | B over all values does it need a competing region?
    write       shared row of the store's update removed at every training step

Ten seeds per arm; writes ingredients.json.

  JAX_PLATFORMS=cpu uv run python ingredients.py --steps 5000 --seeds 0,1,2,...,9
"""
import argparse
import json

import numpy as np

import kmin as m
from sweeps import match_gate

ARMS = (
    # tag,            beta, norm, b_values, project,  label
    ("model",          0.5,    1,   "half", False, "the model"),
    ("relu",           0.5,    2,   "half", False, "ReLU instead of the normalization"),
    ("linear",         0.5,    0,   "half", False, "no normalization"),
    ("beta0",          0.0,    1,   "half", False, "no shared component in the keys"),
    ("allvalues",      0.5,    1,    "all", False, "new answers over the whole vocabulary"),
    ("projected",      0.5,    1,   "half",  True, "shared row of the write removed at every step"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--match_gate", type=int, default=200, help="0 disables the bisection")
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    ap.add_argument("--out", default="ingredients.json")
    a = ap.parse_args()
    want = set(a.arms.split(","))
    out = {}
    for tag, beta, norm, b_values, project, label in ARMS:
        if tag not in want:
            continue
        for seed in [int(x) for x in a.seeds.split(",")]:
            p0, lr, pg, mu, A, Dd, B = m.pretrain(seed, beta, norm, b_values=b_values)
            base = lr / m.INJECT_RATIO
            if a.match_gate:
                ilr, g, ok = match_gate(p0, A, Dd, B, norm, 1.0, base, seed, a.match_gate)
            else:
                ilr, g, ok = base, -1, True
            rows, _ = m.inject(p0, lr, mu, A, Dd, B, norm, 1.0, a.steps, seed,
                               ilr=ilr, project=project)
            r = dict(tag=tag, label=label, seed=seed, beta=beta, norm=norm, b_values=b_values,
                     project=project, pretrain_lr=lr, pretrain_gate=pg, inject_lr=ilr,
                     matched=bool(ok), b_gate=int(g), rows=rows)
            q = m.summarise(r)
            out[f"{tag}-s{seed}"] = r
            print(f"{tag:10s} seed {seed} | ilr {ilr:.2e} Bgate {q['B_gate']} matched={ok} "
                  f"| A {1.0:.2f} -> trough {q['trough']:.3f}@{q['trough_step']} -> after "
                  f"{q['after']:.3f} (rec {q['recovery']:+.3f}) end {q['end']:.3f} "
                  f"| B end {r['rows'][-1]['B']:.2f}", flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"\nwrote {a.out}")
    # one line per arm, means over seeds
    print(f"\n{'arm':10s} {'trough':>8} {'after':>8} {'recovery':>9} {'end':>7} {'B end':>7} {'B gate':>7}")
    for tag, *_ in ARMS:
        rs = [v for k, v in out.items() if v["tag"] == tag]
        if not rs:
            continue
        qs = [m.summarise(r) for r in rs]
        f = lambda k: np.mean([q[k] for q in qs])
        print(f"{tag:10s} {f('trough'):8.3f} {f('after'):8.3f} {f('recovery'):+9.3f} "
              f"{f('end'):7.3f} {np.mean([r['rows'][-1]['B'] for r in rs]):7.3f} "
              f"{np.mean([q['B_gate'] for q in qs]):7.0f}")


if __name__ == "__main__":
    main()
