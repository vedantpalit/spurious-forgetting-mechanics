"""Long injection runs for the erosion law: is eps = c ln t + k over 10^4+ steps?

    python erosion_long.py --steps 30000 --u_scale 1.0 --out erosion_long.json

Base configuration, normalized arm, gate-matched at 200, three seeds; rows on kmin's grid
(dense to 200, then 60 log-spaced points). laws.py fits the tail.
"""
import argparse
import json

import kmin as m
import sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--norms", default="1")
    ap.add_argument("--out", default="erosion_long.json")
    a = ap.parse_args()
    out = {}
    for norm in [int(x) for x in a.norms.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, 0.5, norm)
            ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, a.u_scale, lr / m.INJECT_RATIO, seed, 200)
            rows, _ = m.inject(p0, lr, mu, A, Dd, B, norm, a.u_scale, a.steps, seed, ilr=ilr)
            out[f"n{norm}-s{seed}"] = dict(seed=seed, norm=norm, u_scale=a.u_scale, beta=0.5,
                                           inject_lr=ilr, B_gate=g, matched=bool(ok), rows=rows)
            r = rows[-1]
            print(f"norm {norm} seed {seed}: B gate {g} | A end {r['A']:.2f} own {r['A_own']:.2f} "
                  f"| eps {r['eps']:.3f} eps_h {r['eps_h']:.4f} | s.w {r['s_on_w']:.3f} | dU/U {r['dU_rel']:.3f}",
                  flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
