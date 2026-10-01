"""Data for the paper's first toy figure (plot_paper_base.py): the base configuration, both
readouts, gate-matched, ten seeds.

    python paper_base_runs.py --seeds 0-9 --out paper_base.json

The injection rate is bisected per seed and per arm so B reaches 0.99 at step 200, as in
every sweep figure; the depth spread that remains is instance spread, and ten seeds put the
standard error of the mean at ~0.05.
"""
import argparse
import json

import kmin as m
import sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--match", type=int, default=200)
    ap.add_argument("--norms", default="0,1")
    ap.add_argument("--out", default="paper_base.json")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    m.configure(d=128, v=32, na=128, nd=128, nb=128)

    out = {}
    for seed in range(lo, hi + 1):
        for norm in [int(x) for x in a.norms.split(",")]:
            p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, a.beta, norm)
            base = lr / m.INJECT_RATIO
            ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, a.u_scale, base, seed, a.match)
            rows, _ = m.inject(p0, lr, mu, A, Dd, B, norm, a.u_scale, a.steps, seed, ilr=ilr)
            key = f"b{a.beta}-n{norm}-u{a.u_scale}-s{seed}"
            out[key] = dict(seed=seed, beta=a.beta, norm=norm, u_scale=a.u_scale,
                            pretrain_lr=lr, pretrain_gate=gate, inject_lr=ilr, B_gate=g,
                            matched=bool(ok), rows=rows)
            s = m.summarise(out[key])
            print(f"seed {seed} norm {norm}: pre lr {lr} gate {gate} | B gate {g} (matched={ok}) "
                  f"| A trough {s['trough']:.2f}@{s['trough_step']} -> after {s['after']:.2f} "
                  f"end {s['end']:.2f} | s.w {s['sw_peak']:.2f}->{s['sw_end']:.2f} | dU/U {s['dU_rel']:.3f}",
                  flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
