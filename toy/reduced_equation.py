"""The scalar equation for the shared shift, checked on the paper's minimal model (kmin).

His report (full-batch GD, his notebook model) projects the mean-field write onto the direction
of the old facts' mean state and gets, with m = h_A_bar / s, s = ||h_A_bar||,

    ds/dt = eta * beta * mean_b  rho/||h_b|| * [ sin^2(th_b) G_b  -  sin(th_b) cos(th_b) M_b ]   (6)

    G_b = (U^T m)_{y_b}  - p_b^T U^T m        the readout of the SHARED direction: how much it
                                              favours b's correct answer over the current guess
    M_b = (U^T q_b)_{y_b} - p_b^T U^T q_b     the same for b's own INDIVIDUAL part q_b (the unit
                                              direction of h_b - (m.h_b) m): negative while b is
                                              confidently wrong, positive once it is right

and, without the normalization (z = h U), only the first kind of term survives:
    ds/dt = eta * beta * mean_b G_b                                                          (8)

Here each step of SGD is followed along the actual run: s is measured from the states, and the
prediction integrates the right-hand side of (6) / (8), evaluated on each step's minibatch at
the current parameters, from step 10 (once the mean has aligned with the write, as in the
report). Approximations in (6): the mean-field overlap k_b . k_A_bar ~ beta (costs ~0.5%) and
first order in the step -- a finite step also lengthens the mean by |D_perp|^2/2s, which is
the whole of the remaining gap (17% of the range with the normalization, 44% without, at this
rate); with that term the equation matches to <2%. It vanishes as eta -> 0. Checked per step
against a full-batch step: the vector update is exact (cos 1.0000). Base configuration,
gate-matched rate (the paper's).

Writes reduced_equation.json:  per seed and readout, per step: s measured, s predicted,
mean G, mean M, mean confidence margin, old/new-fact probability of the correct value.

  JAX_PLATFORMS=cpu uv run python reduced_equation.py --seeds 0,1,2,3,4 --steps 1000
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

import kmin as m
from sweeps import match_gate


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def rhs(W, U, kA, kB, vB, beta, norm):
    """Right-hand side of (6) (norm=1) or (8) (norm=0) per unit rate, plus the diagnostics, plus
    the exact expected change of the mean state (a vector), -mean_b (k_b . k_A_bar) r_b."""
    hA = kA @ W; hbar = hA.mean(0); s = np.linalg.norm(hbar); mh = hbar / s
    hB = kB @ W; nb = np.linalg.norm(hB, axis=1)
    x = m.D ** 0.5 * hB / nb[:, None] if norm == 1 else hB
    p = softmax(x @ U)
    Y = np.zeros_like(p); Y[np.arange(len(vB)), vB] = 1.0
    dlt = (p - Y) @ U.T                                      # (nB, d): dl/dx
    if norm == 1:
        hh = hB / nb[:, None]
        rvec = m.D ** 0.5 / nb[:, None] * (dlt - (dlt * hh).sum(1, keepdims=True) * hh)
    else:
        rvec = dlt
    c = kB @ kA.mean(0)                                      # exact overlaps, ~beta each
    dh_exact = -(c[:, None] * rvec).mean(0)
    Um = U.T @ mh                                            # (V,) readout of the shared direction
    G = Um[vB] - p @ Um
    cos = hB @ mh / nb; sin = np.sqrt(np.clip(1 - cos ** 2, 0, None))
    perp = hB - np.outer(hB @ mh, mh)
    q = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-12)
    Uq = q @ U                                               # (nB, V)
    M = Uq[np.arange(len(vB)), vB] - (p * Uq).sum(1)
    z = x @ U
    conf = z[np.arange(len(vB)), vB] - (p * z).sum(1)        # the confidence margin z_y - E_p z
    if norm == 1:
        r = m.D ** 0.5 / nb * (sin ** 2 * G - sin * cos * M)
    else:
        r = G
    return beta * r.mean(), s, G.mean(), M.mean(), conf.mean(), dh_exact, hbar


def run(seed, norm, steps, beta=0.5, t0=10):
    p0, lr, pg, mu, A, Dd, B = m.pretrain(seed, beta, norm)
    ilr, g, ok = match_gate(p0, A, Dd, B, norm, 1.0, lr / m.INJECT_RATIO, seed, 200)
    (kA, vA), _, (kB, vB) = A, Dd, B
    rng = np.random.default_rng(seed + 5)
    p = p0; rows = []; s_pred = None; s_pred2 = None; s_pred0 = None; hvec = None
    for t in range(steps + 1):
        W = np.asarray(p["W"]); U = np.asarray(p["U"])
        f, s, Gm, Mm, cm, dh, hbar = rhs(W, U, kA, kB, vB, beta, norm)
        if s_pred0 is None:
            s_pred0 = s; hvec = hbar.copy()
        if t == t0:
            s_pred = s; s_pred2 = s           # the scalar equation, from once the mean has aligned
        if t % 5 == 0:
            rows.append(dict(step=t, s=float(s), s_pred=float(s_pred if s_pred is not None else s),
                             s_pred2=float(s_pred2 if s_pred2 is not None else s),
                             s_pred0=float(s_pred0), s_vec=float(np.linalg.norm(hvec)),
                             G=float(Gm), M=float(Mm),
                             conf=float(cm), A_p=m.prob_correct(p, kA, vA, norm),
                             B_p=m.prob_correct(p, kB, vB, norm)))
        i = rng.integers(0, m.NB, 32)
        # the equation evaluated on the minibatch the model actually trains on at this step, so
        # the prediction follows the same SGD trajectory. s_pred: eq (6) as written (gradient
        # flow). s_pred2: eq (6) plus the second-order term of a finite step, |D_perp|^2 / 2s --
        # the part of each step perpendicular to the mean lengthens it; it vanishes as eta -> 0,
        # and it is the whole of the gap between (6) and the measured s at this rate.
        fb, *_rest, dhb, _h = rhs(W, U, kA, kB[i], vB[i], beta, norm)
        Dv = ilr * dhb; mh = hbar / s; Dp = Dv - (Dv @ mh) * mh
        s_pred0 += ilr * fb
        if s_pred is not None:
            s_pred += ilr * fb
            s_pred2 += ilr * fb + (Dp @ Dp) / (2 * s)
        hvec = hvec + Dv
        p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, norm)
    return dict(seed=seed, norm=norm, inject_lr=ilr, matched=bool(ok), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--norms", default="1,0")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default="reduced_equation.json")
    a = ap.parse_args()
    out = {}
    for norm in [int(x) for x in a.norms.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = run(seed, norm, a.steps); out[f"n{norm}-s{seed}"] = r
            st = np.array([x["step"] for x in r["rows"]]); s = np.array([x["s"] for x in r["rows"]])
            sp = np.array([x["s_pred"] for x in r["rows"]]); M = np.array([x["M"] for x in r["rows"]])
            sv = np.array([x["s_vec"] for x in r["rows"]])
            ev = np.abs(sv - s).max() / (s.max() - s.min() + 1e-12)
            conf = np.array([x["conf"] for x in r["rows"]])
            k = int(s.argmax()); zc = int(np.argmax(conf > 0)) if (conf > 0).any() else -1
            zm = int(np.argmax(M > 0)) if (M > 0).any() else -1
            err = np.abs(sp - s).max() / (s.max() - s.min() + 1e-12)
            sp2 = np.array([x["s_pred2"] for x in r["rows"]])
            err2 = np.abs(sp2 - s).max() / (s.max() - s.min() + 1e-12)
            print(f"norm={norm} seed={seed} | s {s[0]:.3f} -> peak {s[k]:.3f}@{st[k]} -> end {s[-1]:.3f} "
                  f"| pred peak {sp.max():.3f}@{st[sp.argmax()]} end {sp[-1]:.3f} | max |pred-meas| "
                  f"{100 * err:.1f}% of range, with the step term {100 * err2:.1f}% (vector, exact overlaps: {100 * ev:.1f}%, end {sv[-1]:.3f}) | M>0 from {st[zm] if zm >= 0 else '-'}, "
                  f"conf>0 from {st[zc] if zc >= 0 else '-'}", flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
