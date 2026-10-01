"""Numerical verification of paper/math.tex on the minimal model. Every claim is checked
against autodiff or against the measured trajectory; nothing is asserted.

THE MODEL.  h = k W,  z = rms(h) U  (or z = h U in the linear arm). Row-vector convention
here (k is a row, h = k @ W), so math.tex's column-vector statements transpose.

CLAIM 1 -- the common shift on A is the mu-row of Delta W.
    Delta h_a = k_a Delta W and k_a = sqrt(b) mu + sqrt(1-b) g_a, so
        mean_a Delta h_a = sqrt(b) (mu Delta W) + O(1/sqrt(n_A))       [exact: k_bar_A Delta W]
CLAIM 2 -- the gradient identity for the shared row, against jax.grad:
        mu grad_W L = sum_b <mu, k_b> dL/dh_b                            (exact)
CLAIM 3 -- the normalizer's split (math.tex eq. split):
        dL/dh_b = (sqrt(d)/||h_b||) (p_b - e_b) U^T / n_B  +  (M_b / (n_B ||h_b||)) h_hat_b
    with M_b the confidence margin; the linear arm has only the first term.
CLAIM 4 -- the sign: <ds/dt, s_hat> never negative in the linear arm; changes sign in the
    normalized arm, at B's onset.
CLAIM 5 -- the timing: the normalizer's term flips sign when B's mean margin crosses zero;
    the total turns when the term outgrows the plain term.
CLAIM 6 -- the removing force is proportional to the shift (math.tex, "The removing force
    is proportional to the shift"): mean_b <h_b - h_b^0, w_hat> = sqrt(b) <s, w_hat>, and
    sum_b M_b C_b / n_b has the sign of the mean margin.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

import kmin as m
import sweeps

jax.config.update("jax_enable_x64", True)


def manual_dLdh(p, kB, vB, norm):
    """dL/dh for every B individual, plus the two-term split of the normalized case."""
    z, h = [np.asarray(x) for x in m.fwd(p, jnp.asarray(kB), norm)]
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), axis=-1))
    e = np.zeros_like(pr); e[np.arange(len(vB)), vB] = 1.0
    U = np.asarray(p["U"])
    y = (pr - e) @ U.T / len(vB)                                  # (n_B, d)
    margin = z[np.arange(len(vB)), vB] - (pr * z).sum(-1)         # M_b, both arms
    if not norm:
        return y, y, np.zeros_like(y), h, margin
    hn = np.linalg.norm(h, axis=-1, keepdims=True)
    hhat = h / hn
    term_a = np.sqrt(m.D) / hn * y
    term_b = (margin[:, None] / hn) * hhat / len(vB)
    return term_a + term_b, term_a, term_b, h, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--match", type=int, default=200, help="0 = fixed rate lr/INJECT_RATIO")
    ap.add_argument("--norms", default="1,0")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    saved = {}

    for norm in [int(x) for x in a.norms.split(",")]:
        p0, lr, gate, mu, (kA, vA), (kD, vD), (kB, vB) = m.pretrain(a.seed, a.beta, norm)
        W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
        _, hA0 = [np.asarray(x) for x in m.fwd(p0, jnp.asarray(kA), norm)]
        _, hB0 = [np.asarray(x) for x in m.fwd(p0, jnp.asarray(kB), norm)]
        kbar = kA.mean(0)
        w = U0[:, m.Y].mean(1) - U0[:, m.X].mean(1); w /= np.linalg.norm(w)

        print(f"\n{'='*100}\nARM: {'normalized readout' if norm else 'linear (no norm)'}   "
              f"beta={a.beta} seed={a.seed} pretrain lr={lr} gate@{gate}\n{'='*100}")
        ilr = lr / m.INJECT_RATIO
        if a.match:
            ilr, g, ok = sweeps.match_gate(p0, (kA, vA), (kD, vD), (kB, vB), norm, a.u_scale,
                                           ilr, a.seed, a.match)
            print(f"  gate-matched: rate {ilr:.4g}, B gate at {g} (target {a.match}, matched={ok})")
        print(f"{'step':>6} {'B acc':>6} {'||s||':>7} {'c1 err':>9} {'c2 err':>9} {'c3 err':>9} "
              f"{'<ds,s>':>10} {'  plain':>10} {'  norm term':>11} {'margin':>8} {'share/pred':>10}")

        p = p0
        rng = np.random.default_rng(a.seed + 5)
        gi = set(m.grid(a.steps))
        c1, c2, c3, sgn, rec = [], [], [], [], []
        for t in range(a.steps + 1):
            if t in gi:
                dW = np.asarray(p["W"]) - W0
                s = mu @ dW; sn = np.linalg.norm(s); shat = s / max(sn, 1e-30)

                _, hA = [np.asarray(x) for x in m.fwd(p, jnp.asarray(kA), norm)]
                meas = (hA - hA0).mean(0)
                err1 = np.linalg.norm(meas - kbar @ dW) / max(np.linalg.norm(meas), 1e-30)

                gW = np.asarray(jax.grad(m.loss_fn)(p, jnp.asarray(kB), jnp.asarray(vB), norm)["W"])
                auto = mu @ gW
                dLdh, ta, tb, hB, margin = manual_dLdh(p, kB, vB, norm)
                coef = kB @ mu
                exact = (coef[:, None] * dLdh).sum(0)
                err2 = np.linalg.norm(auto - exact) / max(np.linalg.norm(auto), 1e-30)
                err3 = np.linalg.norm(dLdh - (ta + tb)) / max(np.linalg.norm(dLdh), 1e-30)

                force = -ilr * exact
                fa = -ilr * (coef[:, None] * ta).sum(0)
                fb = -ilr * (coef[:, None] * tb).sum(0)
                proj, pa, pb = force @ shat, fa @ shat, fb @ shat
                # math.tex eq. force is the projection on w_hat (the between-half axis), where
                # the plain term is the writing force Phi_b >= 0; logged next to the s_hat one.
                projw, paw, pbw = force @ w, fa @ w, fb @ w

                # claim 6
                hn = np.linalg.norm(hB, axis=-1); C = (hB / hn[:, None]) @ w
                share_meas = float(((hB - hB0) @ w).mean())
                share_pred = float(np.sqrt(a.beta) * (s @ w))
                rem_force = float((margin * C / hn).sum())

                c1.append(err1); c2.append(err2); c3.append(err3)
                if sn > 1e-9:
                    sgn.append(proj)
                rec.append(dict(step=t, A=m.accuracy(p, kA, vA, norm), B=m.accuracy(p, kB, vB, norm),
                                margin=float(margin.mean()), s_norm=float(sn), s_on_w=float(s @ w),
                                proj=float(proj), proj_plain=float(pa), proj_anti=float(pb),
                                proj_w=float(projw), proj_w_plain=float(paw), proj_w_anti=float(pbw),
                                C_mean=float(C.mean()), share_meas=share_meas,
                                share_pred=share_pred, rem_force=rem_force))
                if t in (0, 5, 10, 20, 40, 60, 80, 100, 150, 200) or t in (
                        [x for x in sorted(gi) if x > 200][::10]):
                    ratio = share_meas / share_pred if abs(share_pred) > 1e-9 else float("nan")
                    print(f"{t:>6} {rec[-1]['B']:>6.2f} {sn:>7.3f} {err1:>9.2e} {err2:>9.2e} "
                          f"{err3:>9.2e} {proj:>+10.3e} {pa:>+10.3e} {pb:>+11.3e} "
                          f"{margin.mean():>8.3f} {ratio:>10.3f}")
            i = rng.integers(0, m.NB, 32)
            p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, a.u_scale, norm)

        sg = np.array(sgn)
        print(f"\n  claim 2  gradient identity vs autodiff          max rel err {max(c2):.2e}")
        print(f"  claim 3  two-term split of the normalized grad  max rel err {max(c3):.2e}")
        print(f"  claim 1  common shift is k_bar_A's row of dW    max rel err {max(c1):.2e}")
        print(f"  claim 4  sign of <ds/dt, s_hat>: min {sg.min():+.3e}, "
              f"negative at {100 * (sg < 0).mean():.0f}% of checkpoints")
        pw = np.array([r["proj_w"] for r in rec if r["step"] > 0])
        pwp = np.array([r["proj_w_plain"] for r in rec if r["step"] > 0])
        print(f"  claim 4w on w_hat (math.tex eq. force): total negative at {100 * (pw < 0).mean():.0f}%, "
              f"plain/writing term negative at {100 * (pwp < 0).mean():.0f}% (min {pwp.min():+.2e})")

        def cross(key, rising):
            rows = [r for r in rec if r["step"] > 0]
            for i in range(1, len(rows)):
                a0, b0 = rows[i - 1][key], rows[i][key]
                if (rising and a0 < 0 <= b0) or (not rising and a0 > 0 >= b0):
                    return rows[i]["step"]
            return None

        mcross = cross("margin", True); pcross = cross("proj", False); acrs = cross("proj_anti", False)
        print(f"  claim 5  B's mean margin crosses 0 at {mcross}; normalizer's term at {acrs}; "
              f"total <ds/dt, s_hat> at {pcross}")
        if norm:
            sm = np.array([r["share_meas"] for r in rec]); sp = np.array([r["share_pred"] for r in rec])
            k = np.abs(sp) > 1e-6
            mg = np.array([r["margin"] for r in rec]); rf = np.array([r["rem_force"] for r in rec])
            print(f"  claim 6  share/pred mean {np.mean(sm[k] / sp[k]):.3f} sd {np.std(sm[k] / sp[k]):.3f}; "
                  f"removing force has the margin's sign at {100 * (np.sign(mg) == np.sign(rf)).mean():.0f}%")
        saved[f"n{norm}-s{a.seed}"] = dict(rows=rec, margin_cross=mcross, proj_cross=pcross,
                                           anti_cross=acrs, inject_lr=ilr)
    if a.save:
        json.dump(saved, open(a.save, "w"))
        print(f"\n  wrote {a.save}")


if __name__ == "__main__":
    main()
