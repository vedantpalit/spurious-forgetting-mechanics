"""The minimal associative memory: one matrix, one normalizer, one readout.

    h = k W                     W: (d, d)   the store
    z = rms(h) U                U: (d, V)   the readout; rms(x) = sqrt(d) x/||x||
    p = softmax(z)

The readout has three arms, set by `norm`: 1 = rms (the model), 2 = ReLU, 0 = none (z = h U,
the Zucchet toy). That is the only difference between them, and it is what separates a
monotone shift from one that is withdrawn (paper/math.tex): only the rms puts a term along
the new fact's own state into the gradient, with the sign of its confidence.

Against toy_final/k1: no skip connection, no gain, no store-input normalizer, no separate
readout rate -- W and U move at the same rate throughout, and both are initialised N(0, 1/d).
Everything else (keys, populations, the LR-probe pretraining, the injection ratio, the logged
observables) is the same, so the sweep and figure code carries over.

KEYS.  k_i = sqrt(beta) mu + sqrt(1-beta) g_i,  g_i a random unit vector, mu shared by all.
beta is the expected cosine between any two keys: the one relatedness knob.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

D, V = 128, 32
NA = ND = NB = 128            # the notebook's sizes. n_B sets the write's share of the state
                              # (there is no gain here): at n = 50 the share is ~0.5 and A
                              # barely crashes (0.84-0.96), at 128 it is ~0.8, the transformer's
                              # regime, and A crashes to 0.1-0.2 (factor test, 2026-09-16).


def configure(d=None, v=None, na=None, nd=None, nb=None):
    """Rebind the problem sizes. One process per configuration, called before the first
    traced call: the jitted functions close over D and V at trace time."""
    global D, V, NA, ND, NB, X, Y
    if d:
        D = int(d)
    if v:
        V = int(v)
        X = np.arange(0, V // 2)
        Y = np.arange(V // 2, V)
    if na:
        NA = int(na)
    if nd:
        ND = int(nd)
    if nb:
        NB = int(nb)


X = np.arange(0, V // 2)
Y = np.arange(V // 2, V)
GATE = 0.99
INJECT_RATIO = 13.33          # the transformer pretrain:inject learning-rate ratio
LR_GRID = [0.5, 0.3, 0.2, 0.1, 0.05]


def rms(h):
    return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def act(h, norm):
    """The readout's nonlinearity: 1 rms, 2 ReLU, 0 none."""
    if norm == 1:
        return rms(h)
    if norm == 2:
        return jax.nn.relu(h)
    return h


def build(seed, beta, b_values="half"):
    """A and the ballast D are pretrained; B is injected. All three share mu.

    `b_values="half"`: B's values lie in half Y, disjoint from A's half X -- the structured
    case. `"all"`: B's values are drawn from the whole vocabulary, so there is no region for
    a common shift to point at; the contrast with ordinary forgetting."""
    r = np.random.default_rng(seed)
    mu = r.normal(size=D); mu /= np.linalg.norm(mu)

    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(beta) * mu[None] + np.sqrt(1.0 - beta) * g

    return (mu,
            (keys(NA), X[r.integers(0, len(X), NA)]),
            (keys(ND), Y[r.integers(0, len(Y), ND)]),
            (keys(NB), (np.arange(V) if b_values == "all" else Y)[
                r.integers(0, V if b_values == "all" else len(Y), NB)]))


def init(seed):
    r = np.random.default_rng(seed + 77)
    return {"W": jnp.asarray(r.normal(size=(D, D)) / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}


@partial(jax.jit, static_argnames=("norm",))
def fwd(p, k, norm):
    h = k @ p["W"]
    z = act(h, norm) @ p["U"]
    return z, h


def loss_fn(p, k, v, norm):
    z, _ = fwd(p, k, norm)
    return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])


@partial(jax.jit, static_argnames=("norm",))
def step(p, k, v, lr, u_scale, norm):
    """u_scale multiplies the readout's rate; 1.0 (the default everywhere) is one rate."""
    g = jax.grad(loss_fn)(p, k, v, norm)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


def accuracy(p, k, v, norm, restrict=None):
    z = np.asarray(fwd(p, jnp.asarray(k), norm)[0])
    if restrict is not None:
        m = np.full(V, -np.inf); m[restrict] = 0.0
        z = z + m[None, :]
    return float((z.argmax(-1) == v).mean())


def mean_loss(p, k, v, norm):
    """Mean cross-entropy of the correct values."""
    return float(loss_fn(p, jnp.asarray(k), jnp.asarray(v), norm))


def prob_correct(p, k, v, norm):
    """Mean probability on the correct value: the smooth companion of accuracy (an argmax
    over 128 facts moves in steps of 1/128; this does not)."""
    z = np.asarray(fwd(p, jnp.asarray(k), norm)[0])
    pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    return float(pr[np.arange(len(v)), v].mean())


def margins(p, k, v, norm):
    """Per-individual confidence margin M = z_y - E_p[z], and the mass on half Y."""
    z = np.asarray(fwd(p, jnp.asarray(k), norm)[0])
    pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    return z[np.arange(len(v)), v] - (pr * z).sum(1), pr[:, Y].sum(1)


PRETRAIN_STEPS = 0            # 0: stop at the first 100-step check where A and D are >= GATE.
                              # n > 0: train exactly n steps (the notebook's fixed recipe) and
                              # require the gate at the end.


def pretrain(seed, beta, norm, cap=20000, b_values="half"):
    """LR probe, then pretrain A + ballast to the gate. Raises if no LR reaches it."""
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed, beta, b_values)
    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    for lr in LR_GRID:
        rng = np.random.default_rng(seed)
        p = init(seed)
        n = PRETRAIN_STEPS or cap
        for t in range(n):
            i = rng.integers(0, len(kp), 32)
            p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0, norm)
            if (t % 100 == 0 and not PRETRAIN_STEPS or t == n - 1) and \
                    min(accuracy(p, kA, vA, norm), accuracy(p, kD, vD, norm)) >= GATE:
                return p, lr, t, mu, (kA, vA), (kD, vD), (kB, vB)
    raise SystemExit(f"pretraining never reached {GATE} (beta={beta}, norm={norm}, seed={seed})")


def grid(steps):
    g = set(range(0, 201, 5))
    g |= {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(steps), 60)))}
    return sorted(s for s in g if s <= steps)


def inject(p0, lr, mu, A, Dd, B, norm, u_scale, steps, seed, ilr=None, project=False):
    """Fine-tune on B alone. Returns one row per checkpoint.

    `ilr` overrides the default lr/INJECT_RATIO; gate-matched sweeps pass the bisected rate.
    `project=True` removes the shared row of the store's update at every training step (not
    only at evaluation): W <- W0 + (I - mu mu^T) (W - W0). The write can then never contain a
    coherent component along the shared key direction, and whether B is still learned -- in
    the directions that remain -- is the stability/plasticity question."""
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
    _, hA0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA), norm)]
    postA0 = np.asarray(act(jnp.asarray(hA0), norm))
    w = U0[:, Y].mean(1) - U0[:, X].mean(1)
    w /= np.linalg.norm(w)                       # the X -> Y separation direction

    p = p0
    rng = np.random.default_rng(seed + 5)
    if ilr is None:
        ilr = lr / INJECT_RATIO
    gi = set(grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            _, hA = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), norm)]
            postA = np.asarray(act(jnp.asarray(hA), norm))
            W = np.asarray(p["W"]); U = np.asarray(p["U"]); dW = W - W0
            s = mu @ dW                                   # the shared write (the mu-row)
            dW_perp = dW - np.outer(mu, s)                # the write with its mu-row removed
            dpost = (postA - postA0).mean(0)
            eps = (postA - postA0) - dpost
            dh = hA - hA0; eps_h = dh - dh.mean(0)
            M, pY = margins(p, kB, vB, norm)
            rows.append(dict(
                step=t,
                A=accuracy(p, kA, vA, norm),
                A_own=accuracy(p, kA, vA, norm, restrict=X),
                Dacc=accuracy(p, kD, vD, norm), B=accuracy(p, kB, vB, norm),
                s_norm=float(np.linalg.norm(s)),
                s_share=float(np.linalg.norm(s) ** 2 / max(np.linalg.norm(dW) ** 2, 1e-30)),
                s_on_w=float(s @ w),                      # the write's X -> Y content
                dW=float(np.linalg.norm(dW)),
                dW_perp=float(np.linalg.norm(dW_perp)),
                dU_rel=float(np.linalg.norm(U - U0) / np.linalg.norm(U0)),
                delta=float(np.linalg.norm(dpost)),
                eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
                eps_h=float(np.sqrt((eps_h ** 2).sum(-1)).mean()),   # on the raw state h
                h_norm=float(np.linalg.norm(hA, axis=-1).mean()),
                post_norm=float(np.linalg.norm(postA, axis=-1).mean()),
                a_frac=float(np.linalg.norm(dpost) / np.linalg.norm(postA, axis=-1).mean()),
                M_mean=float(M.mean()), pY_mean=float(pY.mean()),
                A_loss=mean_loss(p, kA, vA, norm), B_loss=mean_loss(p, kB, vB, norm),
                A_p=prob_correct(p, kA, vA, norm), B_p=prob_correct(p, kB, vB, norm),
                D_p=prob_correct(p, kD, vD, norm),
            ))
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm)
        if project:
            dW = p["W"] - jnp.asarray(W0)
            dW = dW - jnp.outer(jnp.asarray(mu), jnp.asarray(mu) @ dW)
            p = {"W": jnp.asarray(W0) + dW, "U": p["U"]}
    return rows, p


def run(seed, beta, norm, u_scale, steps, b_values="half", project=False):
    p0, lr, gate, mu, A, Dd, B = pretrain(seed, beta, norm, b_values=b_values)
    rows, _ = inject(p0, lr, mu, A, Dd, B, norm, u_scale, steps, seed, project=project)
    return dict(seed=seed, beta=beta, norm=norm, u_scale=u_scale, b_values=b_values,
                project=project, pretrain_lr=lr, pretrain_gate=gate, rows=rows)


def summarise(r):
    rows = r["rows"]
    st = np.array([x["step"] for x in rows])
    A = np.array([x["A"] for x in rows]); B = np.array([x["B"] for x in rows])
    s = np.array([x["s_norm"] for x in rows])
    af = np.array([x["a_frac"] for x in rows])
    sw = np.array([x["s_on_w"] for x in rows])      # the observable that tracks accuracy
    tr = int(A.argmin()); pk = int(s.argmax()); pw = int(sw.argmax())
    bgate = int(st[np.argmax(B >= GATE)]) if (B >= GATE).any() else -1
    return dict(trough=float(A[tr]), trough_step=int(st[tr]),
                after=float(A[tr:].max()), end=float(A[-1]),
                recovery=float(A[tr:].max() - A[tr]),
                s_peak=float(s[pk]), s_peak_step=int(st[pk]), s_end=float(s[-1]),
                s_drop=float(1 - s[-1] / s[pk]) if s[pk] > 0 else 0.0,
                sw_peak=float(sw[pw]), sw_peak_step=int(st[pw]), sw_end=float(sw[-1]),
                sw_drop=float(1 - sw[-1] / sw[pw]) if sw[pw] > 1e-9 else 0.0,
                B_at_sw_peak=float(B[pw]),
                B_at_s_peak=float(B[pk]), B_gate=bgate, a_peak=float(af.max()),
                dU_rel=float(rows[-1]["dU_rel"]), A_own_end=float(rows[-1]["A_own"]),
                eps_end=float(rows[-1]["eps"]), D_end=float(rows[-1]["Dacc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.5")
    ap.add_argument("--norms", default="0,1")
    ap.add_argument("--u_scales", default="1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.d:
        configure(d=a.d)
    out = {}
    for beta in [float(x) for x in a.betas.split(",")]:
        for norm in [int(x) for x in a.norms.split(",")]:
            for us in [float(x) for x in a.u_scales.split(",")]:
                for seed in [int(x) for x in a.seeds.split(",")]:
                    r = run(seed, beta, norm, us, a.steps); q = summarise(r); r["d"] = D
                    out[f"b{beta}-n{norm}-u{us}-s{seed}"] = r
                    print(f"d={D} beta={beta} norm={norm} u={us} seed={seed} lr={r['pretrain_lr']} "
                          f"| A trough {q['trough']:.2f}@{q['trough_step']} -> after {q['after']:.2f} "
                          f"(rec {q['recovery']:+.2f}) end {q['end']:.2f} | s.w peak {q['sw_peak']:.3f}"
                          f"@{q['sw_peak_step']} end {q['sw_end']:.3f} ({-100 * q['sw_drop']:+.0f}%) "
                          f"| Bgate {q['B_gate']} | dU/U {q['dU_rel']:.3f} | a {q['a_peak']:.2f} "
                          f"| Aown {q['A_own_end']:.2f}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w")); print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
