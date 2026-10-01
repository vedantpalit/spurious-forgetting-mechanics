"""The OLMo weight-space patch, done on the minimal model.

At every checkpoint: dW = W(t) - W(0). Remove the top-r singular directions of dW from W(t)
(r = 1, 4), score A and B with the patched store; control: a random rank-r part of dW of
the same Frobenius norm removed instead. Also the theory's exact component: the mu-row of
dW removed (dW - mu mu^T dW), which is what the top-1 direction should be if the account is
right. The readout U is left as it is (as OLMo's lm_head was).

Base configuration, normalized arm, one rate, gate-matched at 200, ten seeds. Writes
patch_weights.json and plots/toy_min_patch_weights.{png,pdf}.
"""
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kmin as m
import sweeps

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.grid": False})
SEEDS = range(10)
STEPS = 5000
RANKS = (1, 4)


def patched(p, W):
    return {"W": jnp.asarray(W), "U": p["U"]}


def run(seed):
    p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, 0.5, 1)
    ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, seed, 200)
    (kA, vA), (kB, vB) = A, B
    W0 = np.asarray(p0["W"]); rng = np.random.default_rng(seed + 5); crng = np.random.default_rng(seed + 99)
    gi = set(m.grid(STEPS)); rows = []; p = p0
    for t in range(STEPS + 1):
        if t in gi:
            W = np.asarray(p["W"]); dW = W - W0
            row = dict(step=t, A=m.accuracy(p, kA, vA, 1), B=m.accuracy(p, kB, vB, 1))
            if t > 0:
                U_, S, Vt = np.linalg.svd(dW, full_matrices=False)
                for r in RANKS:
                    top = (U_[:, :r] * S[:r]) @ Vt[:r]
                    q = patched(p, W - top)
                    row[f"patch{r}/A"] = m.accuracy(q, kA, vA, 1); row[f"patch{r}/B"] = m.accuracy(q, kB, vB, 1)
                    a_ = crng.normal(size=(W.shape[0], r)); b_ = crng.normal(size=(W.shape[1], r))
                    rnd = a_ @ b_.T; rnd *= np.linalg.norm(top) / np.linalg.norm(rnd)
                    q = patched(p, W - rnd)
                    row[f"control{r}/A"] = m.accuracy(q, kA, vA, 1); row[f"control{r}/B"] = m.accuracy(q, kB, vB, 1)
                    row[f"top{r}_share"] = float((S[:r] ** 2).sum() / (S ** 2).sum())
                murow = np.outer(mu, mu @ dW)                      # the theory's component
                q = patched(p, W - murow)
                row["murow/A"] = m.accuracy(q, kA, vA, 1); row["murow/B"] = m.accuracy(q, kB, vB, 1)
                row["cos_top1_mu"] = float(abs(U_[:, 0] @ mu))    # is the top direction the mu-row?
            rows.append(row)
        if t < STEPS:
            i = rng.integers(0, m.NB, 32)
            p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, 1)
    return rows


def main():
    out = os.path.join(HERE, "patch_weights.json")
    if os.path.exists(out):
        data = json.load(open(out))
    else:
        data = {}
        for s in SEEDS:
            data[str(s)] = run(s); r = data[str(s)]
            tr = min(range(len(r)), key=lambda i: r[i]["A"])
            print(f"seed {s}: trough {r[tr]['A']:.2f}@{r[tr]['step']} | patch1 {r[tr]['patch1/A']:.2f} patch4 {r[tr]['patch4/A']:.2f} "
                  f"control4 {r[tr]['control4/A']:.2f} murow {r[tr]['murow/A']:.2f} | B actual {r[tr]['B']:.2f} patch1 {r[tr]['patch1/B']:.2f} "
                  f"| top1 share {r[tr]['top1_share']:.2f} cos(top1,mu) {r[tr]['cos_top1_mu']:.2f}", flush=True)
        json.dump(data, open(out, "w"))
    st = np.array([x["step"] for x in data["0"]], float)
    get = lambda key: np.array([[x.get(key, x["A"] if key.endswith("/A") else x["B"]) for x in data[s]] for s in data])

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    for key, col, ls, lw, lab in (("A", NAVY, "-", 2.0, "old facts"),
                                  ("patch4/A", OLIVE, "-", 2.0, "old facts, 4 directions removed from the update"),
                                  ("patch1/A", OLIVE, ":", 1.4, "old facts, 1 direction removed"),
                                  ("murow/A", TEAL, "--", 1.4, "old facts, the shared row ($\\mu$-row) removed"),
                                  ("control4/A", GREY, "--", 1.3, "old facts, random directions of the same size removed"),
                                  ("B", CRIM, ":", 2.0, "new facts"),
                                  ("patch4/B", CRIM, "-.", 1.3, "new facts, 4 directions removed")):
        Y = get(key); mu_, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, mu_ - sd, mu_ + sd, color=col, alpha=0.10, lw=0)
        ax.plot(st, mu_, color=col, ls=ls, lw=lw, label=lab)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, STEPS); ax.set_ylim(0, 1.03)
    ax.set_xlabel("injection step"); ax.set_ylabel("first-token accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_patch_weights.{e}"), bbox_inches="tight")
    A = get("A").mean(0); tr = int(A.argmin())
    for key in ("A", "patch1/A", "patch4/A", "murow/A", "control4/A", "B", "patch1/B", "patch4/B"):
        v = get(key).mean(0); print(f"{key:>11}: at trough({int(st[tr])}) {v[tr]:.3f}  at 200 {v[st == 200][0]:.3f}  end {v[-1]:.3f}")
    print(f"top-1 share of ||dW||^2 at the trough {get('top1_share').mean(0)[tr]:.2f}; |cos(top-1 direction, mu)| {get('cos_top1_mu').mean(0)[tr]:.2f}")
    print("wrote plots/toy_min_patch_weights")


if __name__ == "__main__":
    main()
