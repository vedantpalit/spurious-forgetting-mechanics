"""Where the keys land in the hidden state (2026-09-16).

h_i = W k_i = sqrt(beta) W mu + sqrt(1-beta) W g_i. Measured at pretrain end, at A's trough
and at the end of injection, base configuration, normalized arm, one learning rate:

    common share        ||proj of h_i on c_hat||^2 / ||h_i||^2,   c = W mu
    cos within / between  mean pairwise cosine of the states inside A, inside B, A vs B
    own-value alignment  cos(h_i - common part, u_{y_i})           the stored fact
    tilt                 <mean_pop h, w_hat>, w = u_bar_Y - u_bar_X   where each cloud sits on
                         the between-half axis
Three seeds; also writes plots/toy_min_hidden_geometry: the three clouds projected on
(c_hat, w_hat) at the three times.
"""
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kmin as m
import sweeps

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, OLIVE = "#1B2A4E", "#C4245F", "#8A8C30"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10, "axes.grid": False})


def pair_cos(H, G=None):
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    if G is None:
        C = Hn @ Hn.T; return C[np.triu_indices(len(H), 1)].mean()
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    return (Hn @ Gn.T).mean()


def stats(p, mu, pops, w, beta):
    W = np.asarray(p["W"]); U = np.asarray(p["U"])
    c = mu @ W; chat = c / np.linalg.norm(c)
    out = {}
    for name, (k, v) in pops.items():
        h = k @ W
        common = (h @ chat)[:, None] * chat[None]
        ind = h - common
        out[name] = dict(common_share=float(((h @ chat) ** 2 / (h ** 2).sum(1)).mean()),
                         own_align=float(np.mean([ind[i] @ U[:, v[i]] / np.linalg.norm(ind[i]) / np.linalg.norm(U[:, v[i]]) for i in range(len(v))])),
                         tilt=float(h.mean(0) @ w), norm=float(np.linalg.norm(h, axis=1).mean()),
                         within=float(pair_cos(h)), h=h)
    out["A_vs_B"] = float(pair_cos(pops["A"][0] @ W, pops["B"][0] @ W))
    out["A_vs_D"] = float(pair_cos(pops["A"][0] @ W, pops["D"][0] @ W))
    return out, chat


def main():
    beta = 0.5; norm = 1
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=200, sharex=True, sharey=True)
    for seed in (0, 1, 2):
        p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, beta, norm)
        ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, 1.0, lr / m.INJECT_RATIO, seed, 200)
        U0 = np.asarray(p0["U"]); w = U0[:, m.Y].mean(1) - U0[:, m.X].mean(1); w /= np.linalg.norm(w)
        pops = {"A": A, "D": Dd, "B": B}
        rows, _ = m.inject(p0, lr, mu, A, Dd, B, norm, 1.0, 5000, seed, ilr=ilr)
        st = np.array([r["step"] for r in rows]); Acc = np.array([r["A"] for r in rows])
        t_tr = int(st[Acc.argmin()])
        snaps = {}
        # re-run to the snapshot steps (inject returns only the final params)
        p = p0; rng = np.random.default_rng(seed + 5); (kB, vB) = B
        for t in range(5001):
            if t in (0, t_tr, 5000):
                snaps[t] = stats(p, mu, pops, w, beta)
            if t < 5000:
                i = rng.integers(0, m.NB, 32)
                p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, norm)
        print(f"\nseed {seed}: trough at {t_tr} (A {Acc.min():.2f}), end A {Acc[-1]:.2f}")
        print(f"  {'time':>10} {'pop':>3} {'||h||':>6} {'common share':>12} {'within cos':>10} {'own-value cos':>13} {'tilt on w':>9}")
        for t, (S, chat) in snaps.items():
            lab = {0: "pretrained", t_tr: f"trough@{t_tr}", 5000: "end@5000"}[t]
            for name in ("A", "D", "B"):
                s = S[name]
                print(f"  {lab:>10} {name:>3} {s['norm']:6.2f} {s['common_share']:12.3f} {s['within']:10.3f} "
                      f"{s['own_align']:13.3f} {s['tilt']:9.3f}")
            print(f"  {'':>10} A vs B cos {S['A_vs_B']:.3f}, A vs D cos {S['A_vs_D']:.3f}")
        if seed == 0:
            for ax, (t, (S, chat)) in zip(axes, snaps.items()):
                for name, col in (("A", NAVY), ("D", OLIVE), ("B", CRIM)):
                    h = S[name]["h"]
                    ax.scatter(h @ chat, h @ w, s=6, color=col, alpha=0.6, label=name, lw=0)
                ax.set_title({0: "pretrained", t_tr: f"A's trough (step {t_tr})", 5000: "end (step 5000)"}[t], fontsize=10)
                ax.set_xlabel("along $\\hat c = W\\mu/\\|W\\mu\\|$")
                ax.axhline(0, color="0.7", lw=0.6)
            axes[0].set_ylabel("along $\\hat w$ (X $\\to$ Y)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", "toy_plots_working", f"toy_min_hidden_geometry.{e}"), bbox_inches="tight")
    print("\nwrote plots/toy_plots_working/toy_min_hidden_geometry")


if __name__ == "__main__":
    main()
