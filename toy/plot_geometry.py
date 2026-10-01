"""The geometry of the two processes, drawn: each old fact's displacement in state space.

    plots/toy_min_geometry.{png,pdf}

For every old fact a, the displacement of its state from where it started, d_a(t) = h_a(t) -
h_a(0) with h = kW (the space in which the split delta + eps_a is exact: delta = sqrt(beta)
mu dW, eps_a = sqrt(1-beta) g_a dW), projected onto two directions: the readout's own
"toward the new facts' answers" direction (mean unembedding of half Y minus half X, at step 0)
and one direction orthogonal to it (the leading direction of the individual displacements at
the end). One dot per fact, at four moments. The big marker is the mean displacement, the
common shift; the dots' spread around it is the individual part.

What the picture shows: at the trough the whole cloud has moved together toward the new
answers -- one vector, the same for every fact; in recovery the cloud comes back; at the end
the mean sits near where it started while the dots have scattered, each fact its own way.
The fast, shared motion is the suppression and its reversal; the slow scatter is the erosion.

Base configuration (d = 128, |V| = 32, beta = 0.5, rms readout), seed 0.

Run: JAX_PLATFORMS=cpu uv run python plot_geometry.py
"""
import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kmin as m

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30"
SEED, STEPS = 0, 5000


def main():
    p0, lr, gate, mu, A, Dd, B = m.pretrain(SEED, 0.5, 1)
    (kA, vA), _, (kB, vB) = A, Dd, B
    U0 = np.asarray(p0["U"])
    w = U0[:, m.Y].mean(1) - U0[:, m.X].mean(1); w /= np.linalg.norm(w)
    x0 = np.asarray(m.fwd(p0, jnp.asarray(kA), 1)[1])
    # the full trajectory, states kept at every checkpoint of the standard grid
    rng = np.random.default_rng(SEED + 5); ilr = lr / m.INJECT_RATIO
    grid = set(m.grid(STEPS)); p = p0; X, acc = {}, {}
    for t in range(STEPS + 1):
        if t in grid:
            X[t] = np.asarray(m.fwd(p, jnp.asarray(kA), 1)[1]) - x0
            acc[t] = m.accuracy(p, kA, vA, 1)
        i = rng.integers(0, m.NB, 32)
        p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, 1)
    st = np.array(sorted(X)); a = np.array([acc[t] for t in st])
    tr = int(st[a.argmin()]); pk = int(st[a.argmin() + a[a.argmin():].argmax()])
    early = int(st[np.searchsorted(st, tr // 3)])
    moments = ((early, "early"), (tr, "trough"), (pk, "recovered"), (STEPS, "end"))
    # the orthogonal axis: leading direction of the individual displacements at the end
    e_end = X[STEPS] - X[STEPS].mean(0); e_end -= np.outer(e_end @ w, w)
    _, _, vt = np.linalg.svd(e_end, full_matrices=False); q = vt[0]
    proj = lambda D: np.stack([D @ w, D @ q], 1)
    allP = np.concatenate([proj(X[t]) for t, _ in moments])
    xlo, xhi = min(-0.15 * allP[:, 0].max(), allP[:, 0].min() * 1.1), allP[:, 0].max() * 1.1
    ylim = np.abs(allP[:, 1]).max() * 1.15

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), dpi=200, sharex=True, sharey=True)
    cols = (NAVY, NAVY, TEAL, CRIM)
    for ax, (t, name), col in zip(axes, moments, cols):
        P = proj(X[t]); c = P.mean(0)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0); ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.scatter(P[:, 0], P[:, 1], s=9, color=col, alpha=0.45, lw=0, zorder=2)
        ax.annotate("", xy=c, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.4, shrinkA=0, shrinkB=3), zorder=3)
        ax.plot(*c, marker="o", ms=9, mfc=col, mec="white", mew=1.5, zorder=4)
        ax.plot(0, 0, marker="+", ms=9, color="0.2", mew=1.4, zorder=4)
        ax.set_title(f"{name}, step {t}   (old facts {acc[t]:.2f})", fontsize=10, color="0.25")
        ax.set_xlabel("toward the new facts' answers")
        ax.set_aspect("equal")
        d = X[t]; dm = d.mean(0); eps = d - dm
        print(f"{name:10s} step {t:5d} acc {acc[t]:.3f} | ||mean|| {np.linalg.norm(dm):.3f} "
              f"(on w {dm @ w:+.3f}) | rms individual {np.sqrt((eps ** 2).sum(1)).mean():.3f}")
    axes[0].set_ylabel("orthogonal direction")
    for ax in axes:
        ax.set_xlim(xlo, xhi); ax.set_ylim(-ylim, ylim)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_geometry.{e}"), bbox_inches="tight")
    print("wrote plots/toy_min_geometry")


if __name__ == "__main__":
    main()
