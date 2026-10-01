"""Figure 1, right panel: the two processes, drawn and measured.

Top: a schematic. Old representations as dots; at the trough every dot has received the SAME
arrow (the common shift, teal) plus a small arrow of its own (the individual displacement,
olive); afterwards the common arrow has been withdrawn and the individual one remains.

Bottom: the accuracy each process would produce on its own, measured on the minimal model.
At every checkpoint, with z_a(0) the pretrained logits of old fact a and z_a(t) the current
ones, dz_bar(t) = mean_a [z_a(t) - z_a(0)] is the common part of the logit change and
dz_a(t) - dz_bar(t) the individual part:

    common only      argmax( z_a(0) + dz_bar(t) )            == y_a
    individual only  argmax( z_a(0) + dz_a(t) - dz_bar(t) )  == y_a
    both (actual)    argmax( z_a(t) )                        == y_a

Base configuration, normalized arm, one learning rate, gate-matched at 200, ten seeds.
Writes fig1_two_processes.json and plots/fig1_two_processes.{png,pdf}.
"""
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import kmin as m
import sweeps

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.62"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10, "axes.grid": False})
SEEDS = range(10)


def measure(seed, steps=5000):
    p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, 0.5, 1)
    ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, seed, 200)
    (kA, vA), (kB, vB) = A, B
    z0 = np.asarray(m.fwd(p0, jnp.asarray(kA), 1)[0])
    p = p0; rng = np.random.default_rng(seed + 5); gi = set(m.grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            z = np.asarray(m.fwd(p, jnp.asarray(kA), 1)[0])
            dz = z - z0; dzb = dz.mean(0)
            rows.append(dict(step=t,
                             actual=float((z.argmax(1) == vA).mean()),
                             common=float(((z0 + dzb).argmax(1) == vA).mean()),
                             individual=float(((z0 + dz - dzb).argmax(1) == vA).mean()),
                             B=m.accuracy(p, kB, vB, 1)))
        if t < steps:
            i = rng.integers(0, m.NB, 32)
            p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, 1)
    return rows


def schematic(ax):
    ax.set_xlim(0, 10.7); ax.set_ylim(-0.55, 2.35); ax.axis("off")
    rng = np.random.default_rng(3)
    xs = np.linspace(0.6, 2.6, 5)
    ind = rng.normal(size=(5, 2)); ind /= np.linalg.norm(ind, axis=1, keepdims=True); ind *= 0.32
    common = np.array([0.55, 0.75])
    for col, (x0, title) in enumerate(((0.0, "before"), (3.5, "at the trough"), (7.3, "after"))):
        ax.text(x0 + 1.6, 2.15, title, ha="center", va="center", fontsize=9, color="0.25")
        for i, x in enumerate(xs):
            base = np.array([x0 + x, 0.75])
            ax.plot(*base, "o", color=NAVY, ms=4.5, zorder=3)
            if col >= 1:
                a = ind[i] * (1.0 if col == 1 else 1.45)
                ax.add_patch(FancyArrowPatch(base, base + a, color=OLIVE, lw=1.4, arrowstyle="-|>",
                                             mutation_scale=7, zorder=2))
            if col == 1:
                ax.add_patch(FancyArrowPatch(base, base + common, color=TEAL, lw=1.6, arrowstyle="-|>",
                                             mutation_scale=8, zorder=2))
            if col == 2:
                ax.add_patch(FancyArrowPatch(base, base + common, color=TEAL, lw=1.0, ls=(0, (2, 2)),
                                             alpha=0.45, arrowstyle="-|>", mutation_scale=7, zorder=1))
    ax.text(3.5 + 1.6, 0.0, "the same shift on every fact", ha="center", fontsize=7, color=TEAL)
    ax.text(3.5 + 1.6, -0.35, "+ a displacement of its own", ha="center", fontsize=7, color=OLIVE)
    ax.text(7.3 + 1.6, 0.0, "the shift withdrawn", ha="center", fontsize=7, color=TEAL)
    ax.text(7.3 + 1.6, -0.35, "the displacement kept", ha="center", fontsize=7, color=OLIVE)


def main():
    out = os.path.join(HERE, "fig1_two_processes.json")
    if os.path.exists(out):
        data = json.load(open(out))
    else:
        data = {str(s): measure(s) for s in SEEDS}
        json.dump(data, open(out, "w"))
    st = np.array([r["step"] for r in data["0"]], float)
    get = lambda k: np.array([[r[k] for r in data[str(s)]] for s in SEEDS])
    for k in ("actual", "common", "individual"):
        v = get(k).mean(0); tr = int(v.argmin())
        print(f"{k:>10}: trough {v[tr]:.3f}@{int(st[tr])}  max after {v[tr:].max():.3f}  end {v[-1]:.3f}")

    # Two files: the curves panel, matched to the OLMo panel (llm/plot_seeds.py --paper: same
    # figure size, type size, linear x to 1000, y to 1.14, no y label since the left panel has
    # it), and the schematic as a wide strip to sit under both panels.
    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.0,
                         "xtick.direction": "out", "ytick.direction": "out"})
    XMAX = 1000
    keep = st <= 1.15 * XMAX; x = st[keep]        # one grid point past XMAX, clipped by xlim
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    for k, col, lab in (("common", TEAL, "common shift alone"),
                        ("individual", OLIVE, "individual displacement alone"),
                        ("actual", NAVY, "both")):
        Y = get(k)[:, keep]; mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(x, mu, color=col, lw=2.4, label=lab, zorder=3)
    ax.plot(x, get("B")[:, keep].mean(0), color=CRIM, lw=2.4, ls=":", label="new facts", zorder=3)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.14); ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Injection step")
    ax.legend(frameon=False, fontsize=11, loc="lower right", handlelength=1.6)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig1_two_curves.{e}"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 1.9), dpi=200)
    plt.rcParams.update({"font.size": 10})
    schematic(ax)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"fig1_schematic.{e}"), bbox_inches="tight")
    print("wrote plots/fig1_two_curves and plots/fig1_schematic")


if __name__ == "__main__":
    main()
