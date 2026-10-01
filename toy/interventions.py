"""Interventions the mechanism predicts, tested on the minimal model.

The common shift is ONE vector shared by every old fact; the individual displacement is
per fact. So:

  (1) REPLAY.  Fine-tuning briefly on k old facts pushes the shared row back for everyone:
      accuracy on the NON-replayed old facts should return with k << n_A at the trough
      (suppression), and barely move at a late checkpoint (erosion), where only the replayed
      facts themselves are repaired.
  (2) PATCH (no training).  Estimate delta from k old facts (mean change of their readout
      state), subtract it from every old fact's state, read out. Full repair from one vector
      at the trough; little at the late checkpoint.

Base configuration, normalized arm, one rate, gate-matched at 200, ten seeds. Two
checkpoints: A's trough, and step 5000. Replay: R steps at the injection rate on batches
from the k facts alone. Writes interventions.json and plots/toy_min_interventions.{png,pdf}.
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
NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.62"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.grid": False})
SEEDS = range(10)
KS = [1, 2, 4, 8, 16, 32, 64]
R_LIST = [50, 200]
LATE = 5000


def readout_state(p, k):
    _, h = m.fwd(p, jnp.asarray(k), 1)
    return np.asarray(m.rms(h))


def acc_from_state(p, x, v):
    z = x @ np.asarray(p["U"])
    return (z.argmax(1) == v)


def run_to(p0, kB, vB, ilr, seed, steps):
    p = p0; rng = np.random.default_rng(seed + 5)
    for t in range(steps):
        i = rng.integers(0, m.NB, 32)
        p = m.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 1.0, 1)
    return p


def one(seed):
    p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, 0.5, 1)
    ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, seed, 200)
    (kA, vA), (kB, vB) = A, B
    rows, _ = m.inject(p0, lr, mu, A, Dd, B, 1, 1.0, LATE, seed, ilr=ilr)
    st = np.array([r["step"] for r in rows]); Aacc = np.array([r["A"] for r in rows])
    trough = int(st[Aacc.argmin()])
    x0 = readout_state(p0, kA)
    out = {"trough_step": trough}
    rng = np.random.default_rng(seed + 11)
    for label, T in (("trough", trough), ("late", LATE)):
        p = run_to(p0, kB, vB, ilr, seed, T)
        xt = readout_state(p, kA)
        actual = acc_from_state(p, xt, vA)
        dx = xt - x0
        res = {"actual": float(actual.mean()), "patch_full": float(acc_from_state(p, xt - dx.mean(0), vA).mean()),
               "patch_k": {}, "replay_k": {}}
        for k in KS:
            accs_p, accs_r = [], {R: ([], []) for R in R_LIST}
            for rep in range(3):
                idx = rng.choice(m.NA, k, replace=False)
                mask = np.zeros(m.NA, bool); mask[idx] = True
                # (2) patch: delta estimated from the k facts, subtracted from everyone
                accs_p.append(float(acc_from_state(p, xt - dx[idx].mean(0), vA)[~mask].mean()))
                # (1) replay: R steps on the k facts alone (R_LIST is cumulative: 200 continues 50)
                q = p; r2 = np.random.default_rng(seed * 100 + k + rep); done = 0
                for R in R_LIST:
                    for t in range(R - done):
                        i = idx[r2.integers(0, k, min(32, k))]
                        q = m.step(q, jnp.asarray(kA[i]), jnp.asarray(vA[i]), ilr, 1.0, 1)
                    done = R
                    a = acc_from_state(q, readout_state(q, kA), vA)
                    accs_r[R][0].append(float(a[~mask].mean())); accs_r[R][1].append(float(a[mask].mean()))
            res["patch_k"][str(k)] = float(np.mean(accs_p))
            res["replay_k"][str(k)] = {str(R): {"non_replayed": float(np.mean(accs_r[R][0])), "replayed": float(np.mean(accs_r[R][1]))} for R in R_LIST}
        out[label] = res
        print(f"seed {seed} {label}@{T}: actual {res['actual']:.2f} | patch(full delta) {res['patch_full']:.2f} | "
              + " ".join(f"k={k}: patch {res['patch_k'][str(k)]:.2f} replay50 {res['replay_k'][str(k)]['50']['non_replayed']:.2f} replay200 {res['replay_k'][str(k)]['200']['non_replayed']:.2f}" for k in (1, 8, 64)),
              flush=True)
    return out


def figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), dpi=200, sharey=True)
    for ax, label, title in zip(axes, ("trough", "late"), ("at the trough (suppression)", "at step 5000 (erosion)")):
        P = np.array([[data[s][label]["patch_k"][str(k)] for k in KS] for s in data])
        R50 = np.array([[data[s][label]["replay_k"][str(k)]["50"]["non_replayed"] for k in KS] for s in data])
        R200 = np.array([[data[s][label]["replay_k"][str(k)]["200"]["non_replayed"] for k in KS] for s in data])
        act = np.array([data[s][label]["actual"] for s in data])
        for Y, c, ls, lab in ((R200, TEAL, "-", "replay $k$ old facts, 200 steps: the others"),
                              (R50, TEAL, ":", "replay $k$ old facts, 50 steps: the others"),
                              (P, OLIVE, "-", "subtract the shift estimated from $k$ facts")):
            mu, sd = Y.mean(0), Y.std(0, ddof=1)
            ax.fill_between(KS, mu - sd, mu + sd, color=c, alpha=0.13, lw=0)
            ax.plot(KS, mu, color=c, ls=ls, lw=1.9, marker="o", ms=3.5, label=lab)
        ax.axhline(act.mean(), color=NAVY, lw=1.4, ls="--", label="no intervention")
        ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("$k$ old facts used"); ax.set_title(title, fontsize=10.5); ax.set_ylim(0, 1.04)
    axes[0].set_ylabel("first-token accuracy, old facts")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right", handlelength=1.8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_interventions.{e}"), bbox_inches="tight")
    print("wrote plots/toy_min_interventions")


def main():
    out = os.path.join(HERE, "interventions.json")
    if os.path.exists(out):
        data = json.load(open(out))
    else:
        data = {str(s): one(s) for s in SEEDS}
        json.dump(data, open(out, "w"))
    for label in ("trough", "late"):
        act = np.mean([data[s][label]["actual"] for s in data]); pf = np.mean([data[s][label]["patch_full"] for s in data])
        print(f"\n{label}: actual {act:.3f}; subtract full delta {pf:.3f}")
        print(f"  {'k':>4} {'patch_k':>8} {'replay50 others':>16} {'replay200 others':>17} {'the k (200)':>12}")
        for k in KS:
            g = lambda f: np.mean([f(data[s][label]) for s in data])
            print(f"  {k:>4} {g(lambda d: d['patch_k'][str(k)]):>8.3f} "
                  f"{g(lambda d: d['replay_k'][str(k)]['50']['non_replayed']):>16.3f} "
                  f"{g(lambda d: d['replay_k'][str(k)]['200']['non_replayed']):>17.3f} "
                  f"{g(lambda d: d['replay_k'][str(k)]['200']['replayed']):>12.3f}")
    figure(data)


if __name__ == "__main__":
    main()
