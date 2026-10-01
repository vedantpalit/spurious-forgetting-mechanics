"""The interventions as time courses: what the old facts' accuracy curve looks like when
you intervene, against the curve as it is.

    none        the model, no intervention
    patch       at every checkpoint, subtract the shift estimated from PATCH_K old facts
                from every other old fact's readout state, read out (no training)
    replay-k    k old facts mixed into every injection batch (REPLAY_FRAC of the batch),
                accuracy on the old facts NOT replayed

Base configuration, normalized arm, one rate, gate-matched at 200, ten seeds. B's curve is
drawn for the no-intervention run. Writes interventions_timecourse.json and
plots/toy_min_interventions_timecourse.{png,pdf}.
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
NAVY, CRIM, TEAL, OLIVE, DEEP, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "#1E5E6B", "0.62"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.grid": False})
SEEDS = range(10)
STEPS = 5000
PATCH_K = 4
REPLAY_KS = (8, 32)
REPLAY_FRAC = 0.25        # of each 32-sample batch: 8 old-fact samples, 24 new


def readout_state(p, k):
    _, h = m.fwd(p, jnp.asarray(k), 1)
    return np.asarray(m.rms(h))


def acc_state(p, x, v):
    return (x @ np.asarray(p["U"])).argmax(1) == v


def run(seed, replay_k=0, rng_seed=0):
    p0, lr, gate, mu, A, Dd, B = m.pretrain(seed, 0.5, 1)
    ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, 1, 1.0, lr / m.INJECT_RATIO, seed, 200)
    (kA, vA), (kB, vB) = A, B
    x0 = readout_state(p0, kA)
    rng = np.random.default_rng(seed + 5); prng = np.random.default_rng(seed + 21)
    rep_idx = np.random.default_rng(seed + 31).choice(m.NA, replay_k, replace=False) if replay_k else np.array([], int)
    rep_mask = np.zeros(m.NA, bool); rep_mask[rep_idx] = True
    n_old = int(round(32 * REPLAY_FRAC)) if replay_k else 0
    gi = set(m.grid(STEPS)); rows = []; p = p0
    for t in range(STEPS + 1):
        if t in gi:
            xt = readout_state(p, kA); act = acc_state(p, xt, vA)
            row = dict(step=t, B=m.accuracy(p, kB, vB, 1))
            if replay_k:
                row["others"] = float(act[~rep_mask].mean()); row["replayed"] = float(act[rep_mask].mean())
            else:
                row["A"] = float(act.mean())
                accs = []
                for _ in range(5):
                    pick = prng.choice(m.NA, PATCH_K, replace=False); mask = np.zeros(m.NA, bool); mask[pick] = True
                    accs.append(float(acc_state(p, xt - (xt - x0)[pick].mean(0), vA)[~mask].mean()))
                row["patch"] = float(np.mean(accs))
            rows.append(row)
        if t < STEPS:
            ib = rng.integers(0, m.NB, 32 - n_old)
            k = kB[ib]; v = vB[ib]
            if n_old:
                ia = rep_idx[rng.integers(0, replay_k, n_old)]
                k = np.concatenate([k, kA[ia]]); v = np.concatenate([v, vA[ia]])
            p = m.step(p, jnp.asarray(k), jnp.asarray(v), ilr, 1.0, 1)
    return rows


def main():
    out = os.path.join(HERE, "interventions_timecourse.json")
    if os.path.exists(out):
        data = json.load(open(out))
    else:
        data = {}
        for s in SEEDS:
            data[str(s)] = {"none": run(s)}
            for k in REPLAY_KS:
                data[str(s)][f"replay{k}"] = run(s, replay_k=k)
            r = data[str(s)]
            print(f"seed {s}: A trough {min(x['A'] for x in r['none']):.2f} | patch min {min(x['patch'] for x in r['none']):.2f} | "
                  + " ".join(f"replay{k} others min {min(x['others'] for x in r[f'replay{k}']):.2f} end {r[f'replay{k}'][-1]['others']:.2f}" for k in REPLAY_KS), flush=True)
        json.dump(data, open(out, "w"))
    st = np.array([x["step"] for x in data["0"]["none"]], float)
    get = lambda run_, key: np.array([[x[key] for x in data[s][run_]] for s in data])

    fig, ax = plt.subplots(figsize=(6.0, 3.9), dpi=200)
    series = [("none", "A", NAVY, "-", "no intervention"),
              ("none", "patch", OLIVE, "-", f"subtract the shift estimated from {PATCH_K} old facts (no training)")]
    for k, c in zip(REPLAY_KS, (DEEP, TEAL)):
        series.append((f"replay{k}", "others", c, "-", f"replay {k} of the {m.NA} old facts during training: the others"))
    for run_, key, c, ls, lab in series:
        Y = get(run_, key); mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, mu - sd, mu + sd, color=c, alpha=0.13, lw=0)
        ax.plot(st, mu, color=c, ls=ls, lw=1.9, label=lab)
        tr = int(mu.argmin()); print(f"{lab}: trough {mu[tr]:.3f}@{int(st[tr])} end {mu[-1]:.3f}")
    ax.plot(st, get("none", "B").mean(0), color=CRIM, ls=":", lw=1.8, label="new facts")
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, STEPS); ax.set_ylim(0, 1.03)
    ax.set_xlabel("injection step"); ax.set_ylabel("first-token accuracy, old facts")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", handlelength=1.8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"toy_min_interventions_timecourse.{e}"), bbox_inches="tight")
    print("wrote plots/toy_min_interventions_timecourse")


if __name__ == "__main__":
    main()
