"""Do the unembedding rows for A's values move during injection? (measurement 1)

Descriptive. Forward passes and parameter reads on existing MLP-free p16000 checkpoints; nothing
is trained and no mechanism is asserted beyond what the numbers show.

THE QUESTION. Suppression is individual-blind, template-blind, position-blind and
pool-size-blind, but depends on a value token's cross-attribute sharing count. This asks whether
that dependence is visible in the OUTPUT geometry: the readout is `logit_v = h . W[:, v]`, so if
A's value rows `W[:, v]` move relative to B's, the effect is output-side and naturally
individual-blind, since `W` does not know which individual is being asked about.

WHAT IS MEASURED, per dense checkpoint, for A's value rows (the union of the X halves), B's
value rows (the union of the Y halves), and non-value rows as a control:

  * displacement magnitude ||dW[:, v]||, absolute and relative to ||W_0[:, v]||
  * coherence -- the share of the displacement set's total energy along its leading singular
    direction. Rows moving together give a high share; rows moving independently give ~1/rank.
  * alignment cos(dW[:, v], h) against the mean post-LayerNorm residual at the value slot for
    A's prompts and for B's prompts, since those are the vectors the rows are dotted with.

READ. If A's rows barely move relative to the control, the output-side framing is wrong and the
effect is representation-side; measurements 2 and 3 should not be run.

PERSISTED. Per token and per step: displacement norm and both alignments. Plus the step-0
unembedding itself (3.9MB), so the pretrained-geometry questions can be answered without
re-reading checkpoints.

Run:
  uv run python -m src.experiments.analyze_unembedding_geometry --seed 0
"""
import argparse
import os

import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.analyze_arms import ARM_SPECS, load_arm
from src.experiments.knowledge_injection import first_token_positions, get_partition

OUT_DIR = "unembedding_geometry"
BATCH_SIZE = 64


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def mean_readout_state(model, params, ds, n_batches=8):
    """Mean post-final-LayerNorm residual at the value slots -- the vector the unembedding rows
    are actually dotted with, so the one an alignment should be measured against."""
    acc, n = None, 0
    for start in range(0, min(ds.eval_inputs.shape[0], n_batches * BATCH_SIZE), BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        x = jnp.array(ds.eval_inputs[sl])
        _, aux = model.apply({"params": params}, x, deterministic=True,
                             capture_intermediates=True, mutable=["intermediates"])
        h = np.asarray(_get(aux["intermediates"], "backbone", "__call__")[0])   # = post-final-LN, or last residual if no final LN
        mask = np.asarray(ds.eval_mask[sl])
        targets = np.asarray(ds.eval_targets[sl])
        rows = np.arange(mask.shape[0])
        for k in range(NUM_ATTRIBUTES):
            cols, _ = first_token_positions(mask, targets, k)
            v = h[rows, cols]
            acc = v.sum(0) if acc is None else acc + v.sum(0)
            n += v.shape[0]
    return acc / n


def coherence(d):
    """Share of a displacement set's energy along its leading singular direction.

    Rows moving together give a value near 1; independent moves give ~1/min(rows, dim). The
    baseline for comparison is that of the control set, not zero.
    """
    if d.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(d, compute_uv=False)
    return float(s[0] ** 2 / max((s ** 2).sum(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free", choices=sorted(ARM_SPECS))
    ap.add_argument("--pretrain_step", type=int, default=None,
                    help="defaults to the arm's own (16000 for the 8-layer arms, 8000 for 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    a = ap.parse_args()

    # Parameter reads and a few forward passes; nothing here depends on the stack being
    # MLP-free, so every arm in the registry is allowed. The ballast contrast needs the
    # readout bias (below) in the no-ballast arms, where the Y rows leave pretraining cold.
    ctx = load_arm(a.arm, a.seed, a.condition, a.total_steps, a.checkpoint_dir, a.pretrain_step)
    model, data_a, data_b, pop, cfg = ctx.model, ctx.data_a, ctx.data_b, ctx.pop, ctx.cfg
    vocab = ctx.data_cfg.vocab_size
    path_for = ctx.path_for
    steps = ctx.steps

    halves = get_partition(cfg, pop)
    x_ids = np.unique(np.concatenate(
        [np.asarray(pop.attr_first_token_ids[k][halves[k][0]]) for k in range(NUM_ATTRIBUTES)]))
    y_ids = np.unique(np.concatenate(
        [np.asarray(pop.attr_first_token_ids[k][halves[k][1]]) for k in range(NUM_ATTRIBUTES)]))
    other = np.setdiff1d(np.arange(vocab), np.concatenate([x_ids, y_ids]))
    print(f"A value rows {len(x_ids)}, B value rows {len(y_ids)}, non-value rows {len(other)}, "
          f"vocab {vocab}")

    if not os.path.exists(path_for(0)):
        raise SystemExit(f"step-0 checkpoint missing: {path_for(0)}")
    params0 = ctx.load(0)
    W0 = np.asarray(_get(params0, "head", "Dense_0", "kernel")).astype(np.float64)   # (E, V)
    sA0 = mean_readout_state(model, params0, data_a).astype(np.float64)   # unnormalised
    hA = sA0 / np.linalg.norm(sA0)
    hB = mean_readout_state(model, params0, data_b).astype(np.float64)
    hB /= np.linalg.norm(hB)
    nY0, nX0 = np.linalg.norm(W0[:, y_ids], axis=0).mean(), np.linalg.norm(W0[:, x_ids], axis=0).mean()
    print(f"  head kernel {W0.shape};  cos(hA, hB) = {float(hA @ hB):.4f}")
    print(f"  mean ||W0[:, v]||: A {nX0:.3f}  B {nY0:.3f}  "
          f"other {np.linalg.norm(W0[:, other], axis=0).mean():.3f}   Y/X ratio {nY0 / nX0:.3f}")
    print()

    # THE READOUT BIAS A RECEIVES, in nats. RB(t) = <s_A(0), u_bar_Y(t) - u_bar_X(t)> with A's
    # step-0 readout direction held fixed, so only the rows move. In the K=1 toy this is
    # monotone non-decreasing whenever B still puts mass on the other half, because dL/dU has
    # no normalizer Jacobian and so no restoring term; a bias parked here is never withdrawn.
    # A negative excursion at the population level is the falsifier.
    def rb(W):
        return float(sA0 @ (W[:, y_ids].mean(1) - W[:, x_ids].mean(1)))

    out = {k: [] for k in ("steps", "norm", "cosA", "cosB", "rb", "yx_ratio")}
    print(f"  {'step':>5} | {'||dW|| A':>9} {'B':>8} {'other':>8} | {'rel A':>7} {'rel B':>7} "
          f"| {'coh A':>6} {'coh B':>6} {'coh oth':>7} | {'cos(dW,hB)':>11} {'cos(dW,hA)':>11} "
          f"| {'RB nats':>8} {'dRB':>7} {'Y/X':>6}")
    rb0 = rb(W0)
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING")
            continue
        W = np.asarray(_get(ctx.load(s), "head", "Dense_0", "kernel")).astype(np.float64)
        dW = W - W0
        nrm = np.linalg.norm(dW, axis=0)                       # (V,)
        cosA = (hA @ dW) / np.maximum(nrm, 1e-30)
        cosB = (hB @ dW) / np.maximum(nrm, 1e-30)
        rbt = rb(W)
        yx = np.linalg.norm(W[:, y_ids], axis=0).mean() / np.linalg.norm(W[:, x_ids], axis=0).mean()
        out["steps"].append(s)
        out["norm"].append(nrm.astype(np.float32))
        out["cosA"].append(cosA.astype(np.float32))
        out["cosB"].append(cosB.astype(np.float32))
        out["rb"].append(rbt)
        out["yx_ratio"].append(yx)
        rel = nrm / np.maximum(np.linalg.norm(W0, axis=0), 1e-30)
        print(f"  {s:>5} | {nrm[x_ids].mean():>9.4f} {nrm[y_ids].mean():>8.4f} "
              f"{nrm[other].mean():>8.4f} | {rel[x_ids].mean():>7.4f} {rel[y_ids].mean():>7.4f} "
              f"| {coherence(dW[:, x_ids].T):>6.3f} {coherence(dW[:, y_ids].T):>6.3f} "
              f"{coherence(dW[:, other].T):>7.3f} | {cosB[x_ids].mean():>11.4f} "
              f"{cosA[x_ids].mean():>11.4f} | {rbt:>8.3f} {rbt - rb0:>+7.3f} {yx:>6.3f}")
    rbs = np.array(out["rb"])
    if len(rbs) > 1:
        neg = int((np.diff(rbs) < 0).sum())
        print(f"\n  readout bias: {rb0:.3f} -> {rbs[-1]:.3f} nats; "
              f"{neg} of {len(rbs) - 1} intervals decreasing "
              f"({'monotone, as the toy predicts' if neg == 0 else 'NOT monotone'})")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}-p{ctx.pretrain_step}-{a.condition}-t{a.total_steps}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(out["steps"]), norm=np.stack(out["norm"]),
        cosA=np.stack(out["cosA"]), cosB=np.stack(out["cosB"]),
        rb=rbs, rb0=rb0, yx_ratio=np.array(out["yx_ratio"]), yx_ratio0=nY0 / nX0,
        sA0=sA0.astype(np.float32),
        W0=W0.astype(np.float32), hA=hA.astype(np.float32), hB=hB.astype(np.float32),
        x_ids=x_ids, y_ids=y_ids, other_ids=other,
        attr_x=np.concatenate([np.full(len(halves[k][0]), k) for k in range(NUM_ATTRIBUTES)]),
        attr_x_ids=np.concatenate(
            [np.asarray(pop.attr_first_token_ids[k][halves[k][0]]) for k in range(NUM_ATTRIBUTES)]),
        attr_y_ids=np.concatenate(
            [np.asarray(pop.attr_first_token_ids[k][halves[k][1]]) for k in range(NUM_ATTRIBUTES)]),
        attr_y=np.concatenate([np.full(len(halves[k][1]), k) for k in range(NUM_ATTRIBUTES)]),
        seed=a.seed, pretrain_step=ctx.pretrain_step, condition=a.condition, arm=a.arm,
        key=ctx.key)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz "
          f"(per-token norms and alignments, plus the step-0 unembedding)")


if __name__ == "__main__":
    main()
