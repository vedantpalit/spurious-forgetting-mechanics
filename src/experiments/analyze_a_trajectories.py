"""Four descriptive measurements of what happens to A. No mechanism assumed.

The earlier mechanism account is set aside pending re-derivation. This rebuilds
from the phenomenon: purely descriptive readouts on existing MLP-free p16000 checkpoints,
forward passes only, nothing trained. None of them presupposes a channel, a carrier, or a
decomposition, and the write-up of their results carries no mechanism proposals -- the point is
to narrow which CLASS of explanation is even possible before any is offered.

1. WHAT DOES A LOSE TO? For every A individual and attribute, where does the correct answer sit
   and what is above it. `rank_own` (among the population's own value half) and `rank_full`
   (among every value token for that attribute) separate the first fork: `rank_own` near 1 with
   `rank_full` large means A's answers fell below B's region while staying ordered among
   themselves; `rank_own` large means they became unranked among their own peers. The identity
   of the argmax is recorded too, categorised as own-half / opposite-half / another attribute's
   value / not a value token -- "displaced by B specifically" and "displaced by anything" are
   different findings.

2. IS THE DAMAGE UNIFORM ACROSS INDIVIDUALS? Per-individual accuracy trajectories, not the
   population mean. A mean of 0.30 is equally consistent with every individual at 0.30 and with
   30% of A destroyed while 70% is untouched, and those are different phenomena. This has not
   been looked at before.

3. DO THE INDIVIDUALS THAT CRASH RECOVER? From the same trajectories. If the recovering set is
   the crashing set, the event is reversible on those individuals. If different individuals
   crash and recover at different times, the population curve is churn and the "recovery" is an
   averaging artifact. This is the measurement most likely to change the picture.

4. WHERE DOES A'S REPRESENTATION FIRST DIVERGE? The residual at each layer against step 0, over
   injection. Descriptive only: at which depth the difference becomes substantial, and whether
   it starts early and propagates or appears late. Not which component caused it.

PRECISION. Left at the JAX default, which is TF32 on GPU, because every other analysis in this
repo ran that way and these numbers need to be comparable with the accuracy curves. That
costs ~0.03 on a logit, which can flip an argmax only at a near-tie.

Everything is logged per individual and per layer, un-summarised, so follow-ups do not need a
re-run.

Run (one invocation per seed):
  uv run python -m src.experiments.analyze_a_trajectories --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, get_partition, init_state, make_optimizer,
    rank_and_loss_within,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "a_trajectories"
BATCH_SIZE = 64

# top-1 categories
OWN, OPPOSITE, OTHER_ATTR, NON_VALUE = 0, 1, 2, 3
CAT_NAMES = ["own-half", "opposite-half", "other-attr", "non-value"]


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def forward(model, params, inputs, cols, n_layers):
    """Logits and per-layer residuals from ONE pass.

    Both readouts come from the same `apply`: running `eval_forward` for the logits and a
    second capturing pass for the residuals would double the compute for identical numbers.
    Residual entry 0 is the embedding output, entry l+1 the output of block l.
    """
    logits, aux = model.apply({"params": params}, inputs, deterministic=True,
                              capture_intermediates=True, mutable=["intermediates"])
    inter = aux["intermediates"]
    bi = np.arange(cols.shape[0])[:, None]
    res = [np.asarray(_get(inter, "input_layer", "__call__")[0])[bi, cols]]
    for i in range(n_layers):
        res.append(np.asarray(_get(inter, "backbone", f"block_{i}", "__call__")[0])[bi, cols])
    return np.asarray(logits), np.stack(res, axis=2).astype(np.float32)


def value_slot_columns(mask, targets):
    cols = np.zeros((mask.shape[0], NUM_ATTRIBUTES), dtype=np.int64)
    correct = np.zeros((mask.shape[0], NUM_ATTRIBUTES), dtype=np.int32)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], correct[:, k] = first_token_positions(mask, targets, k)
    return cols, correct


def categorise(top1, x_ids, y_ids):
    """(S?, N, A) -> category of the argmax token, per attribute."""
    cat = np.full(top1.shape, NON_VALUE, dtype=np.int8)
    all_vals = np.concatenate([np.concatenate([x_ids[k], y_ids[k]])
                               for k in range(NUM_ATTRIBUTES)])
    for k in range(NUM_ATTRIBUTES):
        col = top1[..., k]
        cat[..., k] = np.where(np.isin(col, all_vals), OTHER_ATTR, NON_VALUE)
        cat[..., k] = np.where(np.isin(col, y_ids[k]), OPPOSITE, cat[..., k])
        cat[..., k] = np.where(np.isin(col, x_ids[k]), OWN, cat[..., k])
    return cat


def score_checkpoint(model, params, ds, cols, correct, x_ids, y_ids, n_layers, model_dim):
    """Per-item accuracy, argmax identity, own-half and full-pool rank, and layer residuals."""
    n = ds.eval_inputs.shape[0]
    acc = np.zeros((n, NUM_ATTRIBUTES), dtype=np.uint8)
    top1 = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int32)
    r_own = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float32)
    r_full = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float32)
    z_correct = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float32)
    res = np.zeros((n, NUM_ATTRIBUTES, n_layers + 1, model_dim), dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        c = cols[sl]
        logits, res[sl] = forward(
            model, params, jnp.array(ds.eval_inputs[sl]), c, n_layers)
        bi = np.arange(c.shape[0])[:, None]
        z = logits[bi, c]                                        # (b, A, V)
        top1[sl] = z.argmax(-1)
        acc[sl] = (top1[sl] == correct[sl]).astype(np.uint8)
        for k in range(NUM_ATTRIBUTES):
            # the repo's own rank definition, reused rather than restated, so these numbers
            # are directly comparable with every published rank_own
            r_own[sl, k], _ = rank_and_loss_within(z[:, k], correct[sl, k], x_ids[k])
            r_full[sl, k], _ = rank_and_loss_within(
                z[:, k], correct[sl, k], np.concatenate([x_ids[k], y_ids[k]]))
            z_correct[sl, k] = z[np.arange(z.shape[0]), k, correct[sl, k]]
    return acc, top1, r_own, r_full, z_correct, res


def analyze(arm, pretrain_step, seed, condition, checkpoint_dir, total_steps):
    if arm != "mlp_free":
        raise SystemExit("mlp_free only")
    spec = dict(ARMS[arm])
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, std_model, data_cfg, _, data_a, _b, _c, _d = build(cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)
    n_layers = cfg.model.num_layers

    pre_key = mlpfree_pretrain_key(cfg, data_cfg)
    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
        pretrain_key=pre_key, pretrain_step=pretrain_step, seed=cfg.seed, **spec["key_extra"])
    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_opt, cfg.inject.total_steps), cfg.seed + 20)

    halves = get_partition(cfg, pop)
    x_ids = [np.asarray(pop.attr_first_token_ids[k][halves[k][0]])
             for k in range(NUM_ATTRIBUTES)]
    y_ids = [np.asarray(pop.attr_first_token_ids[k][halves[k][1]])
             for k in range(NUM_ATTRIBUTES)]
    cols, correct = value_slot_columns(np.asarray(data_a.eval_mask),
                                       np.asarray(data_a.eval_targets))
    steps = [int(s) for s in SCHEDULES[total_steps].split(",")]

    def path_for(s):
        return os.path.join(
            checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")

    print(f"arm={arm} p={pretrain_step} seed={seed} condition={condition} "
          f"N={data_a.eval_inputs.shape[0]} attrs={NUM_ATTRIBUTES} layers={n_layers}")
    print(f"  own-half pool sizes {[len(i) for i in x_ids]}, "
          f"opposite-half {[len(i) for i in y_ids]}, vocab {data_cfg.vocab_size}")

    if not os.path.exists(path_for(0)):
        raise SystemExit(f"step-0 checkpoint missing, and it is the baseline every layer "
                         f"divergence is measured against: {path_for(0)}")

    out = {k: [] for k in ("steps", "acc", "top1", "rank_own", "rank_full", "z_correct",
                           "layer_div")}
    res0 = norm0 = None
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING ({p})")
            continue
        a_, t_, ro, rf, zc, res = score_checkpoint(
            model, load_params(p, template.params), data_a, cols, correct,
            x_ids, y_ids, n_layers, cfg.model.model_dim)
        if res0 is None:
            assert s == 0, f"first present checkpoint is step {s}, not 0"
            res0 = res
            norm0 = np.linalg.norm(res0, axis=-1)                 # (N, A, L+1)
        div = np.linalg.norm(res - res0, axis=-1)                 # (N, A, L+1)
        out["steps"].append(s)
        out["acc"].append(a_)
        out["top1"].append(t_)
        out["rank_own"].append(ro)
        out["rank_full"].append(rf)
        out["z_correct"].append(zc)
        out["layer_div"].append(div.astype(np.float32))
        print(f"  {s:>5}: acc={a_.mean():.4f}  rank_own={ro.mean():.2f}  "
              f"rank_full={rf.mean():.2f}  layer-div/|h| last={np.nanmean(div[..., -1] / np.maximum(norm0[..., -1], 1e-9)):.4f}")

    acc = np.stack(out["acc"])                                    # (S, N, A)
    top1 = np.stack(out["top1"])
    cat = categorise(top1, x_ids, y_ids)
    div = np.stack(out["layer_div"])                              # (S, N, A, L+1)
    r_own = np.stack(out["rank_own"])
    r_full = np.stack(out["rank_full"])
    per_ind = acc.mean(axis=2).astype(np.float32)                 # (S, N)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{arm}-p{pretrain_step}-{condition}-t{total_steps}-seed{seed}"
    meta = dict(steps=np.array(out["steps"]), num_layers=n_layers, arm=arm,
                condition=condition, seed=seed, pretrain_step=pretrain_step,
                inject_total_steps=total_steps)
    # Full detail, gitignored: per individual, per attribute, per layer.
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"), acc=acc, top1=top1, cat=cat,
        rank_own=r_own, rank_full=r_full,
        z_correct=np.stack(out["z_correct"]), layer_div=div, layer_norm0=norm0,
        correct_id=correct, **meta)
    # Summary, tracked. per_ind is (S, N) at 88KB and is what measurements 2 and 3 need, so it
    # travels in full rather than as a histogram -- a histogram cannot answer question 3.
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}-summary.npz"),
        per_individual_acc=per_ind,
        cat_frac=np.stack([(cat == c).mean(axis=(1, 2)) for c in range(4)], axis=1),
        rank_own_mean=r_own.mean(axis=(1, 2)),
        rank_full_mean=r_full.mean(axis=(1, 2)),
        rank_own_top1_frac=(r_own == 1).mean(axis=(1, 2)),
        layer_div_mean=div.mean(axis=(1, 2)),
        layer_div_rel=(div / np.maximum(norm0, 1e-9)).mean(axis=(1, 2)),
        layer_norm0_mean=norm0.mean(axis=(0, 1)), **meta)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz (full, gitignored) and -summary.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    a = ap.parse_args()
    analyze(a.arm, a.pretrain_step, a.seed, a.condition, a.checkpoint_dir, a.total_steps)


if __name__ == "__main__":
    main()
