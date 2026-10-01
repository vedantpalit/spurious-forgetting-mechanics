"""Freeze-group ablation: which representation group carries recovery?

The representation-vs-readout split put 100% of the constant component's fall on the
representation side, but "representation" in the MLP-free model is THREE trainable groups --
embeddings, attention, LayerNorm -- and the split was correlational. This isolates them
causally by freezing groups during injection only, from pretrain checkpoints already in hand.
No new pretraining.

ARMS (arm 5, nothing frozen, is the EXISTING baseline and is deliberately not re-run here:
adding a freeze tag to its key would change the key and orphan the checkpoints already
computed):

    all_rep      embeddings + attention + layernorm frozen; head trainable only
    attention    attention frozen
    embeddings   embeddings frozen
    layernorm    layernorm frozen

`all_rep` IS THE POINT. The toy's no-go theorem says a model whose only free path to A's
logits is a fixed-key readout cannot recover, because every V_A column's driving term
`(1/|B|) sum_i p_i[j]` is strictly positive, so depression is monotone and saturating. That
arm is exactly that model, instantiated in the transformer. Pre-registered prediction
(recorded before submission): it crashes and does not recover, with ||c|| monotone and its
maximum at the final injection step, matching `c_readout`'s measured 0.0% drop in the split.

FREEZING IS DONE WITH `optax.multi_transform`, NOT `optax.masked`. `masked` passes RAW
updates through for the unmasked subset rather than zeroing them, so a mask built the obvious
way would leave the "frozen" group training at the bare gradient with no optimizer -- worse
than not freezing at all, and silent. `multi_transform` routes frozen leaves to
`optax.set_to_zero()` explicitly.

AND THE FREEZING IS VERIFIED, not trusted: at every dense checkpoint each frozen leaf is
asserted BIT-IDENTICAL to the pretrained checkpoint, and a mismatch raises rather than warns.

Run:
  uv run python -m src.experiments.mlpfree_freeze_injection \\
      --pretrain_step 16000 --inject_seed 0 --freeze_arm all_rep
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import optax
import wandb
from flax.traverse_util import flatten_dict

from src.config import parse_config
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_state, save_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, B_HALF, build, get_partition, init_state,
    make_optimizer, own_half_baseline_nats, parse_steps, population_baseline_nats,
    reset_stream, save_exposures, train_loop,
)
from src.experiments.mlpfree_common import (
    create_mlpfree_model, mlpfree_pretrain_key, verify_no_mlp,
)
from src.experiments.scale8_common import verify_architecture

MODEL_DIM, NUM_HEADS, NUM_LAYERS = 512, 8, 8
PRETRAIN_TOTAL_STEPS = 16000
PRETRAIN_PEAK_LR = 5e-4
SCHEDULES = {
    1200: "0,10,25,50,100,200,400,600,800,1000,1200",
    3000: "0,10,25,50,100,200,400,600,800,1000,1200,1600,2000,2400,3000",
}

# Which top-level path prefixes belong to which group. Verified against the actual MLP-free
# parameter tree, not assumed: with the MLPs removed each block has a single LayerNorm_0
# (LayerNorm_1 was the pre-MLP norm), plus one final backbone LayerNorm_0.
# Attention is split into QK (routing: what the pattern can be) and VO (the value and
# output projections, which feed the shared forward path). The QK arm is primary: the
# attention patterns were measured as flat through the recovery window, so freezing QK
# tests routing by intervention rather than by correlation.
GROUPS = {
    "embeddings": lambda path: path[0] == "input_layer",
    "attention_qk": lambda path: (path[0] == "backbone" and "CausalSelfAttention_0" in path
                                  and path[-2] in ("query", "key")),
    "attention_vo": lambda path: (path[0] == "backbone" and "CausalSelfAttention_0" in path
                                  and path[-2] in ("value", "out")),
    "layernorm": lambda path: path[0] == "backbone" and any(p.startswith("LayerNorm") for p in path),
    "head": lambda path: path[0] == "head",
}
ATTN = ("attention_qk", "attention_vo")
ARMS = {
    "all_rep": ("embeddings",) + ATTN + ("layernorm",),
    "attention": ATTN,
    "attention_qk": ("attention_qk",),
    "embeddings": ("embeddings",),
    "layernorm": ("layernorm",),
}


def group_of(path):
    hits = [g for g, pred in GROUPS.items() if pred(path)]
    if len(hits) != 1:
        raise RuntimeError(f"parameter {'/'.join(path)} matched groups {hits}; the model "
                           f"structure changed and GROUPS needs updating")
    return hits[0]


def build_labels(params, frozen_groups):
    """Per-leaf 'train'/'freeze' labels, and a printed census so the split is visible."""
    flat = flatten_dict(params)
    labels, census = {}, {}
    for path, leaf in flat.items():
        g = group_of(path)
        lab = "freeze" if g in frozen_groups else "train"
        labels[path] = lab
        c = census.setdefault(g, {"train": 0, "freeze": 0, "params": 0})
        c[lab] += 1
        c["params"] += int(np.prod(leaf.shape))
    from flax.traverse_util import unflatten_dict
    return unflatten_dict(labels), census


def assert_frozen_bit_identical(params, ref_flat, frozen_groups, step):
    """Every frozen leaf must be BIT-identical to the pretrained checkpoint. Raises."""
    bad = []
    for path, leaf in flatten_dict(params).items():
        if group_of(path) in frozen_groups:
            a, b = np.asarray(leaf), ref_flat[path]
            if a.shape != b.shape or not np.array_equal(a, b):
                bad.append(('/'.join(path), float(np.abs(a - b).max()) if a.shape == b.shape else -1.0))
    if bad:
        raise RuntimeError(
            f"FROZEN GROUPS MOVED at injection step {step}: {bad[:6]}"
            f"{' ...' if len(bad) > 6 else ''}. The masked-optimizer path is wrong and this "
            f"arm is invalid.")


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, required=True,
                       choices=list(range(6000, 16001, 1000)))
    extra.add_argument("--inject_condition", choices=["high_overlap", "disjoint"],
                       default="disjoint")
    extra.add_argument("--inject_seed", type=int, required=True)
    extra.add_argument("--inject_total_steps", type=int, default=1200, choices=sorted(SCHEDULES))
    extra.add_argument("--freeze_arm", required=True, choices=sorted(ARMS),
                       help="which representation group(s) to freeze during injection")
    known, remaining = extra.parse_known_args()
    dense_steps = SCHEDULES[known.inject_total_steps]
    frozen_groups = ARMS[known.freeze_arm]

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", "0",
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", known.inject_condition, "--inject.seed", str(known.inject_seed),
        "--inject.total_steps", str(known.inject_total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", dense_steps,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + remaining
    cfg = parse_config(KIConfig, description="MLP-free freeze-group ablation")
    pop, _std, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)

    pre_key = mlpfree_pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                            f"mlpfree-pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"No MLP-free pretrain checkpoint at {pre_path}")
    # Pretraining is deliberately UNFROZEN in every arm. The phenomenon is a property of the
    # FINE-TUNING update, so the question is which components have to be free for it to
    # happen; matching the pretrain would change the question being asked.
    print(f"Continuing from MLP-free pretrain step {known.pretrain_step}: {pre_path}")

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    base_tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    probe = init_state(model, cfg, data_cfg, base_tx, cfg.seed + 20)
    verify_architecture(probe.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_mlp(probe.params)
    labels, census = build_labels(probe.params, frozen_groups)

    print(f"\nFREEZE ARM: {known.freeze_arm}   frozen groups: {frozen_groups}")
    print(f"  {'group':>12} {'leaves train':>13} {'leaves freeze':>14} {'parameters':>12}")
    for g in sorted(census):
        c = census[g]
        print(f"  {g:>12} {c['train']:>13} {c['freeze']:>14} {c['params']:>12,}")
    n_frozen = sum(c["params"] for g, c in census.items() if g in frozen_groups)
    n_total = sum(c["params"] for c in census.values())
    print(f"  frozen parameters: {n_frozen:,} / {n_total:,} ({n_frozen / n_total:.1%})\n")

    # multi_transform, NOT masked: optax.masked passes RAW updates through for the
    # unmasked subset instead of zeroing them, which would leave "frozen" groups training
    # at the bare gradient, silently.
    # Load through the BASE-tx template, not the masked one. The saved pretrain checkpoint
    # holds a plain chain(clip, adamw) opt_state, serialized as {'0','1'}; a multi_transform
    # state is MultiTransformState(inner_states=...), serialized as {'inner_states'}, and
    # flax's from_bytes matches structurally, so loading into the masked template raises
    # "field names of the state dict and the named tuple do not match". Params are what we
    # want from the checkpoint anyway -- the masked optimizer is initialized fresh below.
    params, saved_opt_state, pre_step = load_state(pre_path, probe)

    tx = optax.multi_transform({"train": base_tx, "freeze": optax.set_to_zero()}, labels)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        # inherit_moments walks tuples looking for ScaleByAdamState; MultiTransformState
        # hides its inner states behind a dict field, so the walk would silently find
        # nothing and the arm would run with moments that were never transferred. Refuse
        # rather than pretend.
        raise SystemExit(
            "fresh_optimizer=False is not supported by the freeze arms: moment inheritance "
            "into a multi_transform state is not implemented, and doing it silently wrong "
            "would invalidate the arm.")
    ref_flat = {p: np.asarray(v).copy() for p, v in flatten_dict(state.params).items()}

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c) if d]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="mlpfreefreezeinject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed, arm="mlp_free",
        freeze_arm=known.freeze_arm)
    print(f"Injecting on dataB, condition={inject_cfg.condition}, seed={inject_cfg.seed}, "
          f"total_steps={inject_cfg.total_steps}, checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        assert_frozen_bit_identical(st.params, ref_flat, frozen_groups, step)
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir,
                         f"mlpfreefreezeinject-{known.freeze_arm}-p{known.pretrain_step}"
                         f"-{key}-step{step:04d}.msgpack")
        save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}  (frozen groups verified bit-identical)")

    baseline = population_baseline_nats(pop)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    own_halves = {"dataA": "X", "ballast": "Y"}
    b_half = B_HALF[inject_cfg.condition]
    if b_half in ("X", "Y"):
        own_halves["dataB"] = b_half
    if cfg.c_half in ("X", "Y"):
        own_halves["dataC"] = cfg.c_half
    rank_ctx = {"x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
                "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                                 "Y": own_half_baseline_nats(halves, "Y")}}

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
                    name=f"mlpfree-freeze-{known.freeze_arm}-p{known.pretrain_step}",
                    config={"pretrain_step": known.pretrain_step,
                            "condition": inject_cfg.condition, "seed": inject_cfg.seed,
                            "arm": "mlp_free", "freeze_arm": known.freeze_arm}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)

    assert_frozen_bit_identical(state.params, ref_flat, frozen_groups, "final")
    save_exposures(cfg, data_b,
                   f"mlpfreefreezeinject-{known.freeze_arm}-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")
    print(f"Frozen groups verified bit-identical at every dense checkpoint and at the end.")


if __name__ == "__main__":
    main()
