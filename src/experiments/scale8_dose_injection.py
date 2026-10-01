"""Dose experiment on the STANDARD 8-layer transformer: does the turnover track B's
acquisition, or the step number? (The MLP-free version is mlpfree_dose_injection.py,
FINDINGS 3.19; this is the same design on the arm the paper's transformer section uses.)

THE MANIPULATION IS TIMING, NOT CARDINALITY. Vary B's acquisition speed with two knobs that
change when discrimination gets written without changing what has to be learned:

    --lr_mult        0.5 / 1 / 2 x the matched injection LR   (step magnitude)
    --inject_batch   128 / 256 / 512                          (gradient noise, per-step signal)

The reading is on the eval lines (every 10 steps): A's trough step and B's accuracy at that
step, per condition -- the small model's Law 1 (toy/THEORY.md section 3). No
decomposition pass is needed, so checkpoints are kept to the two ends.

Same populations, pretrained checkpoint and optimiser as scale8_injection.py; 3000 injection
steps so the slow conditions reach their turnover. lr_mult 1.0 / batch 256 at 3000 steps is
its own run here (the 1200-step scale8 baseline has a different schedule, hence a different
key) and is part of the sweep.

Run:
  uv run python -m src.experiments.scale8_dose_injection --inject_seed 0 --lr_mult 0.5
"""
import argparse
import os
import sys
from dataclasses import replace

import wandb

from src.config import parse_config
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_state, save_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, get_partition, inherit_moments, init_state,
    make_optimizer, own_half_baseline_nats, parse_steps, population_baseline_nats,
    pretrain_key, reset_stream, save_exposures, train_loop,
)
from src.experiments.scale8_common import MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, verify_architecture

PRETRAIN_TOTAL_STEPS = 16000  # must match scale8_pretrain.py exactly -- part of the key
PRETRAIN_PEAK_LR = 5e-4
INJECT_TOTAL_STEPS = 3000
CHECKPOINT_STEPS = "0,3000"   # the reading is on the eval lines; no decomposition pass


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, default=16000)
    extra.add_argument("--inject_condition", choices=["high_overlap", "disjoint"], default="disjoint")
    extra.add_argument("--inject_seed", type=int, required=True)
    extra.add_argument("--lr_mult", type=float, default=1.0)
    extra.add_argument("--inject_batch", type=int, default=256)
    known, remaining = extra.parse_known_args()

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", known.inject_condition, "--inject.seed", str(known.inject_seed),
        "--inject.total_steps", str(INJECT_TOTAL_STEPS),
        "--inject.batch_size", str(known.inject_batch),
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", CHECKPOINT_STEPS,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + remaining
    cfg = parse_config(KIConfig, description="8-layer standard arm: dose experiment")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                            f"pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretrain checkpoint at {pre_path}")

    base_lr = PRETRAIN_PEAK_LR / INJECT_LR_RATIO
    inject_opt = replace(cfg.inject.opt, peak_lr=base_lr * known.lr_mult)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    print(f"\nDOSE CONDITION: lr_mult={known.lr_mult}  inject_lr={inject_opt.peak_lr:.3e} "
          f"(matched ratio gives {base_lr:.3e})   batch={inject_cfg.batch_size}")
    print(f"Continuing from pretrain step {known.pretrain_step}: {pre_path}\n")

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
    if pre_step != known.pretrain_step:
        raise ValueError(f"checkpoint step counter {pre_step} != requested {known.pretrain_step}")

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c)
                 if d is not None and len(d.person_ids) > 0]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    # both knobs enter the key through the hashed inject config (peak_lr via inject.opt,
    # batch_size directly), so every condition gets a distinct key with nothing appended.
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="scale8inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed)
    print(f"Injecting on dataB, condition={inject_cfg.condition}, seed={inject_cfg.seed}, "
          f"total_steps={inject_cfg.total_steps}, checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir,
                         f"scale8inject-p{known.pretrain_step}-{key}-step{step:04d}.msgpack")
        save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

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
                    name=f"scale8-dose-lr{known.lr_mult}-bs{known.inject_batch}-p{known.pretrain_step}",
                    config={"pretrain_step": known.pretrain_step, "arm": "standard",
                            "condition": inject_cfg.condition, "seed": inject_cfg.seed,
                            "lr_mult": known.lr_mult, "inject_batch": known.inject_batch}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)

    save_exposures(cfg, data_b, f"scale8inject-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")


if __name__ == "__main__":
    main()
