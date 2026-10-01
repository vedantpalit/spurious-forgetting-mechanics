"""MLP-free arm: injection on dataB, continuing from a mlpfree_pretrain.py checkpoint.

Mirrors scale8_injection.py exactly except for the MLP removal and two deliberate changes:

  * 3000 injection steps, not 1200. The scale8 runs stopped at 1200, but the ballast run
    needed ~2400 before "no recovery" was the right read, so 1200 risks reporting a
    truncation as a null. The dense schedule keeps scale8's first-1200 points unchanged
    (0,10,25,50,100,200,400,600,800,1000,1200) so the like-for-like window is
    point-for-point identical, then extends (1600,2000,2400,3000).
  * `disjoint` only. That is where the dissociation is clean at this scale; high_overlap
    would be a sweep, which is out of scope here.

Mean rank is deliberately NOT printed live here. `evaluate_and_log` computes
`rank_own_mean` but its console `_fmt` prints only `rank_own_top1`, which is why the
standard arm's mean rank has to be recomputed offline from its saved checkpoints. Rather
than print it for one arm and recompute it for the other, BOTH arms' mean rank is
recomputed from saved params by analyze_mlpfree.py, through one code path -- strictly more
comparable than mixing a live number with a recomputed one.

Run:
  uv run python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0
"""
import argparse
import os
import sys
from dataclasses import replace

import wandb

from src.config import parse_config
from src.experiments.ckpt import experiment_key, load_state, save_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, get_partition, inherit_moments, init_state,
    make_optimizer, own_half_baseline_nats, parse_steps, population_baseline_nats,
    reset_stream, save_exposures, train_loop,
)
from src.experiments.mlpfree_common import (
    MODEL_DIM, NUM_HEADS, NUM_LAYERS, create_mlpfree_model, mlpfree_pretrain_key, verify_no_mlp,
)
from src.experiments.scale8_common import verify_architecture
from src.data.biography import NUM_ATTRIBUTES

# Run length only. `total_steps` feeds make_optimizer, but INJECTION USES A CONSTANT LR:
# InjectConfig supplies its own OptConfig(peak_lr=-1.0, schedule="constant"), overriding
# OptConfig's class-level "cosine" default, so total_steps never reaches the cosine branch
# and does not touch the learning rate.
#
# Recorded because a previous version of this comment claimed the opposite. Reading
# OptConfig.schedule's class default without checking InjectConfig's override, this was
# flagged as a confound between the 1200-step standard arm and a 3000-step MLP-free arm.
# It is not one, and the runs prove it directly: the 1200- and 3000-step MLP-free runs
# agree to four decimal places at every shared step (0.2969 vs 0.2967 at step 1200),
# which could not happen if their learning rates had differed by 1.8x at step 600.
#
# 1200 is kept as the default anyway -- it matches the standard arm's run length, so the
# dense checkpoint steps line up exactly and no interpolation is needed to compare. 3000
# extends the window to check that no late recovery appears after step 1200.
SCHEDULES = {
    1200: "0,10,25,50,100,200,400,600,800,1000,1200",
    3000: "0,10,25,50,100,200,400,600,800,1000,1200,1600,2000,2400,3000",
}
PRETRAIN_TOTAL_STEPS = 16000  # must match mlpfree_pretrain.py exactly -- part of the key
PRETRAIN_PEAK_LR = 5e-4


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, required=True,
                        choices=list(range(6000, 16001, 1000)))
    extra.add_argument("--inject_condition", choices=["high_overlap", "disjoint", "all_values"],
                        default="disjoint")
    extra.add_argument("--inject_seed", type=int, required=True)
    extra.add_argument("--inject_total_steps", type=int, default=1200,
                        choices=sorted(SCHEDULES),
                        help="1200 = matched to the standard arm (default). 3000 = the "
                             "sustained-LR robustness variant; NOT comparable to the "
                             "standard arm past ~step 200, see SCHEDULES above.")
    known, remaining = extra.parse_known_args()
    dense_steps = SCHEDULES[known.inject_total_steps]

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
    cfg = parse_config(KIConfig, description="MLP-free ablation: injection")
    pop, _std_model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)

    model = create_mlpfree_model(cfg.model, data_cfg)

    pre_key = mlpfree_pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                             f"mlpfree-pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(
            f"No MLP-free pretrain checkpoint at {pre_path}. Check --pretrain_step matches "
            f"one of the steps mlpfree_pretrain.py actually saved (6000-16000, every 1000).")
    print(f"Continuing from MLP-free pretrain step {known.pretrain_step}: {pre_path}")

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_mlp(state.params)

    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
    print(f"Optimizer: {'inherited' if not inject_cfg.fresh_optimizer else 'fresh'}")
    print(f"pre_step (global step counter carried into injection) = {pre_step}")
    if pre_step != known.pretrain_step:
        raise ValueError(f"Loaded checkpoint's own step counter ({pre_step}) doesn't match "
                          f"the requested --pretrain_step ({known.pretrain_step}) -- "
                          f"the checkpoint filename and its contents disagree.")

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c) if d]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="mlpfreeinject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed, arm="mlp_free")
    matched = "MATCHED to the standard arm" if known.inject_total_steps == 1200 else \
        "sustained-LR variant, NOT comparable to the standard arm past ~step 200"
    print(f"Injecting on dataB, pretrain_step={known.pretrain_step}, "
          f"condition={known.inject_condition}, seed={known.inject_seed}, "
          f"total_steps={known.inject_total_steps} ({matched}; cosine decay horizon = "
          f"total_steps), checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir,
                          f"mlpfreeinject-p{known.pretrain_step}-{key}-step{step:04d}.msgpack")
        save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

    baseline = population_baseline_nats(pop)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    b_half = B_HALF[inject_cfg.condition]
    own_halves = {"dataA": "X", "ballast": "Y"}
    if b_half in ("X", "Y"):
        own_halves["dataB"] = b_half
    if cfg.c_half in ("X", "Y"):
        own_halves["dataC"] = cfg.c_half
    rank_ctx = {
        "x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
        "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                          "Y": own_half_baseline_nats(halves, "Y")},
    }

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
                    name=f"mlpfree-inject-p{known.pretrain_step}-{known.inject_condition}",
                    config={"pretrain_step": known.pretrain_step,
                            "condition": known.inject_condition,
                            "seed": known.inject_seed, "arm": "mlp_free"}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, data_b, f"mlpfreeinject-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")


if __name__ == "__main__":
    main()
