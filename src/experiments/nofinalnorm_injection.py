"""No-final-norm arm: injection on dataB, continuing from a nofinalnorm_pretrain.py checkpoint.

Mirrors scale8_injection.py (the standard arm, the control) exactly except for the model
factory: same populations, injection rate (pretrain peak / INJECT_LR_RATIO, constant),
batch, eval interval, and the dense checkpoint schedule
0,10,25,50,100,200,400,600,800,1000,1200, so every fixed-checkpoint analysis
(analyze_weight_patch --score accuracy, analyze_delta_structure, analyze_unembedding_geometry)
runs on this arm through analyze_arms.load_arm("no_final_norm", ...) with no change.

THE READ. Three things, in this order, against the standard arm at the same steps:
  1. dataA first-token accuracy: crash, then does it come back?  (prediction: it does not)
  2. delta, the common part of A's readout displacement (analyze_delta_structure): does it
     peak and reverse as in the standard arm (37% off its peak by step 200), or keep
     growing / stall?  (prediction: no reversal)
  3. eps, the individual part: should be unchanged in shape, monotone.
Report B's acquisition curve next to every one of these: if B is slower in this arm, the crash
is slower too, and "no recovery by step 1200" could be a slower clock rather than a missing
force. The 3000-step schedule exists for exactly that check.

Run:
  uv run python -m src.experiments.nofinalnorm_injection --pretrain_step 16000 --inject_seed 0
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
from src.experiments.nofinalnorm_common import (
    ARM, MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, create_nofinalnorm_model,
    nofinalnorm_pretrain_key, verify_no_final_norm,
)
from src.experiments.scale8_common import verify_architecture
from src.data.biography import NUM_ATTRIBUTES

# Run length only: injection uses a constant LR (InjectConfig overrides the cosine default),
# see the note in mlpfree_injection.py. 1200 matches the standard arm point for point; 3000
# extends the window to tell "no recovery" from "not yet".
SCHEDULES = {
    1200: "0,10,25,50,100,200,400,600,800,1000,1200",
    3000: "0,10,25,50,100,200,400,600,800,1000,1200,1600,2000,2400,3000",
}
PRETRAIN_TOTAL_STEPS = 16000  # must match nofinalnorm_pretrain.py exactly -- part of the key
PRETRAIN_PEAK_LR = 5e-4
PHASE = "nofinalnorminject"
PREFIX = "nofinalnorminject"


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, required=True,
                        choices=list(range(6000, 16001, 1000)))
    extra.add_argument("--inject_condition", choices=["high_overlap", "disjoint"],
                        default="disjoint")
    extra.add_argument("--inject_seed", type=int, required=True)
    extra.add_argument("--inject_total_steps", type=int, default=1200, choices=sorted(SCHEDULES))
    known, remaining = extra.parse_known_args()
    dense_steps = SCHEDULES[known.inject_total_steps]

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", known.inject_condition, "--inject.seed", str(known.inject_seed),
        "--inject.total_steps", str(known.inject_total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", dense_steps,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + remaining
    cfg = parse_config(KIConfig, description="No-final-norm ablation: injection")
    pop, _std_model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)

    model = create_nofinalnorm_model(cfg.model, data_cfg)

    pre_key = nofinalnorm_pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                             f"nofinalnorm-pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(
            f"No no-final-norm pretrain checkpoint at {pre_path}. Check --pretrain_step "
            f"matches one of the steps nofinalnorm_pretrain.py saved (6000-16000, every 1000).")
    print(f"Continuing from no-final-norm pretrain step {known.pretrain_step}: {pre_path}")

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_final_norm(state.params)

    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
    print(f"Optimizer: {'inherited' if not inject_cfg.fresh_optimizer else 'fresh'}")
    if pre_step != known.pretrain_step:
        raise ValueError(f"Loaded checkpoint's own step counter ({pre_step}) doesn't match "
                          f"the requested --pretrain_step ({known.pretrain_step}).")

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c) if d]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=PHASE,
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed, arm=ARM)
    print(f"Injecting on dataB, pretrain_step={known.pretrain_step}, "
          f"condition={known.inject_condition}, seed={known.inject_seed}, "
          f"total_steps={known.inject_total_steps}, checkpoints at {sorted(ckpt_steps)}, "
          f"key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir,
                          f"{PREFIX}-p{known.pretrain_step}-{key}-step{step:04d}.msgpack")
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
                    name=f"nofinalnorm-inject-p{known.pretrain_step}-{known.inject_condition}",
                    config={"pretrain_step": known.pretrain_step,
                            "condition": known.inject_condition,
                            "seed": known.inject_seed, "arm": ARM}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, data_b, f"{PREFIX}-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")


if __name__ == "__main__":
    main()
