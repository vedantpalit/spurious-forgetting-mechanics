"""Dose experiment: does the ||c|| turnover track B's resolution, or the step number?

(Pre-registered before submission.)

THE CLAIM. Recovery is the constant component being bought back as B's individuals become
mutually resolvable -- the coherent "push V_A down" write stops paying once individual-specific
structure has to be written into the same parameters.

WHAT IS MISSING. ||c|| falls 38-45% and B reaches ceiling around step 135. Those coincide, and
the toy supplies the counterfactual for why a shared carrier reverses while a private one does
not, but nothing shows IN THE TRANSFORMER that B's resolution drives the fall rather than both
being functions of step count.

THE MANIPULATION IS TIMING, NOT CARDINALITY. Vary B's acquisition SPEED and see whether the
turnover moves with it. Two knobs, both of which change when discrimination gets written
without changing what has to be learned:

    --lr_mult        0.5 / 1 / 2 x the matched injection LR   (step magnitude)
    --inject_batch   128 / 256 / 512                          (gradient noise, per-step signal)

Population size is deliberately not a knob here. What the claim is about is WHEN discrimination
becomes necessary, not how many individuals there are; changing cardinality changes the task
itself, moving the depression's magnitude, the number of distinct values and the resolution
point together. Cardinality tests a separate prediction -- that effect size scales with how much
coherent push B can generate -- which is about magnitude, not timing.

KEY COMPATIBILITY, DELIBERATE. Phase and filename prefix are left identical to
`mlpfree_injection`, so `--lr_mult 1.0 --inject_batch 256` reproduces the existing unfrozen
baseline's key EXACTLY and that condition is reused rather than re-run. Both knobs already
enter the key through the hashed inject config (peak_lr via inject.opt, batch_size directly),
so every other setting gets a distinct key automatically -- nothing is appended to the key.

Run:
  uv run python -m src.experiments.mlpfree_dose_injection \\
      --pretrain_step 16000 --inject_seed 0 --lr_mult 0.5
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
    INJECT_LR_RATIO, KIConfig, B_HALF, build, get_partition, inherit_moments, init_state,
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

# The turnover in ||c|| can only be located where a checkpoint exists, and the standard grid
# jumps 50 -> 100 -> 200. That is why two dose conditions both reported a turnover at step 50:
# they may have turned over at genuinely different steps and the grid cannot tell. The result
# was that ||c|| (r=0.9695, four points with two artificially identical) was measured WORSE
# than the accuracy proxy (r=0.9968 on the dense eval grid) -- backwards, since ||c|| is the
# quantity the mechanism claim is about.
#
# 5-step resolution from 20 to 140 brackets both the fastest condition's turnover (LR 2x, at
# 25 on the coarse grid) and the slowest (LR 0.5x, at 100), locating each to +/-2.5 steps.
# The sparse tail beyond 140 is kept so the endpoint and drop stay comparable to the other
# arms. 31 checkpoints x 12 runs x 41.3MB = ~15GB, which can be deleted once the
# decomposition passes have run.
DENSE_SCHEDULE = ",".join(str(x) for x in
                          [0] + list(range(20, 141, 5)) + [200, 400, 800, 1600, 3000])

# Figure 4 resolution for the t1200 run: every 10 steps through the trough (~60) and the recovery
# peak (~210), then every 50. 49 checkpoints x 3 seeds x 41.3MB = ~6GB. Hashed like DENSE.
FIG_SCHEDULE = ",".join(str(x) for x in list(range(0, 301, 10)) + list(range(350, 1201, 50)))


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, default=16000)
    extra.add_argument("--inject_condition", choices=["high_overlap", "disjoint"],
                       default="disjoint")
    extra.add_argument("--inject_seed", type=int, required=True)
    extra.add_argument("--inject_total_steps", type=int, default=3000, choices=sorted(SCHEDULES),
                       help="3000 by default: the slow conditions must run long enough for the "
                            "turnover to be OBSERVED rather than truncated")
    extra.add_argument("--lr_mult", type=float, default=1.0)
    extra.add_argument("--inject_batch", type=int, default=256)
    extra.add_argument("--dense_checkpoints", action="store_true",
                       help="5-step checkpoint resolution through the turnover region. "
                            "Changes inject.checkpoint_steps, which is hashed, so these are "
                            "DISTINCT runs and do not overwrite the coarse-grid ones.")
    extra.add_argument("--fig_checkpoints", action="store_true",
                       help="FIG_SCHEDULE (10-step resolution to 300, t1200 only); distinct runs")
    known, remaining = extra.parse_known_args()
    dense_steps = DENSE_SCHEDULE if known.dense_checkpoints else SCHEDULES[known.inject_total_steps]
    if known.fig_checkpoints:
        assert known.inject_total_steps == 1200, "FIG_SCHEDULE is defined for the t1200 run"
        dense_steps = FIG_SCHEDULE

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", "0",
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", known.inject_condition, "--inject.seed", str(known.inject_seed),
        "--inject.total_steps", str(known.inject_total_steps),
        "--inject.batch_size", str(known.inject_batch),
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", dense_steps,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + remaining
    cfg = parse_config(KIConfig, description="MLP-free dose experiment")
    pop, _std, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)

    pre_key = mlpfree_pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                            f"mlpfree-pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"No MLP-free pretrain checkpoint at {pre_path}")

    base_lr = PRETRAIN_PEAK_LR / INJECT_LR_RATIO
    inject_opt = replace(cfg.inject.opt, peak_lr=base_lr * known.lr_mult)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    print(f"\nDOSE CONDITION: lr_mult={known.lr_mult}  inject_lr={inject_opt.peak_lr:.3e} "
          f"(matched ratio gives {base_lr:.3e})   batch={inject_cfg.batch_size}")
    print(f"Continuing from MLP-free pretrain step {known.pretrain_step}: {pre_path}\n")

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_mlp(state.params)
    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c) if d]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    # phase and key_extra identical to mlpfree_injection ON PURPOSE: lr_mult=1, batch=256 then
    # reproduces the baseline key exactly and is reused rather than re-run. Both knobs already
    # enter the key through the hashed inject config, so nothing is appended.
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="mlpfreeinject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed, arm="mlp_free")
    print(f"Injecting on dataB, condition={inject_cfg.condition}, seed={inject_cfg.seed}, "
          f"total_steps={inject_cfg.total_steps}, checkpoints at {sorted(ckpt_steps)}, key={key}")

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
                    name=f"mlpfree-dose-lr{known.lr_mult}-bs{known.inject_batch}-p{known.pretrain_step}",
                    config={"pretrain_step": known.pretrain_step, "arm": "mlp_free",
                            "condition": inject_cfg.condition, "seed": inject_cfg.seed,
                            "lr_mult": known.lr_mult, "inject_batch": known.inject_batch}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)

    save_exposures(cfg, data_b, f"mlpfreeinject-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")


if __name__ == "__main__":
    main()
