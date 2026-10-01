"""No-final-norm arm: pretrain on A union D at the 8-layer/512-dim scale.

The control is the standard arm (scale8_pretrain.py). This driver is that driver with the
model factory swapped, and nothing else: same LR (5e-4), batch (256), gate (0.90 on both
dataA and ballast), budget (16,000 steps), and dense checkpoint schedule (6000..16000 every
1000). Where the arms must differ they differ loudly: the key carries arm="no_final_norm"
so the two arms cannot overwrite each other's checkpoints, and verify_no_final_norm proves
from the constructed weights that the final LayerNorm does not exist while every block's
pre-norms do.

The gate matters here more than usual. The prediction under test is about INJECTION (no
recovery without the final norm); it is only readable if this arm pretrains to the same
ceiling as the standard arm. If dataA or ballast fails 0.90, stop: the arm is then
uninformative (the freeze-arm rule applied to an architecture arm), and the right next step is
an LR probe for this arm, not injection. The per-checkpoint accuracy line is printed so a
slow-but-climbing curve can be told from a plateau.

Run:
  uv run python -m src.experiments.nofinalnorm_pretrain
"""
import os
import sys

import wandb

from src.config import parse_config
from src.experiments.ckpt import save_state
from src.experiments.knowledge_injection import (
    KIConfig, build, init_state, make_optimizer, population_baseline_nats, reset_stream,
    save_exposures, save_meta, train_loop,
)
from src.experiments.nofinalnorm_common import (
    ARM, MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, create_nofinalnorm_model,
    n_params as count_params, nofinalnorm_pretrain_key, verify_no_final_norm,
)
from src.experiments.scale8_common import occupancy, verify_architecture

PEAK_LR = 5e-4            # identical to scale8_pretrain.py -- one variable at a time
TOTAL_STEPS = 16000
EVAL_INTERVAL = 250
GATE_TARGET = 0.90
CHECKPOINT_STEPS = list(range(6000, 16001, 1000))
PREFIX = "nofinalnorm-pretrain"


def main():
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(TOTAL_STEPS),
        "--pretrain.batch_size", "256", "--pretrain.opt.peak_lr", str(PEAK_LR),
        "--pretrain.eval_interval", str(EVAL_INTERVAL), "--pretrain.target_acc", str(GATE_TARGET),
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + sys.argv[1:]
    cfg = parse_config(KIConfig, description="No-final-norm ablation: pretrain")
    pop, _std_model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.pretrain.max_eval_people)

    # The standard model returned by build() is discarded: this arm builds its own.
    model = create_nofinalnorm_model(cfg.model, data_cfg)

    tx = make_optimizer(cfg.pretrain.opt, cfg.pretrain.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    n_layers, d_model, n_p = verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_final_norm(state.params)
    print(f"Model parameters: {n_p:,} (standard arm minus exactly {2 * MODEL_DIM}: the final "
          f"LayerNorm's scale and bias)")

    occ, kb, cb = occupancy(n_p, cfg.num_a + cfg.num_ballast, pop.num_values_per_attr)
    print(f"Occupancy (N=A+ballast={cfg.num_a + cfg.num_ballast}, {n_p:,} params, "
          f"2 bits/param): knowledge={kb:.0f} bits, capacity={cb:.0f} bits, "
          f"occupancy={occ:.4%}\n")

    reset_stream(data_pre, cfg.seed + 1)
    key = nofinalnorm_pretrain_key(cfg, data_cfg)
    baseline = population_baseline_nats(pop)
    print(f"Checkpoint key (arm={ARM}): {key}")

    checkpoints_saved = {}

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir, f"{PREFIX}-{key}-step{step:05d}.msgpack")
        save_state(p, st)
        checkpoints_saved[step] = p
        print(f"  checkpoint step {step} -> {p}")

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name=PREFIX,
                    config={"model_dim": MODEL_DIM, "num_heads": NUM_HEADS,
                            "num_layers": NUM_LAYERS, "peak_lr": PEAK_LR, "arm": ARM}):
        state, final_loss, metrics = train_loop(
            state, data_pre, cfg.pretrain.total_steps, cfg.pretrain.batch_size,
            [d for d in (data_a, data_ballast, data_c) if d], cfg,
            eval_interval=cfg.pretrain.eval_interval,
            ckpt_steps=set(CHECKPOINT_STEPS), ckpt_fn=ckpt_fn, baseline=baseline)

    acc_a = metrics["dataA/first_token_accuracy"]
    acc_ballast = metrics["ballast/first_token_accuracy"]
    print(f"\n=== Final ({cfg.pretrain.total_steps} steps): dataA_acc={acc_a:.4f}  "
          f"ballast_acc={acc_ballast:.4f}  final_train_loss={final_loss:.4f} ===")

    gate_passed = acc_a >= GATE_TARGET and acc_ballast >= GATE_TARGET
    print(f"GATE (target {GATE_TARGET} on both): "
          f"dataA {'PASS' if acc_a >= GATE_TARGET else 'FAIL'}, "
          f"ballast {'PASS' if acc_ballast >= GATE_TARGET else 'FAIL'}")
    if not gate_passed:
        print("GATE FAILED. Do not inject from this checkpoint: an arm that does not reach "
              "the standard arm's ceiling cannot be read for recovery (its A would start "
              "from a different place). Probe the learning rate for this arm first.")

    if cfg.checkpoint_dir and cfg.pretrain.total_steps not in checkpoints_saved:
        ckpt_fn(cfg.pretrain.total_steps, state)

    if cfg.checkpoint_dir and checkpoints_saved:
        final_path = checkpoints_saved[cfg.pretrain.total_steps]
        save_meta(final_path, peak_lr=PEAK_LR, schedule=cfg.pretrain.opt.schedule,
                  total_steps=cfg.pretrain.total_steps, batch_size=cfg.pretrain.batch_size,
                  final_train_loss=final_loss, dataA_first_token_accuracy=acc_a,
                  ballast_first_token_accuracy=acc_ballast, baseline_nats=baseline, key=key,
                  model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                  occupancy=occ, arm=ARM)
        print(f"Saved metadata alongside {final_path}")

    save_exposures(cfg, data_pre, f"{PREFIX}-{key}")
    print(f"\nCheckpoints saved at steps: {sorted(checkpoints_saved.keys())}")


if __name__ == "__main__":
    main()
