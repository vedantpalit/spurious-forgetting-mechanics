"""Full pretrain for the 8-layer/512-dim scale check, at the LR chosen from
scale8_lr_probe.py's result (5e-4, matching the pilot's own LR -- both 5e-4 and 1e-3 reached
ceiling in the probe, 5e-4 kept for consistency with the pilot rather than introducing a second
variable alongside the architecture change).

Does not assume 8000 steps (the pilot's own budget) is right at this scale. Runs a
16,000-step budget (Zucchet's own step count, not an arbitrary guess) with checkpoints
every 1000 steps from 6000 through 16000, so whichever step turns out to be converged
already has a usable checkpoint -- no wasted rerun. Reports dataA/ballast accuracy at
every checkpoint explicitly, confirms both cross the 0.90 gate, and computes occupancy
from the real (weights-derived) parameter count.

Run:
  uv run python -m src.experiments.scale8_pretrain
"""
import os
import sys
from dataclasses import replace

import wandb

from src.config import parse_config
from src.experiments.ckpt import ckpt_path, save_state
from src.experiments.knowledge_injection import (
    KIConfig, build, init_state, make_optimizer, population_baseline_nats, pretrain_key,
    reset_stream, save_exposures, save_meta, train_loop,
)
from src.experiments.scale8_common import (
    MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, occupancy, verify_architecture,
)

PEAK_LR = 5e-4
TOTAL_STEPS = 16000
EVAL_INTERVAL = 250
GATE_TARGET = 0.90
# Dense in the back half only -- no reason to checkpoint before the model could
# plausibly be near convergence, and every earlier step is cheap to re-derive from the
# training curve alone (printed every EVAL_INTERVAL regardless of checkpointing).
CHECKPOINT_STEPS = list(range(6000, 16001, 1000))


def _nonempty(ds):
    """Is this dataset actually populated?

    `if ds` is NOT this test. BiographyDataset defines neither __len__ nor __bool__, so a
    0-person dataset -- which is what build() returns for --num_ballast 0 -- is truthy, passes
    the usual `[d for d in (...) if d]` filter, and then crashes inside evaluate(): the batch
    loop never runs, the accumulator stays None, and _finalize_metrics(None) subscripts it.
    data_c gets guarded in build() (`if len(ids_c) else None`); ballast does not.
    """
    return ds is not None and len(ds.person_ids) > 0


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
    cfg = parse_config(KIConfig, description="8-layer scale check: full pretrain")
    pop, model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.pretrain.max_eval_people)

    tx = make_optimizer(cfg.pretrain.opt, cfg.pretrain.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    n_layers, d_model, n_params = verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    print(f"Model parameters: {n_params:,}")

    occ, kb, cb = occupancy(n_params, cfg.num_a + cfg.num_ballast, pop.num_values_per_attr)
    print(f"Occupancy (N=A+ballast={cfg.num_a + cfg.num_ballast}, {n_params:,} params, "
          f"2 bits/param): knowledge={kb:.0f} bits, capacity={cb:.0f} bits, "
          f"occupancy={occ:.4%}\n")

    reset_stream(data_pre, cfg.seed + 1)
    key = pretrain_key(cfg, data_cfg)
    baseline = population_baseline_nats(pop)

    checkpoints_saved = {}

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir, f"pretrain-{key}-step{step:05d}.msgpack")
        save_state(p, st)
        checkpoints_saved[step] = p
        print(f"  checkpoint step {step} -> {p}")

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="scale8-pretrain",
                    config={"model_dim": MODEL_DIM, "num_heads": NUM_HEADS,
                            "num_layers": NUM_LAYERS, "peak_lr": PEAK_LR}):
        state, final_loss, metrics = train_loop(
            state, data_pre, cfg.pretrain.total_steps, cfg.pretrain.batch_size,
            [d for d in (data_a, data_ballast, data_c) if _nonempty(d)], cfg,
            eval_interval=cfg.pretrain.eval_interval,
            ckpt_steps=set(CHECKPOINT_STEPS), ckpt_fn=ckpt_fn, baseline=baseline)

    acc_a = metrics["dataA/first_token_accuracy"]
    has_ballast = _nonempty(data_ballast)
    acc_ballast = metrics["ballast/first_token_accuracy"] if has_ballast else float("nan")
    print(f"\n=== Final ({cfg.pretrain.total_steps} steps): dataA_acc={acc_a:.4f}  "
          f"ballast_acc={'n/a (num_ballast=0)' if not has_ballast else f'{acc_ballast:.4f}'}  "
          f"final_train_loss={final_loss:.4f} ===")

    gate_passed = acc_a >= GATE_TARGET and (not has_ballast or acc_ballast >= GATE_TARGET)
    print(f"GATE (target {GATE_TARGET}): dataA {'PASS' if acc_a >= GATE_TARGET else 'FAIL'}"
          + ("" if not has_ballast else
             f", ballast {'PASS' if acc_ballast >= GATE_TARGET else 'FAIL'}"))
    if not gate_passed:
        print("BELOW TARGET -- do not proceed to injection using this checkpoint.")

    # Final-step checkpoint, saved regardless of whether it lands on the dense
    # schedule above, so the gate result and the saved weights always correspond to
    # the exact same step even if total_steps isn't a multiple of 1000.
    if cfg.checkpoint_dir and cfg.pretrain.total_steps not in checkpoints_saved:
        ckpt_fn(cfg.pretrain.total_steps, state)

    if cfg.checkpoint_dir and checkpoints_saved:
        final_path = checkpoints_saved[cfg.pretrain.total_steps]
        save_meta(final_path, peak_lr=PEAK_LR, schedule=cfg.pretrain.opt.schedule,
                  total_steps=cfg.pretrain.total_steps, batch_size=cfg.pretrain.batch_size,
                  final_train_loss=final_loss, dataA_first_token_accuracy=acc_a,
                  ballast_first_token_accuracy=(acc_ballast if has_ballast else None),
                  baseline_nats=baseline, key=key,
                  model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                  occupancy=occ)
        print(f"Saved metadata alongside {final_path}")

    save_exposures(cfg, data_pre, f"scale8-pretrain-{key}")
    print(f"\nCheckpoints saved at steps: {sorted(checkpoints_saved.keys())}")
    print("Review dataA/ballast accuracy at each dense eval point above to pick the "
          "converged step for injection -- do not assume the final step is optimal "
          "just because it's last.")


if __name__ == "__main__":
    main()
