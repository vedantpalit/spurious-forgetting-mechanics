"""MLP-free arm: pretrain on A union D at the 8-layer/512-dim scale.

This run IS the feasibility gate. There is no separate, cheaper gate: the gate exists to catch
a load infeasibility before an LR grid discovers it, and load is a property of the population
the model actually pretrains on -- so the gate is defined on the real pretrain configuration (A
union D at 2000/2000, real post-filter pools, real X/Y partition), not on a lighter A-alone
proxy.

Matched to scale8_pretrain.py on everything except the MLP removal: same LR (5e-4), batch
(256), gate target (0.90 on BOTH dataA and ballast), step budget (16,000), and dense
checkpoint schedule (6000..16000 every 1000). Where the two arms must differ, they differ
loudly: mlp_coefficient is 0 (so the config cannot claim an MLP the model lacks), the key
carries arm="mlp_free" (so the arms cannot overwrite each other's checkpoints), and
verify_no_mlp proves from the constructed weights that no MLP tensors exist.

Review point: the accuracy curve at the FIRST checkpoint, step 6000. If dataA accuracy is flat
and far from ceiling there, stop rather than burning the remaining 10,000 steps, and run
mlpfree_diagnostics.py -- an LR-independent plateau is a structural limit to be reported, not
undertuning to be tuned away.

Run:
  uv run python -m src.experiments.mlpfree_pretrain
"""
import os
import sys

import wandb

from src.config import parse_config
from src.experiments.ckpt import save_state
from src.experiments.knowledge_injection import (
    KIConfig, build, get_partition, init_state, make_optimizer, population_baseline_nats,
    reset_stream, save_exposures, save_meta, train_loop,
)
from src.experiments.mlpfree_common import (
    GateProbe, MODEL_DIM, NUM_HEADS, NUM_LAYERS, create_mlpfree_model, mlpfree_pretrain_key,
    print_load_ratio, verify_no_mlp,
)
from src.experiments.scale8_common import occupancy, verify_architecture

PEAK_LR = 5e-4            # identical to scale8_pretrain.py -- one variable at a time
TOTAL_STEPS = 16000
EVAL_INTERVAL = 250
GATE_TARGET = 0.90
# Checkpoints (full TrainState saved) match scale8_pretrain.py exactly.
CHECKPOINT_STEPS = list(range(6000, 16001, 1000))
# Extra *probe-only* steps: the gate probe (per-attribute accuracy + slope) runs at these
# too, but no state is written. Without points before 6000 there is no way to tell "flat"
# from "still climbing slowly" AT the step-6000 review point, and those are different
# readings -- slow-but-climbing is not a structural plateau.
PROBE_ONLY_STEPS = [250, 500, 1000, 2000, 3000, 4000, 5000]


def main():
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", "0",
        "--pretrain.total_steps", str(TOTAL_STEPS),
        "--pretrain.batch_size", "256", "--pretrain.opt.peak_lr", str(PEAK_LR),
        "--pretrain.eval_interval", str(EVAL_INTERVAL), "--pretrain.target_acc", str(GATE_TARGET),
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + sys.argv[1:]
    cfg = parse_config(KIConfig, description="MLP-free ablation: pretrain (and feasibility gate)")
    pop, _std_model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.pretrain.max_eval_people)

    # The standard model returned by build() is discarded: this arm builds its own.
    model = create_mlpfree_model(cfg.model, data_cfg)

    tx = make_optimizer(cfg.pretrain.opt, cfg.pretrain.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    n_layers, d_model, n_params = verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)
    verify_no_mlp(state.params)
    print(f"Model parameters: {n_params:,}")

    occ, kb, cb = occupancy(n_params, cfg.num_a + cfg.num_ballast, pop.num_values_per_attr)
    print(f"Occupancy (N=A+ballast={cfg.num_a + cfg.num_ballast}, {n_params:,} params, "
          f"2 bits/param): knowledge={kb:.0f} bits, capacity={cb:.0f} bits, "
          f"occupancy={occ:.4%}\n")

    # Logged whether the gate passes or fails -- the ratio is the comparison the writeup
    # has to make either way.
    halves = get_partition(cfg, pop)
    load_rows, _load_summary = print_load_ratio(
        pop, halves, cfg.num_a, cfg.num_ballast, cfg.data.support_size)
    probe = GateProbe(load_rows, occ, n_params)

    reset_stream(data_pre, cfg.seed + 1)
    key = mlpfree_pretrain_key(cfg, data_cfg)
    baseline = population_baseline_nats(pop)
    print(f"Checkpoint key (arm=mlp_free): {key}")

    checkpoints_saved = {}

    def ckpt_fn(step, st):
        # Runs at CHECKPOINT_STEPS + PROBE_ONLY_STEPS. Always probes; only writes state at
        # real checkpoint steps, so the pre-6000 trajectory costs one eval pass each and no
        # disk.
        probe.probe(step, st, data_a)
        if step not in CHECKPOINT_STEPS or not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir, f"mlpfree-pretrain-{key}-step{step:05d}.msgpack")
        save_state(p, st)
        checkpoints_saved[step] = p
        print(f"  checkpoint step {step} -> {p}")

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="mlpfree-pretrain",
                    config={"model_dim": MODEL_DIM, "num_heads": NUM_HEADS,
                            "num_layers": NUM_LAYERS, "peak_lr": PEAK_LR, "arm": "mlp_free"}):
        state, final_loss, metrics = train_loop(
            state, data_pre, cfg.pretrain.total_steps, cfg.pretrain.batch_size,
            [d for d in (data_a, data_ballast, data_c) if d], cfg,
            eval_interval=cfg.pretrain.eval_interval,
            ckpt_steps=set(CHECKPOINT_STEPS) | set(PROBE_ONLY_STEPS),
            ckpt_fn=ckpt_fn, baseline=baseline)

    acc_a = metrics["dataA/first_token_accuracy"]
    acc_ballast = metrics["ballast/first_token_accuracy"]
    print(f"\n=== Final ({cfg.pretrain.total_steps} steps): dataA_acc={acc_a:.4f}  "
          f"ballast_acc={acc_ballast:.4f}  final_train_loss={final_loss:.4f} ===")

    # Only if the training loop's own ckpt_fn didn't already probe this step -- otherwise
    # the final step appears twice in the trajectory line.
    if cfg.pretrain.total_steps not in (s for s, _, _ in probe.history):
        probe.probe(cfg.pretrain.total_steps, state, data_a)

    gate_passed = acc_a >= GATE_TARGET and acc_ballast >= GATE_TARGET
    print(f"FEASIBILITY GATE (target {GATE_TARGET} on both): "
          f"dataA {'PASS' if acc_a >= GATE_TARGET else 'FAIL'}, "
          f"ballast {'PASS' if acc_ballast >= GATE_TARGET else 'FAIL'}")
    # Occupancy restated at the verdict, not only at startup: without it a reader assumes
    # removing 62% of the parameters caused a capacity shortfall, and this number is what
    # rules that out. It also matches the project's framing that damage is not scarcity.
    print(f"  occupancy at this verdict: {occ:.4%} ({n_params:,} params, N="
          f"{cfg.num_a + cfg.num_ballast}). Whatever this gate did, it did it in a store "
          f"filled to under one percent of the Allen-Zhu capacity estimate -- a gate failure "
          f"here is NOT a capacity shortfall.")
    if not gate_passed:
        print("GATE FAILED. Do not proceed to injection, and do not widen an LR grid. Run "
              "src.experiments.mlpfree_diagnostics next: load ratio (already printed above), "
              "key span/conditioning, capacity, and only then learning rate. An "
              "LR-independent plateau is a structural limit, which is a result, not an "
              "obstacle -- but check the slope printed in the probes above first: a curve "
              "that still has slope is not a structural limit.")

    if cfg.checkpoint_dir and cfg.pretrain.total_steps not in checkpoints_saved:
        ckpt_fn(cfg.pretrain.total_steps, state)

    if cfg.checkpoint_dir and checkpoints_saved:
        final_path = checkpoints_saved[cfg.pretrain.total_steps]
        save_meta(final_path, peak_lr=PEAK_LR, schedule=cfg.pretrain.opt.schedule,
                  total_steps=cfg.pretrain.total_steps, batch_size=cfg.pretrain.batch_size,
                  final_train_loss=final_loss, dataA_first_token_accuracy=acc_a,
                  ballast_first_token_accuracy=acc_ballast, baseline_nats=baseline, key=key,
                  model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                  occupancy=occ, arm="mlp_free")
        print(f"Saved metadata alongside {final_path}")

    save_exposures(cfg, data_pre, f"mlpfree-pretrain-{key}")
    print(f"\nCheckpoints saved at steps: {sorted(checkpoints_saved.keys())}")


if __name__ == "__main__":
    main()
