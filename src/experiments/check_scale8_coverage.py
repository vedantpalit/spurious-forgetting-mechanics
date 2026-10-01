"""Verify scale8 checkpoint COVERAGE, not count, before deciding to reuse the standard arm.

Why a count is not enough: the injection filename is
`scale8inject-p{pretrain_step}-{key}-step{NNNN}.msgpack`. Condition and seed appear nowhere
in it -- they are hashed into `key` via `experiment_key(..., inject=inject_cfg, ...)`. So
"660 files" is consistent with many different mixes of conditions and seeds, including ones
missing every `disjoint` run we actually want. The only way to check is to recompute each
expected key and look for it.

That recomputation doubles as a config-drift check, for free. The key is a fingerprint of
the model config, the data fingerprint (post-replace data_cfg + a content hash of the .npz +
the data-module sources), the resolved inject config, and the pretrain key. If a recomputed
key matches a file on disk, the configuration behind that checkpoint is identical to what
today's code produces. If it does not match, either the checkpoint is for a different
condition/seed or something has drifted -- and either way reusing it would be wrong.

Note the derived-LR trap this deliberately avoids: the injection key must be computed from
the FULLY RESOLVED inject config, with peak_lr already replaced by
PRETRAIN_PEAK_LR / INJECT_LR_RATIO. Hashing the un-derived sentinel gives a different, wrong
key -- a bug previously hit in analyze_demo_injection.py.

Run on the cluster (it needs the checkpoint directory):
  uv run python -m src.experiments.check_scale8_coverage --checkpoint_dir checkpoints
"""
import argparse
import os
import sys
from dataclasses import replace

from src.config import parse_config
from src.experiments.ckpt import experiment_key
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, parse_steps, pretrain_key,
)
from src.experiments.scale8_common import MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS

DENSE_CHECKPOINT_STEPS = "0,10,25,50,100,200,400,600,800,1000,1200"
PRETRAIN_TOTAL_STEPS = 16000
PRETRAIN_PEAK_LR = 5e-4
PRETRAIN_STEPS = list(range(6000, 16001, 1000))
CONDITIONS = ["high_overlap", "disjoint"]
SEEDS = [0, 1, 2, 3, 4]

# Pretraining checkpoints that were actually injected from in the duration sweep. The odd
# thousands (7000, 9000, ...) have pretrain checkpoints but were never used as injection
# starting points, so their absence below is "never run", not a gap -- distinguished in the
# summary so a reader does not misread 60-of-110 as patchy.
INJECTED_FROM_STEPS = [6000, 8000, 10000, 12000, 14000, 16000]

# What the MLP-free comparison actually needs.
WANTED_PRETRAIN_STEPS = [6000, 16000]
WANTED_CONDITION = "disjoint"


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--checkpoint_dir", default="checkpoints")
    known, remaining = extra.parse_known_args()

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.total_steps", "1200", "--inject.batch_size", "256",
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", DENSE_CHECKPOINT_STEPS,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ] + remaining
    cfg = parse_config(KIConfig, description="scale8 checkpoint coverage check")

    ckpt_dir = known.checkpoint_dir
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"No such checkpoint dir: {ckpt_dir}")
    on_disk = set(os.listdir(ckpt_dir))
    print(f"Checkpoint dir: {ckpt_dir}  ({len(on_disk)} files)\n")

    # --- pretrain coverage ---
    # build() is condition-dependent only through B's value half, which does not enter the
    # pretrain key; any condition gives the same pretrain key.
    cfg_pre = replace(cfg, inject=replace(cfg.inject, condition="disjoint"))
    _pop, _model, data_cfg, *_ = build(cfg_pre, cfg_pre.pretrain.max_eval_people)
    pre_key = pretrain_key(cfg_pre, data_cfg)
    print(f"Recomputed pretrain key: {pre_key}")
    print(f"{'step':>7}  {'present':>7}")
    pre_missing = []
    for s in PRETRAIN_STEPS:
        fname = f"pretrain-{pre_key}-step{s:05d}.msgpack"
        ok = fname in on_disk
        want = " <- WANTED" if s in WANTED_PRETRAIN_STEPS else ""
        print(f"{s:>7}  {'yes' if ok else 'NO':>7}{want}")
        if not ok:
            pre_missing.append(s)
    print()

    # --- injection coverage ---
    ckpt_steps = sorted(parse_steps(DENSE_CHECKPOINT_STEPS, cfg.inject.total_steps))
    print(f"Injection dense steps expected per run: {ckpt_steps}")
    print(f"{'pstep':>7} {'condition':>13} {'seed':>5} {'key':>14} {'files':>7}  status")
    rows = []
    for pstep in PRETRAIN_STEPS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                c = replace(cfg, inject=replace(cfg.inject, condition=condition, seed=seed))
                # Resolve the derived LR before hashing -- see module docstring.
                inject_opt = replace(c.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
                inject_cfg = replace(c.inject, opt=inject_opt)
                key = experiment_key(
                    c.model, data_cfg, c.data.biography_data_path, phase="scale8inject",
                    inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
                    pretrain_step=pstep, seed=c.seed)
                found = sum(
                    1 for s in ckpt_steps
                    if f"scale8inject-p{pstep}-{key}-step{s:04d}.msgpack" in on_disk)
                complete = found == len(ckpt_steps)
                wanted = (pstep in WANTED_PRETRAIN_STEPS and condition == WANTED_CONDITION)
                status = ("complete" if complete else
                           ("MISSING" if found == 0 else f"PARTIAL ({found})"))
                if wanted:
                    status += "  <- WANTED"
                rows.append((pstep, condition, seed, key, found, complete, wanted))
                print(f"{pstep:>7} {condition:>13} {seed:>5} {key:>14} "
                      f"{found:>3}/{len(ckpt_steps):<3}  {status}")

    wanted_rows = [r for r in rows if r[6]]
    wanted_complete = [r for r in wanted_rows if r[5]]
    all_complete = [r for r in rows if r[5]]
    print()
    expected_rows = [r for r in rows if r[0] in INJECTED_FROM_STEPS]
    never_run = [r for r in rows if r[0] not in INJECTED_FROM_STEPS]
    unexpected_gaps = [r for r in expected_rows if not r[5]]
    print(f"=== Summary ===")
    print(f"  runs enumerated:            {len(rows)}")
    print(f"  runs complete on disk:      {len(all_complete)}")
    print(f"  of those enumerated, {len(never_run)} are at pretrain steps that were never "
          f"injected from ({[s for s in PRETRAIN_STEPS if s not in INJECTED_FROM_STEPS]}) "
          f"-- absent by design, not a gap")
    print(f"  genuine gaps among the {len(expected_rows)} runs that SHOULD exist: "
          f"{len(unexpected_gaps)}")
    print(f"  WANTED runs (p{WANTED_PRETRAIN_STEPS}, {WANTED_CONDITION}): "
          f"{len(wanted_complete)}/{len(wanted_rows)} complete")
    print(f"  pretrain steps missing:     {pre_missing or 'none'}")
    print()
    if len(wanted_complete) == len(wanted_rows) and not [
            s for s in WANTED_PRETRAIN_STEPS if s in pre_missing]:
        print("COVERAGE COMPLETE for what the MLP-free comparison needs. The standard arm "
              "does NOT need re-running: recompute rank_own_mean offline from these "
              "checkpoints (analyze_concentration_trough.py precedent).")
    else:
        print("COVERAGE INCOMPLETE or keys do not match what today's code computes (config "
              "drift). Fall back to running both arms fresh with matched config.")


if __name__ == "__main__":
    main()
