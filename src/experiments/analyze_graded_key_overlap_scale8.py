"""Raw first-token accuracy by shared_count, on the 8-layer STANDARD arm (with MLP).

The graded name-overlap dose-response (and its accuracy follow-up) was measured
on the 4-layer pilot -- ModelConfig defaults 256/4/4, pretrain 8000, phase "inject". This is the
same analysis against the 8-layer standard arm: 512/8/8, mlp_coefficient 4, pretrain 16000,
phase "scale8inject".

WHY THE STANDARD ARM AND NOT mlp_free. The claim under test is that a gradient step writing B's
facts lands on the same key region storing an A individual's facts through NAME proximity alone,
with zero value overlap -- feedforward layers as associative memories keyed roughly by name. The
mlp_free arm has no feedforward layers, so a null there would say nothing about the mechanism.

WHY THIS IS A NEW SCRIPT rather than an argument to the 4-layer one. The pilot script hardcodes
phase="inject" and the `inject-{key}-step{s}.msgpack` filename, and its key derivation takes no
pretrain_step. The 8-layer key needs all three different. The pilot version is left untouched so
the existing appendix figure stays reproducible from it.

THE CONSTRUCTION IS DETERMINISTIC given graded_key_seed, so `build()` here reconstructs the same
B population and the same shared_count assignment the injection run used -- the assignment is not
stored, it is rebuilt. `--inject.graded_key_overlap true` is required and checked, because
without it this would silently analyse the chance population instead of the graded one, and the
chance population is exactly what the graded construction exists to replace.

Run (flags must match the injection run's, since they are all part of the key):
  uv run python -m src.experiments.analyze_graded_key_overlap_scale8 \
    --pretrain_step 16000 --inject.condition disjoint --inject.seed 0 \
    --inject.graded_key_overlap true --inject.graded_key_seed 11
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np

from src.config import parse_config
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.analyze_demo_injection import per_person_first_correct
from src.experiments.analyze_name_overlap import per_person_rank_own
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, get_partition, init_state, make_optimizer, pretrain_key,
)
from src.experiments.scale8_common import (
    MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, verify_architecture,
)

OUT_DIR = "graded_scale8"
# Identical to scale8_injection.DENSE_CHECKPOINT_STEPS -- part of the key, so it is not a
# free choice here.
STEPS = [0, 10, 25, 50, 100, 200, 400, 600, 800, 1000, 1200]
PRETRAIN_TOTAL_STEPS = 16000
PRETRAIN_PEAK_LR = 5e-4


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, default=16000)
    known, remaining = extra.parse_known_args()

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.total_steps", "1200", "--inject.batch_size", "256",
        "--inject.eval_interval", "10",
        "--inject.checkpoint_steps", ",".join(str(s) for s in STEPS),
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ] + remaining
    cfg = parse_config(KIConfig, description="graded shared_count accuracy, 8-layer standard")
    if not cfg.inject.graded_key_overlap:
        raise SystemExit(
            "--inject.graded_key_overlap true is required. Without it this analyses the CHANCE "
            "name-overlap population, which concentrates at shared_count=3 and is precisely "
            "what the graded construction replaces -- a silently different experiment.")

    pop, model, data_cfg, _, data_a, _bal, data_b, _c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    pre_key = pretrain_key(cfg, data_cfg)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="scale8inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed)
    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_opt, inject_cfg.total_steps), cfg.seed + 20)
    verify_architecture(template.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)

    def path_for(s):
        return os.path.join(cfg.checkpoint_dir,
                            f"scale8inject-p{known.pretrain_step}-{key}-step{s:04d}.msgpack")

    missing = [s for s in STEPS if not os.path.exists(path_for(s))]
    if missing:
        raise SystemExit(f"checkpoints missing for steps {missing}\n  e.g. {path_for(missing[0])}"
                         f"\n  key={key}. Run the graded injection first.")

    # shared_count: how many of {first, middle, last} an A individual shares with ANY B
    # individual. Rebuilt from the populations, exactly as the 4-layer script does.
    ids_a, ids_b = data_a.person_ids, data_b.person_ids
    names_a, names_b = pop.person_names[ids_a], pop.person_names[ids_b]
    mult = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_b[:, j], return_counts=True)
        m = dict(zip(vals.tolist(), counts.tolist()))
        mult[:, j] = [m.get(v, 0) for v in names_a[:, j]]
    shared_count = (mult > 0).sum(axis=1)
    hist = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"key={key}")
    print(f"shared_count distribution across A: {hist}")
    if min(hist.values()) < 50:
        print(f"  WARNING: a group has under 50 individuals; the graded split did not come out "
              f"even and the group means will be noisy")

    n = len(ids_a)
    first_acc = np.zeros((len(STEPS), n, NUM_ATTRIBUTES))
    rank_own = np.zeros((len(STEPS), n, NUM_ATTRIBUTES))
    for i, s in enumerate(STEPS):
        params = load_params(path_for(s), template.params)
        first_acc[i] = per_person_first_correct(model, params, data_a)
        rank_own[i] = per_person_rank_own(model, params, data_a, x_ids)
        gm = [first_acc[i][shared_count == c].mean() for c in range(4)]
        gr = [rank_own[i][shared_count == c].mean() for c in range(4)]
        print(f"step {s:>5}  first_acc " + "  ".join(f"c={c}:{v:.3f}" for c, v in enumerate(gm))
              + f"   | rank_own_top1 " + "  ".join(f"{v:.3f}" for v in gr))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = (f"scale8-graded-p{known.pretrain_step}-{cfg.inject.condition}-"
            f"gk{cfg.inject.graded_key_seed}-seed{cfg.inject.seed}")
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(STEPS), shared_count=shared_count,
        first_acc=first_acc.astype(np.float32), rank_own=rank_own.astype(np.float32),
        key=key, pretrain_step=known.pretrain_step, condition=cfg.inject.condition,
        graded_key_seed=cfg.inject.graded_key_seed, seed=cfg.inject.seed)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
