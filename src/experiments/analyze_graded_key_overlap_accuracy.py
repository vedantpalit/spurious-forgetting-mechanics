"""Raw first-token accuracy by shared_count group, for the graded key-overlap
construction (follow-up to analyze_graded_key_overlap.py).

analyze_graded_key_overlap.py already answers the dose-response question, but only for
rank_own (own-half-restricted damage), and only as printed group-mean tables, never
saved to disk and never computed for the unrestricted metric. This script adds the
missing piece: per-person, per-step, full-vocabulary first-token correctness
(per_person_first_correct, reused unchanged from analyze_demo_injection.py), grouped by
shared_count (0-3 shared name parts with B), so the shared_count groups' raw accuracy
trajectories can be plotted directly, the same way the demo figure's A/B/D trajectories
were.

Reuses the graded population construction and checkpoint-key logic from
analyze_graded_key_overlap.py verbatim (deterministic given graded_key_seed, so build()
here reconstructs the identical B population and shared_count assignment).

Run (must match the graded injection run's own config exactly -- population sizes,
pretrain shape, inject.condition/seed, checkpoint_steps, graded_key_overlap/seed -- since
all of it is part of the checkpoint key):
  uv run python -m src.experiments.analyze_graded_key_overlap_accuracy \
    --num_a 2000 --num_ballast 2000 --num_b 500 --num_c 500 \
    --pretrain.total_steps 8000 --pretrain.batch_size 256 --pretrain.opt.peak_lr 5e-4 \
    --inject.condition disjoint --inject.seed 0 \
    --inject.total_steps 1200 --inject.batch_size 256 --inject.eval_interval 10 \
    --inject.checkpoint_steps 0,10,20,30,40,50,60,80,100,200,400,600,800,1000,1200 \
    --inject.graded_key_overlap true --inject.graded_key_seed 11 \
    --partition_path data/biography/value_partition.npz --seed 42
"""
import os
from dataclasses import replace

import numpy as np

from src.config import parse_config
from src.experiments.analyze_demo_injection import per_person_first_correct
from src.experiments.analyze_graded_key_overlap import STEPS
from src.experiments.analyze_name_overlap import per_person_rank_own
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    KIConfig, build, get_partition, init_state, make_optimizer, pretrain_key,
)

OUT_DIR = "demo_figure_data"


def load_inject_checkpoint_no_pretrain_file(cfg, data_cfg, model, step):
    """Same key/load logic as analyze_graded_key_overlap.load_inject_checkpoint, minus
    the pretrain-checkpoint file-existence check and meta.json read.

    Workaround for a real, confirmed problem, not a shortcut around an inconvenience:
    the shared crossover/graded pretrain checkpoint (historical key 0f9195ad349b under
    the old formula, 5f3432b7a70d under today's) was deleted from the cluster after this
    session's with-ballast demo-figure injection runs already completed successfully
    against it -- confirmed by checking the ONLY pretrain-*.msgpack actually present,
    which resolves to this session's unrelated without-ballast checkpoint (num_ballast=0),
    not the crossover/graded one.

    Safe to bypass because the pretrain checkpoint's own weights are never read here --
    only its KEY (a pure function of config, needed as an input to the injection
    checkpoint's key) and its recorded peak_lr (needed to reproduce that same key
    exactly). The peak_lr is not read from meta.json here; it must be supplied via
    --inject.opt.peak_lr explicitly. Every run in this project derives it the same way
    (pretrain peak / INJECT_LR_RATIO), so the value is a known constant, not a mystery:
    5e-4 / 13.333... = 3.75e-5.
    """
    if cfg.inject.opt.peak_lr <= 0:
        raise ValueError(
            "The pretrain checkpoint this run would normally read peak_lr from is "
            "missing on disk. Pass --inject.opt.peak_lr explicitly (3.75e-5 for every "
            "existing run in this project, derived from pretrain peak 5e-4 / "
            "INJECT_LR_RATIO) instead of relying on the default -1.0 sentinel.")
    pre_key = pretrain_key(cfg, data_cfg)
    inject_cfg = cfg.inject
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    tx = make_optimizer(inject_cfg.opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No step-{step} checkpoint at {p}")
    return load_params(p, template_state.params), key


def main():
    cfg = parse_config(KIConfig, description="Raw accuracy by shared_count group")
    if not cfg.inject.graded_key_overlap:
        raise ValueError(
            "This script analyzes the graded-overlap construction specifically; "
            "pass --inject.graded_key_overlap true, matching the injection run.")

    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(6)]

    ids_a, ids_b = data_a.person_ids, data_b.person_ids
    names_a, names_b = pop.person_names[ids_a], pop.person_names[ids_b]
    mult = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_b[:, j], return_counts=True)
        mult_map = dict(zip(vals.tolist(), counts.tolist()))
        mult[:, j] = [mult_map.get(v, 0) for v in names_a[:, j]]
    shared_count = (mult > 0).sum(axis=1)
    counts_hist = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"shared_count distribution across A: {counts_hist}")

    n = len(ids_a)
    first_acc_all = np.zeros((len(STEPS), n, 6), dtype=np.float64)
    rank_own_all = np.zeros((len(STEPS), n, 6), dtype=np.float64)

    for i, step in enumerate(STEPS):
        params, key = load_inject_checkpoint_no_pretrain_file(cfg, data_cfg, model, step)
        first_acc_all[i] = per_person_first_correct(model, params, data_a)
        rank_own_all[i] = per_person_rank_own(model, params, data_a, x_ids)
        group_means = [first_acc_all[i][shared_count == c].mean() for c in range(4)]
        print(f"step {step:5d}  key={key}  "
              f"first_acc by shared_count: " + "  ".join(
                  f"c={c}:{m:.3f}" for c, m in enumerate(group_means)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUT_DIR,
        f"graded-{cfg.inject.condition}-seed{cfg.inject.seed}-"
        f"gkseed{cfg.inject.graded_key_seed}.npz")
    np.savez(out_path, steps=np.array(STEPS), first_acc=first_acc_all,
              rank_own=rank_own_all, shared_count=shared_count, person_ids=ids_a)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
