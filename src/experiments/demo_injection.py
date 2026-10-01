"""Injection driver for the first-look demonstration figure (both arms).

Both arms use --inject.condition=disjoint, asserted below, not left as a free choice.
Reasoning: A must play the OPPOSITE-half role relative to B in BOTH arms for its own
trough to be the dramatic, "unmissable" one -- per the already-established crossover
numbers, A's own trough is ~0.79-0.792 as the SAME-half population under high_overlap
(B draws from A's own half), vs. ~0.154-0.158 as the OPPOSITE-half population under
disjoint (B draws from the half A never touches). "B fine-tuned in on A's half" (the
with-ballast arm's original phrasing) is literally high_overlap and would give A only the
mild dip -- so disjoint is used instead, and both arms show the same structural
relationship and are a fair same-shape comparison.

Arm is inferred from cfg.num_ballast (0 -> without_ballast, >0 -> with_ballast), the same
convention demo_pretrain_no_ballast.py uses -- no new KIConfig field needed.

With-ballast reuses the EXISTING crossover pretrain checkpoint (crossover_pretrain_key,
the historical pre-checkpoint-key-fix formula; see analyze_name_overlap.py's docstring
for why pretrain_key() itself would silently resolve to the wrong checkpoint here).
Without-ballast uses the fresh checkpoint from demo_pretrain_no_ballast.py (standard
pretrain_key()) and must exclude data_ballast from eval_sets/rank_ctx: with num_ballast=0
data_ballast is a valid but empty dataset, and BiographyDataset has no __len__/__bool__
override, so an unguarded eval_sets list would still call .evaluate() on it and crash --
same root cause and same fix as demo_pretrain_no_ballast.py.

Checkpoints are saved under a distinct "demoinject" phase/kind, not "inject" or "celeb",
so there is no collision risk with the historical crossover or the celebrities run.

Per-person logging (rank_own, raw first-token correctness, for all of A, at every dense
checkpoint) is NOT computed inline here -- see analyze_demo_injection.py. Checkpoints
capture the exact trained weights, so a separate post-hoc CPU pass loses nothing and
matches this codebase's established train/analyze split.

Run (with-ballast, one seed):
  uv run python -m src.experiments.demo_injection \
    --num_a 2000 --num_ballast 2000 --num_b 500 --num_c 500 \
    --inject.condition disjoint --inject.seed 0 \
    --inject.total_steps 1200 --inject.checkpoint_steps 0,10,25,50,100,200,400,600,800,1000,1200 \
    --partition_path data/biography/value_partition.npz

Run (without-ballast, one seed): SAME --partition_path as with-ballast, deliberately --
B's region size must match across arms so ballast presence is the only variable that
differs (an earlier version of this driver used a smaller f0.85 partition here, which
confounded "no ballast" with "smaller B region" -- fixed).
  uv run python -m src.experiments.demo_injection \
    --num_a 2000 --num_ballast 0 --num_b 500 --num_c 500 \
    --inject.condition disjoint --inject.seed 0 \
    --inject.total_steps 1200 --inject.checkpoint_steps 0,10,25,50,100,200,400,600,800,1000,1200 \
    --partition_path data/biography/value_partition.npz
"""
import os
from dataclasses import replace

import wandb

from src.config import parse_config, vars_nested
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.analyze_name_overlap import crossover_pretrain_key
from src.experiments.ckpt import ckpt_path, experiment_key, load_state, save_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, get_partition, inherit_moments, init_state,
    load_meta, make_optimizer, meta_path, own_half_baseline_nats, parse_steps,
    population_baseline_nats, pretrain_key, reset_stream, save_exposures, train_loop,
)


def demo_inject_key(cfg, data_cfg, pre_key, inject_cfg):
    """Shared with analyze_demo_injection.py so the analysis script reloads exactly the
    checkpoints this driver saved."""
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="demoinject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)


def main():
    cfg = parse_config(KIConfig, description="Demo figure: injection driver (both arms)")
    if cfg.inject.condition != "disjoint":
        raise ValueError(
            f"This demo figure needs A to play the opposite-half role in both arms (the "
            f"dramatic ~0.15 trough, not the mild ~0.79 same-half one -- see module "
            f"docstring); got --inject.condition={cfg.inject.condition!r}, expected "
            f"'disjoint'.")
    arm = "without_ballast" if cfg.num_ballast == 0 else "with_ballast"
    print(f"Arm: {arm} (inferred from num_ballast={cfg.num_ballast})")

    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    if arm == "without_ballast":
        assert len(data_ballast.person_ids) == 0, "arm=without_ballast but ballast is non-empty"
    else:
        assert len(data_ballast.person_ids) > 0, "arm=with_ballast but ballast is empty"

    pre_key = (crossover_pretrain_key(cfg, data_cfg) if arm == "with_ballast"
               else pretrain_key(cfg, data_cfg))
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        hint = ("Run demo_pretrain_no_ballast.py first." if arm == "without_ballast"
                else "Expected the existing crossover pretrain checkpoint.")
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path} for arm={arm}. {hint}")
    meta = load_meta(pre_path) if os.path.exists(meta_path(pre_path)) else None

    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        if meta is None:
            raise FileNotFoundError(
                f"No metadata at {meta_path(pre_path)}; set --inject.opt.peak_lr explicitly.")
        derived = float(meta["peak_lr"]) / INJECT_LR_RATIO
        inject_opt = replace(inject_opt, peak_lr=derived)
        print(f"Injection LR derived: pretrain peak {float(meta['peak_lr']):g} / "
              f"{INJECT_LR_RATIO:.4g} = {derived:g} (constant)")
    else:
        print(f"Injection LR set explicitly: {inject_opt.peak_lr:g} ({inject_opt.schedule})")
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
        print("Optimizer: inherited mu/nu from pretraining, step and schedule reset")
    else:
        print("Optimizer: fresh")

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = ([d for d in (data_a, data_b, data_ballast, data_c) if d] if arm == "with_ballast"
                 else [d for d in (data_a, data_b, data_c) if d])
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = demo_inject_key(cfg, data_cfg, pre_key, inject_cfg)
    print(f"Injecting on dataB, arm={arm}, checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir, f"demoinject-{arm}-{key}-step{step:04d}.msgpack")
        save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

    baseline = population_baseline_nats(pop)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    own_halves = {"dataA": "X", "dataB": B_HALF[inject_cfg.condition]}
    if arm == "with_ballast":
        own_halves["ballast"] = "Y"
    if cfg.c_half in ("X", "Y"):
        own_halves["dataC"] = cfg.c_half
    rank_ctx = {
        "x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
        "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                          "Y": own_half_baseline_nats(halves, "Y")},
    }

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name=f"demo-inject-{arm}",
                    config={**vars_nested(cfg), "arm": arm}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, data_b, f"demoinject-{arm}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")
    return state


if __name__ == "__main__":
    main()
