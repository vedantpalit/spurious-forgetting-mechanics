"""Analysis for the deliberately-graded key-overlap construction: does engineering A's
name-part overlap with B into a real 0/1/2/3-shared-part distribution (rather than the
chance distribution's ~75%-at-3) produce a dose-response, and does it make the erosion
effect emerge earlier than the chance-level baseline's step 100?

Population construction (assign_graded_b_names, knowledge_injection.py, triggered by
InjectConfig.graded_key_overlap) is deterministic given graded_key_seed, so calling
build() here with the same flags used for injection reconstructs the identical B
population -- no separate bookkeeping of which tokens were reserved is needed; the
realized pop.person_names[ids_b] is read directly, exactly as analyze_name_overlap.py
does for the chance-level population.

Unlike analyze_name_overlap.py, this is a NEW run built with today's code, so the
standard pretrain_key()/experiment_key() formulas apply directly -- no historical-key
reconstruction needed.

The primary comparison this script exists to support is NOT internal to this run: it's
matched-step, point-for-point, against the chance-level baseline (analyze_name_overlap
.py, STEPS now including 10 and 25) at steps 10/25/50/100. That comparison is assembled by
reading both scripts' printed emergence curves side by side, not computed in either script,
since the two populations require different checkpoint keys and cannot share a run.

shared_count (0-3 shared parts) is the primary dose-response table here, not a footnote:
the whole point of the construction is that it now has real, engineered spread (~250 /
750 / 750 / 250 out of 2000, vs. the chance population's ~1 / 34 / 461 / 1504). The
continuous overlap_score regression and the intrinsic-crowding (overlap_A) control are
kept for the same reasons as analyze_name_overlap.py -- rigor and confound-closing,
respectively -- not because the coarse dose-response needs them to be readable.
"""
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.experiments.analyze_name_overlap import (
    PARTS, decile_report, ols_fit, partial_r2, per_person_margin_own, per_person_rank_own,
    r_squared, report_group, standardize, token_fixed_effects_test, token_level_correlation,
)
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key, get_partition,
    init_state, load_meta, make_optimizer, pretrain_key,
)

# Dense through the window where the chance-level baseline first became
# statistically real (step 100), plus the usual grid after.
STEPS = (10, 20, 30, 40, 50, 60, 80, 100, 200, 400, 600, 800, 1000, 1200)


def load_inject_checkpoint(cfg, data_cfg, model, step):
    """Standard (non-historical) key reconstruction -- this run is built with today's
    code, so pretrain_key()/experiment_key() apply directly.
    """
    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")
    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No step-{step} checkpoint at {p}")
    return load_params(p, template_state.params), key


def report_regression(damage, overlap_b, lp, exposure, margin0, overlap_a):
    """Same nested-model structure as analyze_name_overlap.py's report_regression;
    duplicated rather than imported because that one is coupled to the six-run summary
    label widths -- the math (ols_fit/r_squared/partial_r2) is shared via import.
    """
    zo, zl, ze, zm, za = (standardize(v) for v in (overlap_b, lp, exposure, margin0, overlap_a))
    specs = [
        ("overlap_B only", [zo]),
        ("+ length", [zo, zl]),
        ("+ length + exposure", [zo, zl, ze]),
        ("+ length + exposure + margin0", [zo, zl, ze, zm]),
        ("+ length + exposure + margin0 + overlap_A", [zo, zl, ze, zm, za]),
    ]
    for label, cols in specs:
        X = np.column_stack(cols)
        beta, se, t = ols_fit(damage, X)
        print(f"  [{label:44s}]  overlap_B coef={beta[1]:+.4f}  se={se[1]:.4f}  "
              f"t={t[1]:+.2f}  n={len(damage)}")

    X_full = np.column_stack([zo, zl, ze, zm, za])
    beta_full, se_full, t_full = ols_fit(damage, X_full)
    print(f"  overlap_A's own coefficient in that full model: coef={beta_full[5]:+.4f}  "
          f"se={se_full[5]:.4f}  t={t_full[5]:+.2f}")

    r2_uni = r_squared(damage, zo.reshape(-1, 1))
    pr2_full = partial_r2(damage, np.column_stack([zl, ze, zm, za]), X_full)
    print(f"  effect size: R^2(overlap_B alone)={r2_uni:.4f}   "
          f"partial R^2(overlap_B | length,exposure,margin0,overlap_A)={pr2_full:.4f}")
    decile_report(damage, overlap_b)

    # Curvature test: reconciling the aggregate result (A's population-mean damage is
    # LOWER under the graded construction despite a much stronger within-population
    # slope) requires damage to be concave in overlap_score -- diminishing marginal harm
    # per additional colliding key. The four-point shared_count ladder doesn't obviously
    # show this by eye; a continuous quadratic fit, fully powered at n=2000, is the direct
    # test rather than reading curvature off four noisy bin means.
    zo2 = zo ** 2
    beta_q_bare, se_q_bare, t_q_bare = ols_fit(damage, np.column_stack([zo, zo2]))
    X_quad_full = np.column_stack([zo, zo2, zl, ze, zm, za])
    beta_q_full, se_q_full, t_q_full = ols_fit(damage, X_quad_full)
    shape_bare = "concave" if beta_q_bare[2] < 0 else "convex/flat"
    shape_full = "concave" if beta_q_full[2] < 0 else "convex/flat"
    print(f"  curvature (overlap_B + overlap_B^2 only): linear coef={beta_q_bare[1]:+.4f} "
          f"(t={t_q_bare[1]:+.2f})  quadratic coef={beta_q_bare[2]:+.4f} "
          f"(t={t_q_bare[2]:+.2f})  [{shape_bare}]")
    print(f"  curvature (+ full controls):               linear coef={beta_q_full[1]:+.4f} "
          f"(t={t_q_full[1]:+.2f})  quadratic coef={beta_q_full[2]:+.4f} "
          f"(t={t_q_full[2]:+.2f})  [{shape_full}]")

    return beta_full[1], se_full[1], t_full[1]


def main():
    cfg = parse_config(KIConfig, description="Graded key-overlap dose-response and timing")
    if not cfg.inject.graded_key_overlap:
        raise ValueError(
            "This script analyzes the graded-overlap construction specifically; "
            "pass --inject.graded_key_overlap true (it must match the injection run's "
            "own flag or the reconstructed population, and hence the checkpoint key, "
            "won't match what's on disk).")

    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(6)]
    # B's OWN value-half is condition-dependent (B_HALF), unlike A's (always X) -- needed
    # for candidate 2 (B's own-half margin, closing the "B's internal crowding absorbs the
    # gradient" account for the aggregate result).
    b_half = B_HALF[cfg.inject.condition]
    b_own_ids = [pop.attr_first_token_ids[k][halves[k][0] if b_half == "X" else halves[k][1]]
                 for k in range(6)]

    ids_a, ids_b = data_a.person_ids, data_b.person_ids
    names_a, names_b = pop.person_names[ids_a], pop.person_names[ids_b]

    mult = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_b[:, j], return_counts=True)
        mult_map = dict(zip(vals.tolist(), counts.tolist()))
        mult[:, j] = [mult_map.get(v, 0) for v in names_a[:, j]]
    overlap_score = mult.sum(axis=1)
    shared = mult > 0
    shared_count = shared.sum(axis=1)

    mult_a = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_a[:, j], return_counts=True)
        count_map = dict(zip(vals.tolist(), counts.tolist()))
        mult_a[:, j] = [count_map.get(v, 0) - 1 for v in names_a[:, j]]
    overlap_with_a = mult_a.sum(axis=1)

    lp = np.array([
        sum(int(pop.name_lengths[PARTS[j]][names_a[i, j]]) for j in range(3))
        for i in range(len(ids_a))
    ])

    print("=== Static covariates (condition/seed-independent given graded_key_seed) ===")
    print(f"A={len(ids_a)}  B={len(ids_b)}")
    print(f"  overlap_score: mean={overlap_score.mean():.2f} std={overlap_score.std():.2f} "
          f"min={overlap_score.min():.0f} max={overlap_score.max():.0f}")
    counts_hist = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"  PRIMARY dose-response predictor -- shared_count distribution across A: "
          f"{counts_hist}  (chance-level population was {{0: 1, 1: 34, 2: 461, "
          f"3: 1504}}; this is the engineered replacement)")
    print(f"  overlap_with_A (intrinsic-crowding control): mean={overlap_with_a.mean():.2f} "
          f"std={overlap_with_a.std():.2f}  "
          f"corr(overlap_score, overlap_with_A)="
          f"{np.corrcoef(overlap_score, overlap_with_a)[0, 1]:+.3f}")
    print(f"  L_p: mean={lp.mean():.2f} std={lp.std():.2f}  "
          f"corr(overlap_score, L_p)={np.corrcoef(overlap_score, lp)[0, 1]:+.3f}")
    print("  SIGN CONVENTION: damage = mean_rank_own (rank 1 = best/least damage, higher "
          "= worse). A POSITIVE overlap coefficient below means higher overlap predicts "
          "WORSE (higher) rank -- i.e. MORE damage.")

    pretrained_params, _ = load_inject_checkpoint(cfg, data_cfg, model, 0)
    margin0 = per_person_margin_own(model, pretrained_params, data_a, x_ids).mean(axis=1)
    print(f"  margin0 (pretrained own-half gap, mean over 6 attrs): "
          f"mean={margin0.mean():.2f} std={margin0.std():.2f}")

    pre_key = pretrain_key(cfg, data_cfg)
    exposure_path = os.path.join(cfg.exposure_dir, f"pretrain-{pre_key}.npz")
    if not os.path.exists(exposure_path):
        raise FileNotFoundError(
            f"No pretraining exposure counts at {exposure_path} -- needed as a regression "
            f"control; rerun --phase pretrain if it is missing.")
    exposure_a = np.load(exposure_path)["exposure_counts"][ids_a]
    print(f"  pretraining exposure (A): mean={exposure_a.mean():.1f} std={exposure_a.std():.1f}")

    emergence = []
    for step in STEPS:
        params, key = load_inject_checkpoint(cfg, data_cfg, model, step)
        print(f"\n=== step {step}  condition={cfg.inject.condition}  "
              f"inject_seed={cfg.inject.seed}  key={key} ===")
        rank_own = per_person_rank_own(model, params, data_a, x_ids)
        top1 = (rank_own == 1).astype(np.float64)
        damage_rank = rank_own.mean(axis=1)
        damage_top1 = top1.mean(axis=1)

        print("-- PRIMARY: dose-response by shared_count (0-3 shared name parts) --")
        for c in range(4):
            report_group(f"shared_count={c}", shared_count == c, damage_rank, damage_top1)

        print("-- continuous regression: damage ~ overlap_score (+ controls) --")
        coef, se, t = report_regression(
            damage_rank, overlap_score, lp, exposure_a, margin0, overlap_with_a)
        emergence.append((step, coef, se, t))

        print("-- secondary: by part (shares vs. does not) --")
        for j, part in enumerate(PARTS):
            report_group(f"{part} shares >=1", shared[:, j], damage_rank, damage_top1)
            report_group(f"{part} shares 0", ~shared[:, j], damage_rank, damage_top1)

        if step in (400, 1200):
            print("-- candidate 1: does damage cluster by shared name-part token? --")
            for j, part in enumerate(PARTS):
                token_level_correlation(damage_rank, names_a[:, j], names_b[:, j], part)
                token_fixed_effects_test(damage_rank, overlap_score, names_a[:, j], part)

        if step >= 400:
            b_margin = per_person_margin_own(model, params, data_b, b_own_ids).mean()
            print(f"-- candidate 2: B's own-half margin (mean over 6 attrs) = "
                  f"{b_margin:.3f} --")

    print(f"\n=== Emergence curve: overlap_B coefficient (full model) by step "
          f"(condition={cfg.inject.condition} inject_seed={cfg.inject.seed}, GRADED) ===")
    for step, coef, se, t in emergence:
        print(f"  step {step:5d}:  coef={coef:+.4f}  se={se:.4f}  t={t:+.2f}")


if __name__ == "__main__":
    main()
