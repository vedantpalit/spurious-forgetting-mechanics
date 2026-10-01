"""Post-hoc analysis for the celebrities experiment (celebrity_injection.py): does the
untouched remainder of A show a margin response graded by name-part overlap with the
celebrity subset? See celebrity_injection.py's module docstring for the full
pre-registration (split, baseline checks, sign-agnostic prediction, falsification
criterion, why confidence intervals matter here).

Two populations analyzed at every dense checkpoint:
  - remainder (n=1900): the population under test.
  - ballast (n=2000): the specificity control -- never fine-tuned, but shares SOME chance-
    level overlap with the celebrity subset via the same name-token pools. If remainder
    shows a graded response and ballast does not, the effect is flowing through the
    specific tokens celebrities hold, not something diffuse.

overlap_with_A (the intrinsic-crowding control, identical formula, computed against
ALL of A -- celebrities and remainder together, self-excluded) is included because a
randomly-chosen celebrity subset means an individual's chance of sharing with it is
proportional to how common their name parts are in A generally; this separates "shares with
celebrities specifically" from "has common name parts in general."

Reports 95% CIs (mean +/- 1.96*SE, SE = std/sqrt(n)) alongside every group mean, not just
point estimates -- group 3 (n=119) has SE ~ 0.052 on the baseline population std, so
detectable between-group gaps are on the order of 0.1, the same magnitude as effects this
project has trusted before (0.05-0.15) but not enormous.
"""
import os
from dataclasses import replace

import numpy as np

from src.config import parse_config
from src.experiments.analyze_name_overlap import (
    crossover_pretrain_key, load_inject_checkpoint, ols_fit, per_person_margin_own,
    per_person_rank_own, standardize,
)
from src.experiments.celebrity_injection import CELEB_SEED, N_CELEB, select_celebrities
from src.experiments.ckpt import ckpt_path, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, experiment_key, get_partition, init_state, load_meta,
    make_optimizer,
)

STEPS = (10, 25, 50, 100, 200, 400, 600, 800, 1000, 1200)
PARTS = ("first", "middle", "last")


def overlap_with_pool(names_target, names_pool):
    """mult_j formula: for each target row, how many rows in the pool share each
    name part (raw count -- self-exclusion, when the target is itself part of the pool, is
    the caller's responsibility; see the `- 3` in overlap_with_a below).
    """
    n = len(names_target)
    mult = np.zeros((n, 3))
    for j in range(3):
        vals, counts = np.unique(names_pool[:, j], return_counts=True)
        count_map = dict(zip(vals.tolist(), counts.tolist()))
        raw = np.array([count_map.get(v, 0) for v in names_target[:, j]])
        mult[:, j] = raw
    return mult


def load_celeb_checkpoint(cfg, data_cfg, model, step):
    pre_key = crossover_pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")
    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="celeb",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        seed=cfg.seed, celeb_seed=CELEB_SEED, n_celeb=N_CELEB)
    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    p = ckpt_path(cfg.checkpoint_dir, "celeb", f"{key}-step{step:04d}")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No step-{step} checkpoint at {p}")
    return load_params(p, template_state.params), key


def ci95(values):
    n = len(values)
    mean, sd = values.mean(), values.std()
    se = sd / np.sqrt(n)
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def report_groups(label, shared_count, values):
    print(f"-- {label} --")
    for c in range(4):
        v = values[shared_count == c]
        if len(v) == 0:
            print(f"  shared_count={c}  n=0")
            continue
        mean, se, lo, hi = ci95(v)
        print(f"  shared_count={c}  n={len(v):4d}  mean={mean:+.4f}  SE={se:.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
    has_both = (shared_count == 3).any() and (shared_count == 0).any()
    gap = values[shared_count == 3].mean() - values[shared_count == 0].mean() if has_both else float("nan")
    print(f"  gap (3-0) = {gap:+.4f}  (baseline gap was -0.029; needs to clear ~0.1 to be convincing)")


def main():
    cfg = parse_config(KIConfig, description="Celebrities experiment: post-hoc margin/rank analysis")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(6)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(6)]

    ids_a = data_a.person_ids
    ids_celeb, ids_remainder = select_celebrities(pop, ids_a)
    print(f"celebrities n={len(ids_celeb)}, remainder n={len(ids_remainder)} (seed={CELEB_SEED})")

    names_a = pop.person_names[ids_a]
    names_celeb = pop.person_names[ids_celeb]
    names_remainder = pop.person_names[ids_remainder]
    names_ballast = pop.person_names[data_ballast.person_ids]
    names_c = pop.person_names[data_c.person_ids]

    def shared_count_vs_celeb(names_target):
        mult = overlap_with_pool(names_target, names_celeb)
        return (mult > 0).sum(axis=1)

    sc_remainder = shared_count_vs_celeb(names_remainder)
    sc_ballast = shared_count_vs_celeb(names_ballast)
    sc_c = shared_count_vs_celeb(names_c)
    group_n_remainder = {c: int((sc_remainder == c).sum()) for c in range(4)}
    group_n_ballast = {c: int((sc_ballast == c).sum()) for c in range(4)}
    group_n_c = {c: int((sc_c == c).sum()) for c in range(4)}
    print(f"remainder shared_count dist: {group_n_remainder}")
    print(f"ballast shared_count dist (chance-level, specificity control): {group_n_ballast}")
    print(f"C shared_count dist (chance-level, NULL check -- never trained by either phase): {group_n_c}")

    # overlap_with_A: against ALL of A (celeb+remainder), self-excluded for remainder (its
    # own name is a row in names_a); not self-excluded for ballast/C (they aren't rows in
    # names_a at all, so no self-match to subtract). The same formula throughout.
    overlap_with_a_remainder = overlap_with_pool(names_remainder, names_a).sum(axis=1) - 3
    overlap_with_a_ballast = overlap_with_pool(names_ballast, names_a).sum(axis=1)

    def name_len(names_target):
        return np.array([
            sum(int(pop.name_lengths[PARTS[j]][names_target[i, j]]) for j in range(3))
            for i in range(len(names_target))
        ])

    lp_remainder = name_len(names_remainder)
    lp_ballast = name_len(names_ballast)

    pre_key = crossover_pretrain_key(cfg, data_cfg)
    exposure_path = os.path.join(cfg.exposure_dir, f"pretrain-{pre_key}.npz")
    if not os.path.exists(exposure_path):
        raise FileNotFoundError(f"No pretraining exposure counts at {exposure_path}.")
    exposure_counts = np.load(exposure_path)["exposure_counts"]
    exposure_remainder = exposure_counts[ids_remainder]
    exposure_ballast = exposure_counts[data_ballast.person_ids]
    exposure_c = exposure_counts[data_c.person_ids]
    print(f"exposure_c: mean={exposure_c.mean():.2f} std={exposure_c.std():.2f} "
          f"(C is never trained by either phase -- expect ~0, confirming C never entered a "
          f"pretraining batch; exposure is dropped from C's regression if so, not silently kept)")

    print("\n=== step 0 (pretrained baseline -- same checkpoint analyze_celebrity_baseline.py used) ===")
    pretrained_params, _ = load_inject_checkpoint(cfg, data_cfg, model, 0)
    margin0 = per_person_margin_own(model, pretrained_params, data_a, x_ids).mean(axis=1)
    margin0_remainder = margin0[np.searchsorted(ids_a, ids_remainder)]
    margin0_ballast = per_person_margin_own(model, pretrained_params, data_ballast, y_ids).mean(axis=1)
    margin0_c = per_person_margin_own(model, pretrained_params, data_c, x_ids).mean(axis=1)
    # BASELINE EQUALITY CHECK (on this run's actual split and checkpoint, not just the
    # earlier pre-check): confirms group means are equal before any celebrity training.
    report_groups("BASELINE remainder margin_own (step 0, pre-training)", sc_remainder, margin0_remainder)
    report_groups("BASELINE ballast margin_own (step 0, pre-training)", sc_ballast, margin0_ballast)
    report_groups("BASELINE C margin_own (step 0, pre-training) -- C was never taught its "
                   "facts, so this may be near-noise, not a clean baseline like the other two",
                   sc_c, margin0_c)

    for step in STEPS:
        params, key = load_celeb_checkpoint(cfg, data_cfg, model, step)
        print(f"\n=== step {step}  key={key} ===")

        margin_remainder = per_person_margin_own(model, params, data_a, x_ids).mean(axis=1)
        margin_remainder = margin_remainder[np.searchsorted(ids_a, ids_remainder)]
        margin_change = margin_remainder - margin0_remainder

        rank_remainder = per_person_rank_own(model, params, data_a, x_ids).mean(axis=1)
        rank_remainder = rank_remainder[np.searchsorted(ids_a, ids_remainder)]
        top1_remainder = (rank_remainder == 1).mean()
        print(f"  remainder rank_own_top1={top1_remainder:.4f} (SUPPRESSION-DIP CHECK: prediction is "
              f"no dip; flag prominently if this drops meaningfully below 1.0)")

        # ballast's own half is Y, not X -- must rank against its own candidates, same as
        # remainder does against X (this was the bug: reused x_ids for both populations).
        margin_ballast = per_person_margin_own(model, params, data_ballast, y_ids).mean(axis=1)
        margin_change_ballast = margin_ballast - margin0_ballast

        margin_c = per_person_margin_own(model, params, data_c, x_ids).mean(axis=1)
        margin_change_c = margin_c - margin0_c

        report_groups(f"remainder margin_own, step {step}", sc_remainder, margin_remainder)
        report_groups(f"remainder margin CHANGE from step 0, step {step}", sc_remainder, margin_change)
        report_groups(f"ballast margin_own (specificity control), step {step}", sc_ballast, margin_ballast)
        report_groups(f"ballast margin CHANGE from step 0, step {step}", sc_ballast, margin_change_ballast)
        report_groups(f"C margin CHANGE from step 0, step {step} (NULL CHECK)", sc_c, margin_change_c)

        # TRAJECTORY line (check: suppression-like dip-then-recover, or erosion-like
        # monotonic decline with no turn?) -- group 0 vs group 3 raw margin_own, every step,
        # one line, so the shape is readable without re-deriving it from the full report.
        m0r = margin_remainder[sc_remainder == 0].mean()
        m3r = margin_remainder[sc_remainder == 3].mean()
        m0b = margin_ballast[sc_ballast == 0].mean()
        m3b = margin_ballast[sc_ballast == 3].mean()
        print(f"  TRAJECTORY step={step:5d}  remainder sc0={m0r:+.3f} sc3={m3r:+.3f} gap={m3r-m0r:+.3f}  |  "
              f"ballast sc0={m0b:+.3f} sc3={m3b:+.3f} gap={m3b-m0b:+.3f}")

        if step >= 100:
            zsc, zoa, zlp, zex = (standardize(v.astype(float)) for v in
                                   (sc_remainder, overlap_with_a_remainder, lp_remainder, exposure_remainder))
            X = np.column_stack([zsc, zoa, zlp, zex])
            beta, se, t = ols_fit(margin_change, X)
            print(f"  regression (remainder): margin_change ~ shared_count + overlap_with_A + length + exposure")
            print(f"    shared_count coef={beta[1]:+.4f} se={se[1]:.4f} t={t[1]:+.2f}")
            print(f"    overlap_with_A coef={beta[2]:+.4f} se={se[2]:.4f} t={t[2]:+.2f}")

            # Ballast now gets the SAME covariate set as remainder (overlap_with_A, length,
            # exposure all built above) -- not the bare version from before the bug fix.
            zsc_b, zoa_b, zlp_b, zex_b = (standardize(v.astype(float)) for v in
                                           (sc_ballast, overlap_with_a_ballast, lp_ballast, exposure_ballast))
            Xb = np.column_stack([zsc_b, zoa_b, zlp_b, zex_b])
            beta_b, se_b, t_b = ols_fit(margin_change_ballast, Xb)
            print(f"  regression (ballast): margin_change ~ shared_count + overlap_with_A + length + exposure")
            print(f"    shared_count coef={beta_b[1]:+.4f} se={se_b[1]:.4f} t={t_b[1]:+.2f}")
            print(f"    overlap_with_A coef={beta_b[2]:+.4f} se={se_b[2]:.4f} t={t_b[2]:+.2f}")

            # C: bare shared_count regression only. No exposure covariate (C was never
            # pretrained, exposure_c is ~constant and would make standardize() divide by
            # ~0) -- and no length/overlap_with_A either, to keep this a clean, minimal
            # null check rather than a matched regression C's own baseline state can't
            # really support (see the BASELINE C note above).
            zsc_c = standardize(sc_c.astype(float))
            beta_c, se_c, t_c = ols_fit(margin_change_c, zsc_c.reshape(-1, 1))
            print(f"  regression (C, bare, NULL CHECK): margin_change ~ shared_count")
            print(f"    shared_count coef={beta_c[1]:+.4f} se={se_c[1]:.4f} t={t_c[1]:+.2f}")


if __name__ == "__main__":
    main()
