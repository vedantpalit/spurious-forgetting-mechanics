"""Does per-individual damage within A track name-part (key) proximity to B, at
chance-level sharing, before any deliberately-graded overlap experiment is built?

The first version of this script used "shares a part with ANY B individual" as the
predictor and found it nearly universal at these pool sizes (200/200/213 vs. |B|=500):
1504 of 2000 A individuals share all 3 parts with some B individual, only 35 sit at
count 0/1 -- not enough spread to read a dose-response from. That predicate is also the
wrong one theoretically: the crowding account (Bietti/Cabannes, gradient descent on
associative memories accumulating outer products of keys and values) is about how many
new keys sit close to a given individual's key, not whether at least one does.

So the primary predictor here is continuous: for each A individual i and part
j in {first, middle, last}, mult_j(i) = the number of B individuals whose part-j token
equals i's -- how many of B's 500 new keys are one-part-identical to i's key along that
axis. `overlap_score(i) = sum_j mult_j(i)`. Under the compositional-key hypothesis this
project already carries (keys as roughly-orthogonal sums of three name-part vectors),
this is the natural linear proxy for "how many new keys are close to this one": a match
with 10 different B individuals on one part contributes as much as a match with 1 B
individual on all three, which is the right accounting if closeness is compositional and
additive over parts rather than requiring a full-name match.

Two confounds are handled by regression, not by stratifying (stratifying loses power and
was only used in the first version of this script):
  - Name length L_p: which parts an individual has determines total name length, which
    with RoPE affects the representation at every downstream position.
  - Pretraining exposure: a well-established fact may resist damage independent of its
    neighbours; per-person counts are already logged (`save_exposures`).
  - Step-0 (pretrained) own-half margin: individuals whose facts were more weakly stored
    at injection start may lose more regardless of overlap.
All three are regressed in jointly with overlap_score (standardized coefficients), and
the overlap coefficient is reported with and without each control so adjustment survival
is visible rather than assumed.

overlap_score, L_p, pretraining exposure, and margin0 are all fixed by cfg.seed=42 and
the shared pretrained checkpoint -- identical across every (condition, inject.seed) run
analyzed here (BiographyPopulation._assign_names runs before _assign_values, and all six
crossover runs share one pretrain). Only per-step damage varies by run.

Damage is rank_own (own-half-restricted), not raw accuracy or rank_full: key overlap is
a candidate mechanism for erosion (individual-level), not the coarse half-level
suppression, which is value-region-based and has nothing to do with name parts.

A fourth covariate closes the intrinsic-crowding confound: overlap_score partly measures
how common an individual's name parts are in the pools generally -- common parts mean
more B individuals share them, but also more A individuals do, so a name-part-crowding
effect intrinsic to A (nothing to do with B at all) could masquerade as overlap_score's
effect. overlap_with_A is the identical multiplicity formula computed against A's own
population instead of B's (self excluded). If overlap_score's coefficient survives adding
it, the effect is B-induced; if not, it's intrinsic crowding -- a different, still
interesting, claim. Effect size (R^2, partial R^2, and the top-vs-bottom-decile damage
difference) is reported alongside significance, since t-stats alone are easy to reach at
n=2000 and don't say how large the effect is.

A is evaluated uncapped during injection (inject.max_eval_people=0), so
data_a.eval_inputs row i is person i's biography, in person-id order -- no id lookup
needed to align rows back to people.
"""
import os
from dataclasses import asdict, replace

import jax.numpy as jnp
import numpy as np
from scipy import stats as scipy_stats

from src.config import parse_config
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key,
    first_token_positions, get_partition, init_state, load_meta, make_optimizer,
    rank_and_loss_within,
)
from src.train import eval_forward

PARTS = ("first", "middle", "last")
# Every dense checkpoint the crossover saved (10 and 25 included): the graded-overlap
# construction tests whether deliberately larger overlap makes the effect appear earlier
# than this chance-level baseline's step 100 -- the strongest version of that comparison
# is point-for-point at matched steps, which needs this baseline computed at every step
# the graded run's early window will also use (10, 25 here; the graded run additionally
# has 20/30/40/60/80, never saved for this run and not retroactively available).
STEPS = (10, 25, 50, 100, 200, 400, 600, 800, 1000, 1200)


# --- per-person forward-pass metrics ---

def per_person_rank_own(model, params, ds, x_ids, batch_size=256):
    """Per-person, per-attribute rank within A's own half (X). (N, 6) array; row i is
    person i, since `ds` is the uncapped A eval set (see module docstring).
    """
    n = ds.eval_inputs.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        for k in range(6):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[np.arange(sl.stop - sl.start), cols, :]
            rank, _ = rank_and_loss_within(row_logits, correct, x_ids[k])
            out[sl, k] = rank
    return out


def per_person_margin_own(model, params, ds, x_ids, batch_size=256):
    """Per-person, per-attribute margin (correct logit minus the best OTHER own-half
    candidate's logit -- gap_own, per person rather than pooled). (N, 6) array.
    """
    n = ds.eval_inputs.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        for k in range(6):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[np.arange(sl.stop - sl.start), cols, :]
            cand_ids = x_ids[k]
            cand_logits = row_logits[:, cand_ids]
            correct_logit = row_logits[np.arange(len(correct)), correct]
            is_correct_col = cand_ids[None, :] == correct[:, None]
            best_other = np.where(is_correct_col, -np.inf, cand_logits).max(axis=1)
            out[sl, k] = correct_logit - best_other
    return out


# --- checkpoint loading ---

def crossover_pretrain_key(cfg, data_cfg):
    """The high_overlap/disjoint crossover ran before the checkpoint-key fix that
    added a partition-file content hash to pretrain_key() (ckpt.py/knowledge_injection.py,
    landed to close a real collision bug). Its checkpoints on disk are keyed under the OLD
    (pre-fix) formula; calling today's pretrain_key() gives a different, real-but-wrong key
    (confirmed: it resolves to a checkpoint from the later exclusion-fraction sweep, not
    this run) and every load below would silently 404. This reproduces the crossover's own
    historical formula -- verified locally to rederive its known real key exactly
    (high_overlap seed0 -> inject key 49d1b419b26d, matching the run's own printed log)
    before this was wired in.
    """
    pretrain_for_key = replace(cfg.pretrain, max_eval_people=0, eval_interval=0, target_acc=0.0)
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="pretrain",
        pretrain=pretrain_for_key, seed=cfg.seed,
        population=[cfg.num_a, cfg.num_ballast, cfg.num_b, cfg.num_c],
        c_half=cfg.c_half, partition_seed=cfg.partition_seed)


def historical_inject_cfg_dict(inject_cfg):
    """The crossover ran before InjectConfig had graded_key_overlap/graded_key_seed
    (added for the graded key-overlap construction).
    experiment_key hashes every field dataclasses.asdict() produces, so the live
    InjectConfig now silently computes a different, real-but-wrong key for the crossover's
    own checkpoints (confirmed: d68c1e521ca6 instead of the real 49d1b419b26d) -- this is
    the checkpoint-key system correctly detecting a real config change, not a bug, but it
    means historical checkpoints need the historical shape reconstructed explicitly, the
    same principle as crossover_pretrain_key. Verified locally to rederive the known real
    key exactly before this was wired in.
    """
    d = asdict(inject_cfg)
    d.pop("graded_key_overlap", None)
    d.pop("graded_key_seed", None)
    return d


def load_inject_checkpoint(cfg, data_cfg, model, step):
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
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=historical_inject_cfg_dict(replace(inject_cfg, max_eval_people=0)),
        pretrain_key=pre_key, seed=cfg.seed)
    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No step-{step} checkpoint at {p}")
    return load_params(p, template_state.params), key


# --- regression ---

def ols_fit(y, X):
    """OLS with intercept, classical (homoskedastic) standard errors. X is (n, p)
    without an intercept column. Returns (beta, se, tstat), each length p+1, intercept
    first.
    """
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = Xd.shape[0] - Xd.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    return beta, se, beta / se


def standardize(x):
    return (x - x.mean()) / x.std()


def r_squared(y, X):
    """R^2 of an OLS fit (with intercept). X is (n, p) without an intercept column."""
    beta, _, _ = ols_fit(y, X)
    n = X.shape[0]
    resid = y - np.column_stack([np.ones(n), X]) @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst


def partial_r2(y, X_reduced, X_full):
    """Fraction of X_reduced's residual variance explained by X_full's extra column(s).
    X_full must contain X_reduced's columns plus the target predictor(s).
    """
    n = X_reduced.shape[0]
    beta_r, _, _ = ols_fit(y, X_reduced)
    resid_r = y - np.column_stack([np.ones(n), X_reduced]) @ beta_r
    sse_r = float(resid_r @ resid_r)
    beta_f, _, _ = ols_fit(y, X_full)
    resid_f = y - np.column_stack([np.ones(n), X_full]) @ beta_f
    sse_f = float(resid_f @ resid_f)
    return (sse_r - sse_f) / sse_r


def decile_report(damage, overlap):
    """Mean damage in the lowest vs. highest overlap_score decile, and their difference --
    the effect-size number a reader wants alongside significance.

    Position-based split (rank order, ties broken by stable sort on original index), not
    value-threshold quantiles. With many people tied at the same low integer overlap_score
    (real for the graded population, where the 10th percentile itself can land exactly
    on a heavily-tied value), threshold binning via np.digitize pushes every tied value
    into the bin above the boundary, silently emptying decile 1 (confirmed: this shipped as
    a NaN in the first graded-run report). Splitting by rank position instead guarantees
    non-empty, ~equal-sized bins regardless of ties.
    """
    n = len(overlap)
    order = np.argsort(overlap, kind="stable")
    bin_idx = np.empty(n, dtype=int)
    bin_idx[order] = (np.arange(n) * 10) // n
    lo, hi = damage[bin_idx == 0], damage[bin_idx == 9]
    lo_ov, hi_ov = overlap[bin_idx == 0], overlap[bin_idx == 9]
    print(f"  decile 1  (lowest overlap,  n={len(lo):4d}, overlap in "
          f"[{lo_ov.min():.0f},{lo_ov.max():.0f}]):  mean_rank_own={lo.mean():.3f}")
    print(f"  decile 10 (highest overlap, n={len(hi):4d}, overlap in "
          f"[{hi_ov.min():.0f},{hi_ov.max():.0f}]):  mean_rank_own={hi.mean():.3f}")
    print(f"  decile10 - decile1 = {hi.mean() - lo.mean():+.3f}")


# --- candidate 1: does damage cluster by shared name-part token, not just overlap_score? ---
#
# The curvature test (report_regression) only rules out models where damage is a function
# of an individual's own overlap_score in isolation. If A individuals sharing a specific
# name-part token also share the parameters encoding it, harm to that token is harm to
# everyone holding it -- and total damage would scale with how many DISTINCT tokens B
# disturbs, not with per-individual overlap. The graded construction shrank B's token
# footprint sharply (fewer distinct tokens used) while leaving mean overlap_score almost
# unchanged (7.35 -> 7.23) -- exactly the pattern this predicts, and it needs no
# curvature at all.

def token_level_correlation(damage, tokens_a, tokens_b, part_name):
    """Token-level (not individual-level) view: does a token's B-usage predict mean
    damage among the A individuals who hold it? Unit of observation is the token, not
    the person -- complements the fixed-effects test below, which asks a related but
    distinct question (does knowing WHICH token you hold explain damage beyond what
    overlap_score already captures, regardless of that token's specific usage count).
    """
    b_vals, b_counts = np.unique(tokens_b, return_counts=True)
    b_usage_map = dict(zip(b_vals.tolist(), b_counts.tolist()))
    unique_a = np.unique(tokens_a)
    usage, mean_damage = [], []
    for tok in unique_a:
        mask = tokens_a == tok
        if mask.sum() < 2:
            continue
        usage.append(b_usage_map.get(int(tok), 0))
        mean_damage.append(float(damage[mask].mean()))
    usage = np.array(usage, dtype=float)
    mean_damage = np.array(mean_damage)
    corr = float(np.corrcoef(usage, mean_damage)[0, 1]) if len(usage) > 2 else float("nan")
    print(f"  candidate-1 token-level corr [{part_name:6s}]: {len(usage):4d} tokens  "
          f"corr(B usage of token, mean damage of its A-holders)={corr:+.3f}")
    return corr


def token_fixed_effects_test(damage, overlap_score, tokens, part_name):
    """Partial R^2 and F-test: does an individual's SPECIFIC token for one name part
    explain damage beyond what the continuous overlap_score already captures? Token
    dummies strictly nest overlap_score (far more free parameters -- ~100-200 dummies vs.
    1), so a raw R^2 comparison is biased toward finding "more" purely from parameter
    count; the F-test is what actually separates signal from that. Restricted model:
    damage ~ overlap_score. Full model: damage ~ overlap_score + (k-1) token dummies
    (one token held out as reference to avoid collinearity with the intercept).
    """
    z = standardize(overlap_score)
    n = len(damage)
    Xd0 = np.column_stack([np.ones(n), z])
    beta0, _, _, _ = np.linalg.lstsq(Xd0, damage, rcond=None)
    sse0 = float(((damage - Xd0 @ beta0) ** 2).sum())

    unique_tokens = np.unique(tokens)
    dummies = np.column_stack([(tokens == tok).astype(float) for tok in unique_tokens[1:]])
    k = dummies.shape[1]
    Xd1 = np.column_stack([np.ones(n), z, dummies])
    beta1, _, _, _ = np.linalg.lstsq(Xd1, damage, rcond=None)
    sse1 = float(((damage - Xd1 @ beta1) ** 2).sum())

    dof1, dof2 = k, n - 2 - k
    f_stat = ((sse0 - sse1) / dof1) / (sse1 / dof2)
    p_value = float(scipy_stats.f.sf(f_stat, dof1, dof2))
    partial_r2 = (sse0 - sse1) / sse0
    print(f"  candidate-1 token-FE test [{part_name:6s}]: {k + 1:4d} tokens  "
          f"partial R^2 (beyond overlap_score)={partial_r2:.4f}  "
          f"F({dof1},{dof2})={f_stat:.2f}  p={p_value:.2e}")
    return partial_r2, f_stat, p_value


def report_regression(damage, overlap_b, lp, exposure, margin0, overlap_a):
    """`overlap_b` is overlap_score (vs. B, the theoretically-motivated predictor).
    `overlap_a` is the intrinsic-crowding control: the identical formula computed against
    A's own population (self excluded) -- if overlap_b's coefficient doesn't survive
    controlling for it, the effect is intrinsic key crowding within A, not B-induced.
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
    beta_a_only, se_a_only, t_a_only = ols_fit(damage, za.reshape(-1, 1))
    print(f"  overlap_A alone (no overlap_B, no other controls): coef={beta_a_only[1]:+.4f}  "
          f"se={se_a_only[1]:.4f}  t={t_a_only[1]:+.2f}")

    r2_uni = r_squared(damage, zo.reshape(-1, 1))
    pr2_full = partial_r2(damage, np.column_stack([zl, ze, zm, za]), X_full)
    print(f"  effect size: R^2(overlap_B alone)={r2_uni:.4f}   "
          f"partial R^2(overlap_B | length,exposure,margin0,overlap_A)={pr2_full:.4f}")
    decile_report(damage, overlap_b)

    return beta_full[1], se_full[1], t_full[1]  # overlap_B's coefficient in the full model


def report_group(label, m, damage_rank, damage_top1):
    n = int(m.sum())
    if n == 0:
        print(f"  {label:34s} n=    0")
        return
    print(f"  {label:34s} n={n:5d}  mean_rank_own={damage_rank[m].mean():7.3f}  "
          f"top1_rate={damage_top1[m].mean():.4f}")


def main():
    cfg = parse_config(KIConfig, description="Name-part overlap vs. per-person damage in A")
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

    # Continuous predictor: overlap_score(i) = sum over the 3 parts of how many B
    # individuals share that exact part token with person i (see module docstring).
    mult = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_b[:, j], return_counts=True)
        mult_map = dict(zip(vals.tolist(), counts.tolist()))
        mult[:, j] = [mult_map.get(v, 0) for v in names_a[:, j]]
    overlap_score = mult.sum(axis=1)

    # Secondary, binary, kept separable by part (not collapsed to a count -- the count
    # split is the degenerate one from the first version of this script, footnoted below).
    shared = mult > 0
    shared_count = shared.sum(axis=1)

    # Intrinsic-crowding control: identical formula, but counting OTHER A individuals
    # (self excluded) who share each part -- common name parts make an individual's key
    # crowded within A's own population regardless of B. If overlap_score (vs. B) doesn't
    # survive controlling for this, the effect could be intrinsic crowding rather than
    # B-induced.
    mult_a = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals, counts = np.unique(names_a[:, j], return_counts=True)
        count_map = dict(zip(vals.tolist(), counts.tolist()))
        mult_a[:, j] = [count_map.get(v, 0) - 1 for v in names_a[:, j]]  # exclude self
    overlap_with_a = mult_a.sum(axis=1)

    lp = np.array([
        sum(int(pop.name_lengths[PARTS[j]][names_a[i, j]]) for j in range(3))
        for i in range(len(ids_a))
    ])

    print("=== Static covariates (condition/seed-independent: fixed by cfg.seed and the "
          "shared pretrained checkpoint) ===")
    print(f"A={len(ids_a)}  B={len(ids_b)}")
    print(f"  overlap_score: mean={overlap_score.mean():.2f} std={overlap_score.std():.2f} "
          f"min={overlap_score.min():.0f} max={overlap_score.max():.0f}  "
          f"percentiles(10/25/50/75/90)={np.percentile(overlap_score, [10, 25, 50, 75, 90]).round(2).tolist()}")
    for j, part in enumerate(PARTS):
        pool_n = len(pop.name_ids[part])
        represented = len(set(names_b[:, j].tolist()))
        print(f"  {part}: pool={pool_n}  represented in B={represented} "
              f"({represented / pool_n:.1%})  P(A shares >=1)={shared[:, j].mean():.3f}  "
              f"mult mean={mult[:, j].mean():.2f} std={mult[:, j].std():.2f}")
    counts_footnote = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"  [footnote -- degenerate, see docstring] binary shared_count distribution: "
          f"{counts_footnote}")
    print(f"  L_p: mean={lp.mean():.2f} std={lp.std():.2f}  "
          f"corr(overlap_score, L_p)={np.corrcoef(overlap_score, lp)[0, 1]:+.3f}")
    print(f"  overlap_with_A (intrinsic-crowding control, self excluded): "
          f"mean={overlap_with_a.mean():.2f} std={overlap_with_a.std():.2f} "
          f"min={overlap_with_a.min():.0f} max={overlap_with_a.max():.0f}  "
          f"corr(overlap_score, overlap_with_A)="
          f"{np.corrcoef(overlap_score, overlap_with_a)[0, 1]:+.3f}")
    print("  SIGN CONVENTION: damage = mean_rank_own (rank 1 = best/least damage, higher "
          "= worse). A POSITIVE overlap coefficient below means higher name-part overlap "
          "predicts WORSE (higher) rank -- i.e. MORE damage.")

    # step 0 of injection is saved before any injection gradient step (save_opt_state is
    # False in every crossover run, so it's a bare-params file in the format load_params
    # expects) -- numerically identical to the pretrained model, and already the correct
    # checkpoint format, unlike checkpoints/pretrain-{key}.msgpack itself (a full
    # TrainState, params+opt_state+step, saved by save_state -- wrong shape for this).
    pretrained_params, _ = load_inject_checkpoint(cfg, data_cfg, model, 0)
    margin0 = per_person_margin_own(model, pretrained_params, data_a, x_ids).mean(axis=1)
    print(f"  margin0 (pretrained own-half gap, mean over 6 attrs): "
          f"mean={margin0.mean():.2f} std={margin0.std():.2f}  "
          f"corr(overlap_score, margin0)={np.corrcoef(overlap_score, margin0)[0, 1]:+.3f}")

    pre_key = crossover_pretrain_key(cfg, data_cfg)
    exposure_path = os.path.join(cfg.exposure_dir, f"pretrain-{pre_key}.npz")
    if not os.path.exists(exposure_path):
        raise FileNotFoundError(
            f"No pretraining exposure counts at {exposure_path} -- needed as a regression "
            f"control; rerun --phase pretrain if it is missing.")
    exposure_a = np.load(exposure_path)["exposure_counts"][ids_a]
    print(f"  pretraining exposure (A): mean={exposure_a.mean():.1f} std={exposure_a.std():.1f}  "
          f"corr(overlap_score, exposure)={np.corrcoef(overlap_score, exposure_a)[0, 1]:+.3f}")

    emergence = []  # (step, full-model overlap_B coef, se, t) -- the emergence curve
    for step in STEPS:
        params, key = load_inject_checkpoint(cfg, data_cfg, model, step)
        print(f"\n=== step {step}  condition={cfg.inject.condition}  "
              f"inject_seed={cfg.inject.seed}  key={key} ===")
        rank_own = per_person_rank_own(model, params, data_a, x_ids)
        top1 = (rank_own == 1).astype(np.float64)
        damage_rank = rank_own.mean(axis=1)
        damage_top1 = top1.mean(axis=1)

        print("-- primary: damage ~ overlap_score (+ controls), standardized predictors --")
        coef, se, t = report_regression(
            damage_rank, overlap_score, lp, exposure_a, margin0, overlap_with_a)
        emergence.append((step, coef, se, t))

        print("-- secondary: by part (shares vs. does not; count-independent; thin/imbalanced) --")
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
          f"(condition={cfg.inject.condition} inject_seed={cfg.inject.seed}) ===")
    for step, coef, se, t in emergence:
        print(f"  step {step:5d}:  coef={coef:+.4f}  se={se:.4f}  t={t:+.2f}")


if __name__ == "__main__":
    main()
