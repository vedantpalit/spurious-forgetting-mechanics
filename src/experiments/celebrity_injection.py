"""The celebrities experiment: fine-tune on a fixed subset of A's own already-known
individuals (no new facts, no new value assignment -- literally continue training on their
existing biographies with their existing values), then measure whether the untouched
remainder of A shows a graded margin response by name-part overlap with that subset. This is
the positive-sign counterpart to the token-clustering damage mechanism: if writing to a shared
component damages everyone holding it, reinforcing one should benefit everyone holding it. It
is also an identical-adjacent boundary condition not yet measured, and the mechanistic account
for Zucchet's own unexplained celebrities result (~50%->70% on the untouched population).

PRE-REGISTRATION (settled before this ran; do not revise after seeing results)

  Split: seed=71 (CELEB_SEED below), n=100 (N_CELEB below), fixed across all three inject
  seeds -- only the celebrity dataset's own batch-sampling stream varies by seed, not which
  100 of A's 2000 people are celebrities. Remainder shared_count distribution at this split:
  {0: 427, 1: 829, 2: 525, 3: 119}.

  Pre-checked on the existing pretrained checkpoint (analyze_celebrity_baseline.py) before
  committing to this design:
    - Accuracy/rank_own are degenerate there (rank_own_top1=1.000 exactly -> mean rank_own
      forced to exactly 1.0, no spread possible) -- margin_own is the only viable continuous
      outcome, not a choice between metrics.
    - margin_own's upper tail is open, not compressed toward a ceiling (decile 9->10 gap
      0.412, larger than every middle-decile gap) -- real room to move upward.
    - Baseline margin_own does not differ by shared_count (9.372/9.363/9.335/9.343 for
      groups 0/1/2/3, a 0.037 spread against a population std of 0.57) -- no pre-existing
      confound to control for.

  Why this design is sound despite fine-tuning itself being a real intervention: margins
  keep moving under cross-entropy after accuracy saturates, so continued training WILL move
  the remainder's margins regardless of token sharing -- that is expected, not a confound to
  design around. What makes the design interpretable is that the shared_count GRADIENT
  self-controls: a generic, token-sharing-independent training effect lifts (or drops) all
  four shared_count groups together and shows up as a flat gradient -- which is already the
  falsification criterion below, not a separate control arm requirement. "The checkpoint is
  converged" was never the reason this is sound.

  Prediction, stated sign-agnostically: higher shared_count predicts a more FAVOURABLE
  margin CHANGE, not necessarily an increase. Fine-tuning on 100 individuals is itself a
  narrowing -- a concentrated value set, which is exactly the condition found to produce
  sharp suppression. The remainder's margins may fall rather than rise; if they do, the
  mechanism predicts high-shared_count individuals fall LESS (or rise more), not that they
  rise in absolute terms. A raw first_acc / rank_own_top1 dip in the remainder is also
  tracked as a secondary prediction: celebrities reinforce values already present in A's own
  region, so nothing new competes for the argmax and no suppression-style dip is expected.
  If one appears anyway, that indicates suppression responds to concentration of what's
  being trained on even when the values are already familiar -- not what the current
  suppression account claims -- and should be flagged prominently, not folded in quietly.

  Falsification: a flat shared_count gradient (uniform margin change across all four groups)
  means the benefit/cost isn't flowing through shared components, and the mechanism doesn't
  explain Zucchet's celebrities result. "No change at all" is reported as a DISTINCT outcome
  from "uniform change" -- the first says fine-tuning on 100 people had no detectable effect
  at this scale, the second says an effect exists but isn't graded by overlap.

  Confidence intervals, not point estimates: group 3 (n=119) has SE ~ 0.570/sqrt(119) ~
  0.052 on the baseline population std -- detectable between-group gaps are on the order of
  0.1, and the baseline gap (-0.029) needs clearing by roughly that much to be convincing.
  Report intervals at every checkpoint, not just means (see analyze_celebrity_result.py).

  Analysis-side (not this script): overlap_with_A (the intrinsic-crowding regressor, same
  formula) is included as a covariate, since a randomly-chosen celebrity subset means an
  individual's chance of sharing with it is proportional to how common their name parts
  are in A generally -- "shares with celebrities" and "has common name parts" are correlated
  by construction and need separating. Ballast (never fine-tuned, its own chance-level
  overlap with the celebrity subset) is the specificity control: if the remainder shows a
  graded response and ballast does not, the effect is flowing through the specific tokens
  celebrities hold, not something diffuse.

MECHANICS

  Celebrities and remainder are both ordinary A individuals -- same pretrained checkpoint,
  same existing value assignment (pop.value_probs, untouched). The only thing that changes
  is which of A's people the injection phase's training batches are drawn from. Checkpoints
  are saved under the "celeb" kind (checkpoints/celeb-<key>-stepNNNN.msgpack), never
  "inject-", so there is no possibility of colliding with or overwriting a real crossover
  checkpoint even if the config hash happened to coincide.
"""
import os
from dataclasses import replace

import numpy as np
import wandb

from src.config import parse_config
from src.experiments.analyze_name_overlap import crossover_pretrain_key
from src.experiments.ckpt import ckpt_path, load_state, save_params, save_state
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, NUM_ATTRIBUTES, build, experiment_key, get_partition,
    init_state, load_meta, make_optimizer, meta_path, own_half_baseline_nats,
    parse_steps, population_baseline_nats, reset_stream, save_exposures, train_loop,
    vars_nested,
)
from src.data.biography import BiographyDataset

CELEB_SEED = 71
N_CELEB = 100


def select_celebrities(pop, ids_a):
    """Same split as analyze_celebrity_baseline.py -- fixed, reused verbatim, not redrawn."""
    rng = np.random.default_rng(CELEB_SEED)
    celeb_local = rng.choice(len(ids_a), size=N_CELEB, replace=False)
    is_celeb = np.zeros(len(ids_a), dtype=bool)
    is_celeb[celeb_local] = True
    remainder_local = np.where(~is_celeb)[0]
    return ids_a[celeb_local], ids_a[remainder_local]


def main():
    cfg = parse_config(KIConfig, description="Celebrities experiment: fine-tune on a subset of A")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)

    pre_key = crossover_pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")
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
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    ids_a = data_a.person_ids
    ids_celeb, ids_remainder = select_celebrities(pop, ids_a)
    print(f"Celebrities: n={len(ids_celeb)} (seed={CELEB_SEED}), remainder: n={len(ids_remainder)}")

    data_celeb = BiographyDataset(pop, ids_celeb, "celeb", seed=cfg.seed + 5 + inject_cfg.seed,
                                   max_eval_people=0)
    data_remainder = BiographyDataset(pop, ids_remainder, "remainder", seed=cfg.seed + 300,
                                       max_eval_people=0)

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        raise ValueError("Celebrities experiment always uses a fresh optimizer; "
                          "--inject.fresh_optimizer=false is not supported here.")
    print("Optimizer: fresh")

    reset_stream(data_celeb, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [data_remainder, data_ballast]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)

    # Distinct "celeb" kind, not "inject" -- see module docstring. inject.condition is not
    # semantically meaningful here (no new values are assigned) but is still part of what
    # experiment_key hashes, so it's pinned in the .sh wrapper for reproducibility, not
    # read for its usual meaning.
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="celeb",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        seed=cfg.seed, celeb_seed=CELEB_SEED, n_celeb=N_CELEB)
    print(f"Fine-tuning on celebrities, checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = ckpt_path(cfg.checkpoint_dir, "celeb", f"{key}-step{step:04d}")
        if inject_cfg.save_opt_state:
            save_state(p, st)
        else:
            save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

    baseline = population_baseline_nats(pop)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    # remainder is a subset of A -- same own-half (X) as A itself. ballast keeps its usual Y.
    own_halves = {"remainder": "X", "ballast": "Y"}
    rank_ctx = {
        "x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
        "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                         "Y": own_half_baseline_nats(halves, "Y")},
    }

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="celeb",
                    config=vars_nested(cfg)):
        state, final_loss, _ = train_loop(
            state, data_celeb, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, data_celeb, f"celeb-{key}")
    print(f"Final celebrity-fine-tuning train loss = {final_loss:.4f}")
    return state


if __name__ == "__main__":
    main()
