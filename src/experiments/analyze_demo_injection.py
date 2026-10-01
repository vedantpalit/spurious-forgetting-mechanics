"""Per-person logging pass for the first-look demonstration figure.

Reloads each dense checkpoint demo_injection.py saved for one (arm, inject_seed) run and
computes, for every individual in A:
  - rank_own: own-half-restricted rank (per_person_rank_own, reused from
    analyze_name_overlap.py -- unchanged).
  - first_correct: RAW full-vocabulary first-token correctness (argmax over the entire
    vocab, no candidate restriction). This does NOT exist as a per-person function
    anywhere else in the codebase -- rank_own is deliberately restricted to own-half
    candidates precisely so it EXCLUDES cross-half suppression, which makes it the wrong
    tool for a figure whose whole point is to show that suppression. first_correct is the
    per-person counterpart of the population-level dataA/first_token_accuracy the training
    run already logs; its per-checkpoint mean should match that logged value as an internal
    consistency check (printed below, not just assumed).

Static per-person covariates (overlap_score vs. B, overlap_with_A vs. A itself, both from
analyze_name_overlap.py's formula) are computed once from the same population construction
demo_injection.py used for this arm. They are NOT interchangeable across arms: population
size (num_ballast: 0 vs. >0) changes RNG consumption in BiographyPopulation's name
assignment (see pretrain_key's own docstring -- the same fact already established for value
assignment applies here too), so A's actual names, and therefore
overlap_score/overlap_with_A, can differ between the two arms even though A's ids are
always [0, num_a). They are seed-invariant WITHIN one arm (inject.seed only varies B's
batch-sampling stream, per InjectConfig.seed's own docstring), so recomputing them fresh in
every (arm, seed) run here is correct, not redundant across arms, and cheap either way.

Saves one compact .npz per (arm, inject_seed): steps, rank_own (num_steps, N, 6),
first_correct (num_steps, N, 6), overlap_score (N,), overlap_with_A (N,), person_ids (N,).
Also prints a bottom-decile-at-final-checkpoint report: which
individuals sit in the bottom decile of mean rank_own at the last saved step, and how their
overlap_score/overlap_with_A compare to the rest of A.

Run (must match the demo_injection.py invocation exactly -- same population/partition
flags -- so the reconstructed checkpoint key resolves to real files):
  uv run python -m src.experiments.analyze_demo_injection \
    --num_a 2000 --num_ballast 2000 --num_b 500 --num_c 500 \
    --inject.condition disjoint --inject.seed 0 \
    --inject.total_steps 1200 --inject.checkpoint_steps 0,10,25,50,100,200,400,600,800,1000,1200 \
    --partition_path data/biography/value_partition.npz

Output directory is a module constant (OUT_DIR), not a CLI flag: parse_config's parser
calls parse_args() strictly and rejects any flag that isn't a KIConfig field.
"""
import os

import jax.numpy as jnp
import numpy as np

from dataclasses import replace

from src.config import parse_config
from src.experiments.analyze_name_overlap import per_person_rank_own
from src.experiments.ckpt import ckpt_path, load_params
from src.experiments.demo_injection import demo_inject_key
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, first_token_positions, get_partition, init_state,
    load_meta, make_optimizer, meta_path, parse_steps, pretrain_key,
)
from src.experiments.analyze_name_overlap import crossover_pretrain_key
from src.train import eval_forward

OUT_DIR = "demo_figure_data"


def per_person_first_correct(model, params, ds, batch_size=256):
    """Per-person, per-attribute RAW first-token correctness: argmax over the FULL
    vocabulary, no candidate restriction. The unrestricted counterpart to
    per_person_rank_own -- see module docstring for why rank_own is the wrong metric for
    this figure. (N, 6) array; row i is person i (ds is A's uncapped eval set)."""
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
            pred = row_logits.argmax(axis=1)
            out[sl, k] = (pred == correct).astype(np.float64)
    return out


def decile_report(rank_own_final, overlap_score, overlap_with_a):
    n = len(rank_own_final)
    order = np.argsort(rank_own_final, kind="stable")[::-1]  # worst (highest rank) first
    bottom = order[: n // 10]
    print(f"\n=== Bottom decile of mean rank_own at the final checkpoint (n={len(bottom)}) ===")
    print(f"  mean_rank_own: bottom decile={rank_own_final[bottom].mean():.3f}  "
          f"rest={rank_own_final[np.setdiff1d(np.arange(n), bottom)].mean():.3f}")
    print(f"  overlap_score (vs B): bottom decile mean={overlap_score[bottom].mean():.2f}  "
          f"rest mean={np.delete(overlap_score, bottom).mean():.2f}")
    print(f"  overlap_with_A: bottom decile mean={overlap_with_a[bottom].mean():.2f}  "
          f"rest mean={np.delete(overlap_with_a, bottom).mean():.2f}")
    print(f"  bottom-decile person_ids: {bottom.tolist()}")


def main():
    cfg = parse_config(KIConfig, description="Per-person logging for the demo figure")

    if cfg.inject.condition != "disjoint":
        raise ValueError(f"Expected --inject.condition disjoint, got {cfg.inject.condition!r}.")
    arm = "without_ballast" if cfg.num_ballast == 0 else "with_ballast"
    print(f"Arm: {arm} (inferred from num_ballast={cfg.num_ballast})")

    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)

    pre_key = (crossover_pretrain_key(cfg, data_cfg) if arm == "with_ballast"
               else pretrain_key(cfg, data_cfg))

    # Must replicate demo_injection.py's peak_lr derivation exactly before hashing: the
    # driver bakes the DERIVED peak_lr into inject_cfg before computing its key (unless
    # --inject.opt.peak_lr was set explicitly), so hashing cfg.inject as-is here (still
    # carrying the -1.0 "derive it" sentinel) computes a different, wrong key. Confirmed
    # directly -- this was the actual cause of a real key mismatch during local testing,
    # not a hypothetical.
    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
        meta = load_meta(pre_path) if pre_path and os.path.exists(meta_path(pre_path)) else None
        if meta is None:
            raise FileNotFoundError(
                f"No metadata at {meta_path(pre_path)}; set --inject.opt.peak_lr explicitly "
                f"to match whatever demo_injection.py derived for this run.")
        derived = float(meta["peak_lr"]) / INJECT_LR_RATIO
        inject_opt = replace(inject_opt, peak_lr=derived)
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    key = demo_inject_key(cfg, data_cfg, pre_key, inject_cfg)
    print(f"Reconstructed key={key}")

    tx = make_optimizer(inject_cfg.opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    # --- static covariates (overlap_score vs B, overlap_with_A -- see module docstring) ---
    ids_a, ids_b = data_a.person_ids, data_b.person_ids
    names_a, names_b = pop.person_names[ids_a], pop.person_names[ids_b]
    mult_b = np.zeros((len(ids_a), 3), dtype=np.float64)
    mult_a = np.zeros((len(ids_a), 3), dtype=np.float64)
    for j in range(3):
        vals_b, counts_b = np.unique(names_b[:, j], return_counts=True)
        map_b = dict(zip(vals_b.tolist(), counts_b.tolist()))
        mult_b[:, j] = [map_b.get(v, 0) for v in names_a[:, j]]
        vals_a, counts_a = np.unique(names_a[:, j], return_counts=True)
        map_a = dict(zip(vals_a.tolist(), counts_a.tolist()))
        mult_a[:, j] = [map_a.get(v, 0) - 1 for v in names_a[:, j]]  # exclude self
    overlap_score = mult_b.sum(axis=1)
    overlap_with_a = mult_a.sum(axis=1)
    print(f"overlap_score: mean={overlap_score.mean():.2f} std={overlap_score.std():.2f}")
    print(f"overlap_with_A: mean={overlap_with_a.mean():.2f} std={overlap_with_a.std():.2f}")

    steps = sorted(parse_steps(cfg.inject.checkpoint_steps, cfg.inject.total_steps))
    n = len(ids_a)
    rank_own_all = np.zeros((len(steps), n, 6), dtype=np.float64)
    first_correct_all = np.zeros((len(steps), n, 6), dtype=np.float64)

    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(6)]

    for i, step in enumerate(steps):
        p = os.path.join(cfg.checkpoint_dir, f"demoinject-{arm}-{key}-step{step:04d}.msgpack")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No step-{step} checkpoint at {p}")
        params = load_params(p, template_state.params)
        rank_own_all[i] = per_person_rank_own(model, params, data_a, x_ids)
        first_correct_all[i] = per_person_first_correct(model, params, data_a)
        pop_first_acc = first_correct_all[i].mean()
        pop_rank_top1 = (rank_own_all[i] == 1).mean()
        print(f"step {step:5d}: mean first_correct (raw, all attrs)={pop_first_acc:.4f}  "
              f"mean rank_own_top1={pop_rank_top1:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"demo-{arm}-seed{cfg.inject.seed}-{key}.npz")
    np.savez(out_path, steps=np.array(steps), rank_own=rank_own_all,
              first_correct=first_correct_all, overlap_score=overlap_score,
              overlap_with_a=overlap_with_a, person_ids=ids_a)
    print(f"\nSaved {out_path}")

    decile_report(rank_own_all[-1].mean(axis=1), overlap_score, overlap_with_a)


if __name__ == "__main__":
    main()
