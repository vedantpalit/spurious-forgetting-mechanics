"""Type-vs-half-vs-individual decomposition of A's margin loss.

`analyze_head_decomposition.py` established that A's own competitive margin drops
substantially too (85-90% loss by step 50, against an always-same-half competitor) even
though the coarse cross-half suppression mechanism should, if anything, favour A (it shares
B's half). That drop is currently unexplained.

The hypothesis under test: the fast phase is not "half-Y targeting" so much as the model
re-fitting the conditional marginal over values AT VALUE POSITIONS generally -- Zucchet's
phase-1 marginal-learning, recurring because B's arrival changes what a value position is
expected to contain. Ballast suffers catastrophically because the new marginal excludes
its half entirely; A suffers mildly because it is on the right side of the marginal, but
its individual-level discrimination (which specific value, among plausible ones) is still
drowned out while the marginal itself is being relearned.

Three nested levels of "is the argmax right", at the SAME value-prediction position used
throughout this project's diagnostics:
  type       -- argmax is ANY value token for this attribute (either half). Tests "does the
                model know a value from this attribute is expected here at all" (as opposed
                to a template/structural token). Expected near-ceiling from pretraining --
                a baseline, not where the interesting movement should be.
  half       -- argmax is a value token from the population's OWN half specifically. Tests
                "does the model know a value from the right half is expected" -- this is
                the level the coarse suppression/preference mechanism (established in
                analyze_head_decomposition.py) operates at.
  individual -- argmax equals the exact correct token (the existing first_token_accuracy).

CONFOUND: `individual | half` (accuracy conditional
on argmax already landing in-half) looked like it showed ballast retaining individual-level
knowledge BETTER than A late in injection (0.778 vs 0.544 at step 1200). That comparison is
badly confounded -- A's conditional is computed on ~99% of rows (basically unconditional),
ballast's on the ~30-40% of rows where its own UNRESTRICTED argmax happened to land in-half,
which is not a random subset: it plausibly selects for whichever individuals are most firmly
stored. Two things fix this:
  own_top1   -- RESTRICT the candidate set to the population's own half FIRST, then take
                argmax. Computed on every row, nothing discarded -- this is what
                `rank_own_top1` already measures elsewhere in this project, just broken out
                per attribute here rather than pooled. This is the metric that actually
                answers "how good is individual-level discrimination", not indiv|half.
  matched comparison / selection check -- see `matched_comparison` and `selection_check`
                below. Bins both populations by pretraining exposure (the obvious fragility
                proxy, already logged) and checks directly whether ballast's in-half subset
                is exposure-biased relative to its out-of-half rows, rather than asserting it.

Run with the SAME flags as the injection / the other analyze_*.py scripts -- used to reconstruct
the checkpoint filenames and the population/value-half assignment. CPU-only: plain forward
passes (no capture_intermediates needed here, just argmax over the returned logits), cheaper
than analyze_head_decomposition.py.
"""
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.data.biography_corpus import ATTRIBUTE_NAMES
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    NUM_ATTRIBUTES, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key,
    first_token_positions, get_partition, init_state, load_meta, make_optimizer,
    meta_path, parse_steps, pretrain_key, rank_and_loss_within, token_half_groups,
)
from src.train import eval_forward

LEVEL_KEYS = ("type", "half", "individual", "own_top1", "own_loss", "argmax")


def level_accuracies(logits: np.ndarray, mask: np.ndarray, targets: np.ndarray,
                     k: int, own_ids: np.ndarray, full_ids: np.ndarray) -> dict:
    """type/half/individual/own_top1/own_loss/argmax for one attribute, one batch. Returns
    per-row arrays, not means -- aggregated by the caller across the whole eval set.
    own_top1 is the unconditional, no-rows-discarded restricted-argmax measure: candidates
    are limited to `own_ids` BEFORE taking argmax, unlike `half` (unrestricted argmax that
    happens to land in own_ids) or `individual | half` (conditional on `half`).
    """
    cols, correct = first_token_positions(mask, targets, k)
    rows = np.arange(len(correct))
    row_logits = logits[rows, cols, :]
    argmax = row_logits.argmax(axis=1)

    type_correct = np.isin(argmax, full_ids)
    half_correct = np.isin(argmax, own_ids)
    individual_correct = argmax == correct
    own_rank, own_loss = rank_and_loss_within(row_logits, correct, own_ids)
    return {
        "type": type_correct, "half": half_correct, "individual": individual_correct,
        "own_top1": own_rank == 1, "own_loss": own_loss, "argmax": argmax,
    }


def population_levels(model, params, ds, own_half: str, x_ids, y_ids, batch_size=256):
    """Per-attribute arrays for LEVEL_KEYS, one checkpoint, concatenated across the whole
    eval set, plus the person id for each row (same order, so index i in every per_attr[k]
    array and in the returned person_ids array refer to the same person).
    """
    n = ds.eval_inputs.shape[0]
    per_attr = {k: {name: [] for name in LEVEL_KEYS} for k in range(NUM_ATTRIBUTES)}
    person_id_chunks = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        person_id_chunks.append(ds.person_ids[sl])
        for k in range(NUM_ATTRIBUTES):
            own_ids = x_ids[k] if own_half == "X" else y_ids[k]
            full_ids = np.concatenate([x_ids[k], y_ids[k]])
            levels = level_accuracies(logits, mask, targets, k, own_ids, full_ids)
            for name, arr in levels.items():
                per_attr[k][name].append(arr)
    out = {k: {name: np.concatenate(v) for name, v in d.items()} for k, d in per_attr.items()}
    return out, np.concatenate(person_id_chunks)


def rates(per_attr_k: dict) -> dict:
    type_c, half_c, indiv_c = per_attr_k["type"], per_attr_k["half"], per_attr_k["individual"]
    own_top1 = per_attr_k["own_top1"]
    out = {
        "type_acc": float(type_c.mean()),
        "half_acc": float(half_c.mean()),
        "individual_acc": float(indiv_c.mean()),
        "own_top1_acc": float(own_top1.mean()),
    }
    out["half_given_type"] = float(half_c[type_c].mean()) if type_c.any() else float("nan")
    out["individual_given_half"] = float(indiv_c[half_c].mean()) if half_c.any() else float("nan")
    out["half_coverage"] = out["half_acc"]  # explicit alias: this IS indiv|half's denominator
    return out


def pooled(per_attr: dict) -> dict:
    return {name: np.concatenate([per_attr[k][name] for k in range(NUM_ATTRIBUTES)])
           for name in LEVEL_KEYS}


def fmt_row(label, step, own_half, per_attr):
    r = rates(pooled(per_attr))
    print(f"  {label:8s} step={step:4d} (own={own_half})  "
          f"type={r['type_acc']:.3f}  half={r['half_acc']:.3f}  indiv={r['individual_acc']:.3f}  "
          f"own_top1={r['own_top1_acc']:.3f}  "
          f"| half|type={r['half_given_type']:.3f}  "
          f"indiv|half={r['individual_given_half']:.3f} (coverage={r['half_coverage']:.3f})")


def fmt_per_attr(label, step, per_attr):
    for k in range(NUM_ATTRIBUTES):
        r = rates(per_attr[k])
        print(f"    {label:8s} step={step:4d} {ATTRIBUTE_NAMES[k]:14s}  "
              f"type={r['type_acc']:.3f}  half={r['half_acc']:.3f}  indiv={r['individual_acc']:.3f}  "
              f"own_top1={r['own_top1_acc']:.3f}  "
              f"| indiv|half={r['individual_given_half']:.3f} (coverage={r['half_coverage']:.3f})")


# --- confound checks: is ballast's indiv|half a selection artifact? ---

def selection_check(label, person_ids, half_correct, exposure_counts, step0_own_loss=None):
    """Direct test: do in-half rows have systematically higher pretraining exposure (and,
    if given, lower step-0 own-half loss = more confidently stored) than out-of-half rows?
    If yes, that IS the selection bias, demonstrated rather than asserted.
    """
    if not half_correct.any() or not (~half_correct).any():
        return  # nothing to split -- e.g. step 0 (all in-half) or a full-collapse step
    exp = exposure_counts[person_ids]
    in_h, out_h = half_correct, ~half_correct
    line = (f"    {label} selection check: in-half (n={int(in_h.sum())}) mean pretrain "
            f"exposure={exp[in_h].mean():.1f}  vs out-of-half (n={int(out_h.sum())}) "
            f"mean exposure={exp[out_h].mean():.1f}")
    if step0_own_loss is not None:
        line += (f"  |  step-0 own-half loss: in-half={step0_own_loss[in_h].mean():.4f}  "
                 f"out-of-half={step0_own_loss[out_h].mean():.4f} (lower=more confidently stored)")
    print(line)


def matched_comparison(label_a, person_ids_a, individual_a, label_b, person_ids_b,
                       select_b, individual_b, exposure_counts, edges):
    """Bin BOTH populations' rows by their person's pretraining exposure (shared bin edges
    across A and B), then compare individual-accuracy within each bin -- B restricted to
    `select_b` (e.g. half_correct) rows, A given every row. If B's apparent edge over A
    shrinks or reverses once both are restricted to the same exposure band, it was
    selection; if it survives in every bin, it is not.
    """
    if not select_b.any():
        return
    exp_a = exposure_counts[person_ids_a]
    exp_b = exposure_counts[person_ids_b][select_b]
    ind_b = individual_b[select_b]
    print(f"    exposure-matched: {label_a} (all rows) vs {label_b} (in-half-only rows)")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bin_a = (exp_a >= lo) & (exp_a < hi)
        bin_b = (exp_b >= lo) & (exp_b < hi)
        a_acc = individual_a[bin_a].mean() if bin_a.any() else float("nan")
        b_acc = ind_b[bin_b].mean() if bin_b.any() else float("nan")
        print(f"      exposure [{lo:.0f},{hi:.0f})  {label_a}={a_acc:.3f} (n={int(bin_a.sum())})  "
              f"{label_b}={b_acc:.3f} (n={int(bin_b.sum())})")


# --- birthdate / type-level check: pool-size confound, not "type never moves" in general ---

def wrong_argmax_category(argmax: np.ndarray, type_correct: np.ndarray,
                          other_attr_ids: np.ndarray, nonvalue_ids: np.ndarray) -> dict:
    """For type-INCORRECT rows only: is the wrong argmax another attribute's value token
    (dense-pool / cross-attribute confusion, arguably still "knows a value is wanted") or a
    non-value token (a genuine type-level failure)? Distinguishes "this attribute's pool is
    just bigger" from "the model doesn't know this position wants a value" for that attribute.
    """
    wrong = ~type_correct
    if not wrong.any():
        return {"n": 0, "other_attr": float("nan"), "non_value": float("nan")}
    wrong_argmax = argmax[wrong]
    return {
        "n": int(wrong.sum()),
        "other_attr": float(np.isin(wrong_argmax, other_attr_ids).mean()),
        "non_value": float(np.isin(wrong_argmax, nonvalue_ids).mean()),
    }


def main():
    cfg = parse_config(KIConfig, description="Type-vs-half-vs-individual decomposition")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    _, _, _, nonvalue_ids = token_half_groups(pop, halves)
    nonvalue_ids = np.array(sorted(nonvalue_ids))
    # For each attribute k, every OTHER attribute's full pool (X union Y) -- used to check
    # whether a type-level miss is cross-attribute confusion rather than a real type failure.
    full_pool = [np.concatenate([x_ids[k], y_ids[k]]) for k in range(NUM_ATTRIBUTES)]
    other_attr_ids = [np.concatenate([full_pool[kk] for kk in range(NUM_ATTRIBUTES) if kk != k])
                      for k in range(NUM_ATTRIBUTES)]
    print("Attribute pool sizes (post-filter, first-token-unique values):")
    for k in range(NUM_ATTRIBUTES):
        print(f"    {ATTRIBUTE_NAMES[k]:14s} {int(pop.num_values_per_attr[k]):4d} values")

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")

    exposure_path = os.path.join(cfg.exposure_dir, f"pretrain-{pre_key}.npz")
    if not os.path.exists(exposure_path):
        raise FileNotFoundError(
            f"No pretraining exposure counts at {exposure_path} -- needed for the "
            f"selection/matched-comparison checks. phase_pretrain writes this via "
            f"save_exposures; rerun --phase pretrain if it is missing.")
    exposure_counts = np.load(exposure_path)["exposure_counts"]
    ids_a, ids_ballast = np.arange(cfg.num_a), np.arange(cfg.num_a, cfg.num_a + cfg.num_ballast)
    exposure_edges = np.quantile(
        exposure_counts[np.concatenate([ids_a, ids_ballast])], [0.0, 0.25, 0.5, 0.75, 1.0])
    exposure_edges[0], exposure_edges[-1] = -np.inf, np.inf

    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    print(f"Injection key: {key}  (peak_lr={inject_opt.peak_lr:g})")

    steps = sorted(parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps))
    populations = [("dataA", data_a, "X"), ("ballast", data_ballast, "Y")]

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    print(f"\n{'=' * 116}\nPooled type/half/individual/own_top1 accuracy (mean over all 6 attributes)\n"
          f"type = argmax is any value token for this attribute -- ceiling baseline. "
          f"half = unrestricted argmax lands in the population's OWN half. "
          f"individual = argmax is the exact correct token. own_top1 = restrict candidates "
          f"to the own half FIRST, then argmax -- unconditional, no rows discarded (this is "
          f"rank_own_top1, per attribute). indiv|half is CONDITIONAL on half and its "
          f"coverage is printed alongside -- do not compare indiv|half across populations "
          f"without checking coverage matches.\n{'=' * 116}")

    results_by_step = {}
    step0_own_loss_pooled = {}  # {population_label: pooled own_loss array at step 0}
    step0_person_ids = {}
    for step in steps:
        p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
        if not os.path.exists(p):
            print(f"  [missing: {p} -- skipping step {step}]")
            continue
        params = load_params(p, template_state.params)
        step_results = {}
        for label, ds, own_half in populations:
            per_attr, person_ids = population_levels(model, params, ds, own_half, x_ids, y_ids)
            fmt_row(label, step, own_half, per_attr)
            step_results[label] = (per_attr, person_ids)
            if step == 0:
                step0_own_loss_pooled[label] = pooled(per_attr)["own_loss"]
                step0_person_ids[label] = np.tile(person_ids, NUM_ATTRIBUTES)
        results_by_step[step] = step_results
        print()

    print(f"\n{'=' * 116}\nConfound checks on indiv|half (ballast), pooled across attributes\n{'=' * 116}")
    a_per_attr, a_person_ids = results_by_step[max(steps)]["dataA"]
    a_pooled = pooled(a_per_attr)
    a_person_ids_pooled = np.tile(a_person_ids, NUM_ATTRIBUTES)
    for step, step_results in results_by_step.items():
        if "ballast" not in step_results:
            continue
        b_per_attr, b_person_ids = step_results["ballast"]
        b_pooled = pooled(b_per_attr)
        b_person_ids_pooled = np.tile(b_person_ids, NUM_ATTRIBUTES)
        half_c = b_pooled["half"]
        if not half_c.any() or not (~half_c).any():
            continue
        print(f"  step={step}")
        selection_check("ballast", b_person_ids_pooled, half_c, exposure_counts,
                        step0_own_loss=step0_own_loss_pooled.get("ballast"))
        # Compare against A's SAME-step pooled individual accuracy, matched on exposure.
        a_step_pooled = pooled(results_by_step[step]["dataA"][0])
        matched_comparison("dataA", a_person_ids_pooled, a_step_pooled["individual"],
                           "ballast", b_person_ids_pooled, half_c, b_pooled["individual"],
                           exposure_counts, exposure_edges)
        print()

    print(f"\n{'=' * 116}\nType-level misses: what does the wrong argmax actually predict? "
          f"(birthdate has the largest pool, 221 vs 127-135 for the rest -- checking whether "
          f"its lower type accuracy is pool-size / cross-attribute confusion rather than a "
          f"genuine type-level failure)\n{'=' * 116}")
    for step in (steps[-1],):
        if step not in results_by_step:
            continue
        for label in ("dataA", "ballast"):
            per_attr, _ = results_by_step[step][label]
            for k in range(NUM_ATTRIBUTES):
                cat = wrong_argmax_category(per_attr[k]["argmax"], per_attr[k]["type"],
                                            other_attr_ids[k], nonvalue_ids)
                if cat["n"] == 0:
                    continue
                print(f"    {label:8s} step={step:4d} {ATTRIBUTE_NAMES[k]:14s} "
                      f"n_wrong={cat['n']:4d}  other_attribute_value={cat['other_attr']:.3f}  "
                      f"non_value_token={cat['non_value']:.3f}")

    print(f"\n{'=' * 116}\nPer-attribute breakdown\n{'=' * 116}")
    for step, step_results in results_by_step.items():
        for label, (per_attr, _) in step_results.items():
            fmt_per_attr(label, step, per_attr)
        print()


if __name__ == "__main__":
    main()
