"""Weight-space localization: is the injection-time suppression an output-head/bias
phenomenon, or Zheng-style bottom-layer/embedding realignment?

Mirrors Zheng et al. (arXiv:2501.13453) Section 3.4 / Figure 4: per-component SVD of
weight-update matrices, principal angle between update subspaces across training phases.
Their finding: the first ~150 steps of Task 1 update weights in a space close to Task 0's
own update direction (undoing alignment), concentrated in bottom layers / input embeddings.

Two predictions, opposite in *where* the phase-1 update concentrates:
  - Zheng-style mechanism: dominated by embedding / bottom-layer components.
  - Suppression account: dominated by the OUTPUT HEAD, and specifically by
    the head columns/bias entries for half-Y candidates during phase 1 (0->10) -- the
    sharp test, since it is falsifiable in a direction Zheng's account does not predict.

Phases (dense checkpoints, params-only): 0->10, 10->50, 50->200, 200->400.

Run with the SAME flags as the injection / analyze_prior_shift.py -- used to reconstruct the
checkpoint filenames.
"""
import os

import numpy as np
from dataclasses import replace

from src.config import parse_config
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    NUM_ATTRIBUTES, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key,
    get_partition, init_state, load_meta, make_optimizer, meta_path, pretrain_key,
    token_half_groups,
)

PHASES = [(0, 10), (10, 50), (50, 200), (200, 400)]
TOP_R = 8  # truncation rank for the update-subspace basis; see principal_angles()


def flatten_params(params) -> dict:
    """{'block_0/attention/query/kernel': array, ...} from the params pytree."""
    import jax
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        out[name] = np.asarray(leaf)
    return out


def principal_angles(A: np.ndarray, B: np.ndarray, r: int = TOP_R) -> np.ndarray:
    """Principal angles (radians) between the top-r left-singular-vector subspaces of
    two update matrices A, B (Bjorck & Golub 1973). 0 = same subspace, pi/2 = orthogonal.

    A and B must have the same number of rows (same output dimension); r is clipped to
    each matrix's actual rank so this is safe for small/thin matrices.
    """
    r_eff = min(r, A.shape[0], A.shape[1], B.shape[0], B.shape[1])
    if r_eff == 0:
        return np.array([])
    Ua = np.linalg.svd(A, full_matrices=False)[0][:, :r_eff]
    Ub = np.linalg.svd(B, full_matrices=False)[0][:, :r_eff]
    sigma = np.linalg.svd(Ua.T @ Ub, full_matrices=False)[1]
    return np.arccos(np.clip(sigma, -1.0, 1.0))


def _self_test():
    """Verify the formula on cases with a known answer before trusting it on real data."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((64, 16))
    # Identical subspace -> angles should be ~0.
    angles_same = principal_angles(M, M * 3.7, r=4)
    assert np.allclose(angles_same, 0.0, atol=1e-5), f"identical-subspace angles not ~0: {angles_same}"
    # Orthogonal complement subspaces (disjoint coordinate blocks) -> angles should be ~pi/2.
    A = np.zeros((64, 4)); A[:32, :] = rng.standard_normal((32, 4))
    B = np.zeros((64, 4)); B[32:, :] = rng.standard_normal((32, 4))
    angles_orth = principal_angles(A, B, r=4)
    assert np.allclose(angles_orth, np.pi / 2, atol=1e-5), f"orthogonal-subspace angles not ~pi/2: {angles_orth}"
    print(f"  self-test OK: identical-subspace angles {np.degrees(angles_same).round(2)} deg, "
          f"orthogonal-subspace angles {np.degrees(angles_orth).round(2)} deg")


def main():
    print("Principal-angle formula self-test:")
    _self_test()

    cfg = parse_config(KIConfig, description="Weight-space localization for Stage 1 injection")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)

    # birthplace and work_location both draw from CITIES (identical post-filter pools,
    # confirmed: np.array_equal(attr_1_first_token_ids, attr_5_first_token_ids) is True)
    # but are partitioned INDEPENDENTLY, so a city token's X/Y label is only well-defined
    # per attribute -- 68 of 133 cities (51%) are X for one attribute and Y for the other.
    # The output head has one column per TOKEN, not per (token, attribute) pair, so these
    # tokens are genuinely ambiguous at the weight level: pulled toward X by one attribute's
    # exposure and toward Y by the other's non-occurrence. Excluded from the clean X/Y
    # comparison and reported as their own group -- whether they show a MUTED signature
    # (conflicting pressure cancelling out) is itself an informative check of the theory,
    # not a nuisance to average away.
    x_ids, y_ids, ambiguous_ids, _ = token_half_groups(pop, halves)
    print(f"\nToken/half labeling: {len(x_ids)} unambiguous X, {len(y_ids)} unambiguous Y, "
          f"{len(ambiguous_ids)} ambiguous (shared pool, X for one attribute and Y for "
          f"another -- birthplace/work_location share CITIES). Ambiguous tokens excluded "
          f"from the clean X-vs-Y comparison, reported separately below.")

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    print(f"\nInjection key: {key}")

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    steps_needed = sorted({s for pair in PHASES for s in pair})
    params_at = {}
    for step in steps_needed:
        p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint for step {step}: {p}")
        params_at[step] = flatten_params(load_params(p, template_state.params))

    component_names = sorted(params_at[steps_needed[0]].keys())
    deltas = {}  # (start, end) -> {component: ndarray}
    for start, end in PHASES:
        deltas[(start, end)] = {
            name: params_at[end][name] - params_at[start][name] for name in component_names
        }

    # --- magnitude: which component moves most, phase by phase ---
    # Raw Frobenius norm is confounded by matrix SIZE: head/Dense_0/kernel has ~481K
    # elements, roughly double a single MLP kernel's ~262K, so "largest Frobenius norm"
    # partly just measures "has more numbers in it." Reported alongside the size-normalized
    # RMS-per-element (Frobenius norm / sqrt(n_elements)) -- the fair comparison across
    # differently-sized components -- and ranking is by RMS, not raw norm.
    print(f"\n{'=' * 100}\nWeight-update magnitude, ranked by size-normalized RMS-per-element "
          f"(top 8), per phase\n{'=' * 100}")
    for phase in PHASES:
        stats = {name: (float(np.linalg.norm(d)), float(np.linalg.norm(d) / np.sqrt(d.size)))
                 for name, d in deltas[phase].items()}
        ranked = sorted(stats.items(), key=lambda kv: -kv[1][1])
        print(f"\nPhase {phase[0]}->{phase[1]}:")
        print(f"  {'RMS/elem':>10s}  {'Frobenius':>10s}  {'n_elements':>10s}  component")
        for name, (fro, rms) in ranked[:8]:
            print(f"  {rms:10.6f}  {fro:10.4f}  {deltas[phase][name].size:10,d}  {name}")
        head_rank = next(i for i, (name, _) in enumerate(ranked) if name == "head/Dense_0/kernel") + 1
        embed_rank = next(i for i, (name, _) in enumerate(ranked) if "Embed" in name) + 1
        print(f"  [head/Dense_0/kernel is rank #{head_rank} of {len(ranked)} by RMS/element; "
              f"embedding is rank #{embed_rank}]")

    # --- principal angles between successive phases, per 2D weight component ---
    print(f"\n{'=' * 100}\nPrincipal angle (degrees, mean over top-{TOP_R} modes) between "
          f"successive phases' updates\n{'=' * 100}")
    print(f"{'component':45s}  {'0-10 vs 10-50':>15s}  {'10-50 vs 50-200':>17s}  {'50-200 vs 200-400':>19s}")
    for name in component_names:
        arr = deltas[PHASES[0]][name]
        if arr.ndim != 2:
            continue  # principal angles need a genuine 2D subspace; skip biases/norms
        row = []
        for i in range(len(PHASES) - 1):
            a = deltas[PHASES[i]][name]
            b = deltas[PHASES[i + 1]][name]
            ang = principal_angles(a, b)
            row.append(float(np.degrees(ang.mean())) if ang.size else float("nan"))
        print(f"{name:45s}  {row[0]:15.2f}  {row[1]:17.2f}  {row[2]:19.2f}")

    # --- the sharp test: output head, phase 1, grouped by value half ---
    print(f"\n{'=' * 100}\nOutput head: per-token norm of change during phase 0->10, grouped by half\n{'=' * 100}")
    head_kernel_delta = deltas[(0, 10)]["head/Dense_0/kernel"]  # (model_dim, vocab)
    head_bias_delta = deltas[(0, 10)].get("head/Dense_0/bias")  # (vocab,)
    col_norms = np.linalg.norm(head_kernel_delta, axis=0)  # per output token
    groups = {
        "half-X value tokens": sorted(x_ids),
        "half-Y value tokens": sorted(y_ids),
        "ambiguous (X for one attr, Y for another)": sorted(ambiguous_ids),
        "non-value tokens": sorted(set(range(col_norms.shape[0])) - x_ids - y_ids - ambiguous_ids),
    }
    for label, ids in groups.items():
        n = col_norms[ids]
        print(f"  {label:44s} (n={len(ids):4d}):  kernel-column norm  mean={n.mean():.5f}  median={np.median(n):.5f}")
    if head_bias_delta is not None:
        print()
        for label, ids in groups.items():
            b = head_bias_delta[ids]
            print(f"  {label:44s}: bias change      mean={b.mean():+.5f}  median={np.median(b):+.5f}")
        print(f"\n  [suppression account predicts half-Y bias moves NEGATIVE and further than "
              f"half-X or non-value tokens during this phase. Ambiguous tokens, pulled by two "
              f"attributes at once, are the sharper test: a MUTED bias change relative to clean "
              f"Y would mean the pressure genuinely comes from per-attribute exposure, not a "
              f"single global per-token direction -- the two signals partially cancelling.]")

    # Same breakdown across all four phases, compact, to see it relax during recovery.
    print(f"\n{'=' * 100}\nHead bias mean-change by half, all four phases (relaxation check)\n{'=' * 100}")
    print(f"{'phase':12s}  {'X':>10s}  {'Y':>10s}  {'ambiguous':>10s}  {'non-value':>10s}")
    for phase in PHASES:
        d = deltas[phase].get("head/Dense_0/bias")
        if d is None:
            continue
        vals = [d[ids].mean() for ids in groups.values()]
        print(f"{phase[0]:>4d}->{phase[1]:<4d}  " + "  ".join(f"{v:+10.5f}" for v in vals))


if __name__ == "__main__":
    main()
