"""Does the OV pieces' de-coherence track B's acquisition, or the step number?

THE STATE CHANGE, ALREADY MEASURED. At the trough the eight blocks' OV contributions are
mutually aligned -- mean pairwise cosine 0.485, banded by depth -- and by the recovery peak all
28 pairs have de-aligned to 0.228, none rising, the band gone. Signal 0.258 against a
seed-to-seed null of 0.008-0.014, with each P_j reproducing across seeds at cos 0.99.

WHAT THAT DOES NOT SHOW. It is two checkpoints. It does not distinguish de-coherence being
driven by B becoming individuable from it being a function of training time.

THE TEST, AND IT IS THE SAME ONE 3.19 USED. The dose conditions move B's onset over a 3.6x range
(LR 2x: 40, batch 512: 60, batch 128: 80, LR 0.5x: 120) while leaving the magnitude of the
effect alone (peak ||c|| within 10.7%, drop 42-46%). If mean pairwise cosine collapses on B's
clock in every condition, de-coherence is caused by B becoming individuable. If it collapses at
the same STEP regardless of condition, the coherence story is a coincidence of timing.

  coherence collapses on B's clock   -> the account has causal support
  coherence collapses on the step clock -> drop the account

B'S PROGRESS IS MEASURED HERE, not read from the old logs. The same run that patches the
checkpoint also evaluates B's accuracy on it, so the two axes cannot come from different places.

ONLY THE INDIVIDUAL BLOCKS ARE PATCHED. Coherence is a statement about the eight per-block
pieces; the cumulative configs cost as much again and answer a different question. All patch
steps run in ONE job so the population build and the step-0 readout are paid once.

KEY DERIVATION. The dose runs' checkpoint keys come from `mlpfree_dose_injection`, which hashes
the inject config -- so lr_mult and batch size enter it, and the checkpoint_steps STRING enters
it too. That string is imported from the dose driver rather than retyped: a byte difference
gives a different key and a "checkpoint missing" that looks like the run never happened.

Run:
  uv run python -m src.experiments.analyze_dose_coherence --lr_mult 0.5 --seed 0
"""
import argparse
import os
import sys
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, eval_forward, first_token_positions, init_state,
    make_optimizer,
)
from src.experiments.analyze_mlpfree import PRETRAIN_PEAK_LR, PRETRAIN_TOTAL_STEPS, SCHEDULES
from src.experiments.analyze_weight_patch import GROUPS, readout, splice, verify
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key
# Imported, not retyped: this string is hashed into the checkpoint key.
from src.experiments.mlpfree_dose_injection import DENSE_SCHEDULE
from src.experiments.mlpfree_dose_injection import SCHEDULES as DOSE_SCHEDULES

OUT_DIR = "dose_coherence"
BATCH_SIZE = 32
MODEL_DIM, NUM_HEADS, NUM_LAYERS = 512, 8, 8

# The dense grid runs 20..140 by 5; these span the turnover in every condition (25 to 90) and
# reach well past B's ceiling in the slowest (220). Step 0 is excluded: patching step-0 weights
# into the step-0 checkpoint is a no-op and every P_j is identically zero.
DOSE_STEPS = "20,40,60,80,100,120,140,200,400"
BASELINE_STEPS = "10,25,50,100,200,400"


def _cfg(condition, seed, inject_batch, total_steps, dense):
    steps = DENSE_SCHEDULE if dense else DOSE_SCHEDULES[total_steps]
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", "0",
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", condition, "--inject.seed", str(seed),
        "--inject.total_steps", str(total_steps),
        "--inject.batch_size", str(inject_batch),
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", steps,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ]
    return parse_config(KIConfig, description="dose coherence")


def _baseline_cfg(condition, seed, total_steps):
    """The lr_mult=1 / batch=256 condition is the existing unfrozen baseline, on its own grid."""
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", "0",
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", condition, "--inject.seed", str(seed),
        "--inject.total_steps", str(total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10",
        "--inject.checkpoint_steps", SCHEDULES[total_steps],
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ]
    return parse_config(KIConfig, description="dose coherence (baseline)")


def accuracy(model, params, ds, n):
    """Argmax accuracy at the value first-token slots, averaged over attributes."""
    hit = tot = 0
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        logits = np.asarray(eval_forward(model.apply, params,
                                         jnp.array(ds.eval_inputs[sl])))
        targets, mask = np.asarray(ds.eval_targets[sl]), np.asarray(ds.eval_mask[sl])
        rows = np.arange(logits.shape[0])
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            hit += int((logits[rows, cols, :].argmax(-1) == correct).sum())
            tot += len(rows)
    return hit / max(tot, 1)


def mean_pairwise_cos(P):
    """P is (8, E). Mean cosine over the 28 distinct block pairs."""
    U = P / np.clip(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12, None)
    G = U @ U.T
    iu = np.triu_indices(P.shape[0], k=1)
    return float(G[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_mult", type=float, default=1.0)
    ap.add_argument("--inject_batch", type=int, default=256)
    ap.add_argument("--dense_checkpoints", action="store_true", default=None,
                    help="default: on for dose conditions, off for the lr1/bs256 baseline")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=3000)
    ap.add_argument("--patch_steps", default=None,
                    help=f"default: {DOSE_STEPS} for dose conditions, {BASELINE_STEPS} for "
                         f"the baseline")
    ap.add_argument("--n_items", type=int, default=512)
    ap.add_argument("--patch", default="ov", choices=sorted(GROUPS))
    a = ap.parse_args()

    is_baseline = (a.lr_mult == 1.0 and a.inject_batch == 256)
    dense = (not is_baseline) if a.dense_checkpoints is None else a.dense_checkpoints
    total = 1200 if is_baseline else a.total_steps
    patch_steps = [int(s) for s in (a.patch_steps or
                                    (BASELINE_STEPS if is_baseline else DOSE_STEPS)).split(",")]

    cfg = (_baseline_cfg(a.condition, a.seed, total) if is_baseline
           else _cfg(a.condition, a.seed, a.inject_batch, total, dense))
    pop, _, data_cfg, _, data_a, _bal, data_b, _c = build(cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)

    base_lr = PRETRAIN_PEAK_LR / INJECT_LR_RATIO
    inject_cfg = replace(cfg.inject, opt=replace(cfg.inject.opt, peak_lr=base_lr * a.lr_mult))
    pre_key = mlpfree_pretrain_key(cfg, data_cfg)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="mlpfreeinject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=a.pretrain_step, seed=cfg.seed, arm="mlp_free")
    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_cfg.opt, inject_cfg.total_steps),
                          cfg.seed + 20)

    tag = ("baseline" if is_baseline else f"lr{a.lr_mult}-bs{a.inject_batch}")
    print(f"condition={tag}  lr={inject_cfg.opt.peak_lr:.3e}  batch={inject_cfg.batch_size}  "
          f"total_steps={total}  dense={dense}  seed={a.seed}")
    print(f"  key={key}")

    def path_for(s):
        return os.path.join(a.checkpoint_dir,
                            f"mlpfreeinject-p{a.pretrain_step}-{key}-step{s:04d}.msgpack")

    missing = [s for s in [0] + patch_steps if not os.path.exists(path_for(s))]
    if missing:
        raise SystemExit(f"checkpoints missing for steps {missing}\n  e.g. {path_for(missing[0])}")

    n = min(a.n_items, data_a.eval_inputs.shape[0])
    nb = data_b.eval_inputs.shape[0]
    m, t = np.asarray(data_a.eval_mask)[:n], np.asarray(data_a.eval_targets)[:n]
    cols = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int64)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(m, t, k)

    p0 = load_params(path_for(0), template.params)
    r0 = readout(model, p0, data_a, cols, n)          # paid once, reused at every patch step
    blocks = [(f"blk{j}", {j}) for j in range(NUM_LAYERS)]

    rows, P_all, bacc, base = [], [], [], []
    for si, s in enumerate(patch_steps):
        pt = load_params(path_for(s), template.params)
        if si == 0:
            verify(model, p0, pt, a.patch, NUM_LAYERS, data_a, cols, min(64, n),
                   readout(model, p0, data_a, cols, min(64, n)))
        b = accuracy(model, pt, data_b, nb)
        d_none = readout(model, pt, data_a, cols, n) - r0
        P = np.stack([d_none - (readout(model, splice(pt, p0, a.patch, bl)[0],
                                        data_a, cols, n) - r0)
                      for _, bl in blocks])                       # (8, A, E)
        mpc = np.array([mean_pairwise_cos(P[:, k]) for k in range(NUM_ATTRIBUTES)])
        bn = np.linalg.norm(d_none, axis=-1)
        P_all.append(P.astype(np.float32)); bacc.append(b); base.append(bn)
        rows.append((s, b, bn.mean(), float(mpc.mean()), mpc))
        print(f"  step {s:>4}  B acc {b:.3f}  ||delta|| {bn.mean():7.3f}  "
              f"mean pairwise cos {mpc.mean():.4f}  | per attr "
              + " ".join(f"{v:.3f}" for v in mpc))

    print(f"\n{tag} seed {a.seed}")
    print(f"  {'step':>6} {'B acc':>7} {'||delta||':>10} {'pairwise cos':>13}")
    for s, b, bn, mc, _ in rows:
        print(f"  {s:>6} {b:>7.3f} {bn:>10.3f} {mc:>13.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"mlp_free-{tag}-p{a.pretrain_step}-{a.condition}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(patch_steps), b_acc=np.array(bacc),
        pairwise_cos=np.stack([r[4] for r in rows]),
        base_norm=np.stack(base), P=np.stack(P_all),
        lr_mult=a.lr_mult, inject_batch=a.inject_batch, patch=a.patch,
        n_items=n, seed=a.seed, pretrain_step=a.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
