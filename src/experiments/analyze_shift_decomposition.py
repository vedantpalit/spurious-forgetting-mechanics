"""Is the transformer's crash carried by an individual-INDEPENDENT logit shift?

The toy established that its entire suppression effect is a constant-across-individuals
shift living in the readout bias `b`: deleting `delta_b` restores A to its exact
pre-injection baseline at every step, while a store update of Frobenius norm 36 reaches A
not at all. This asks whether the transformer's crash is the same object.

For each dense injection checkpoint t, A's logits at each attribute's value slot are
decomposed as

    dz_a(t) = z_a(t) - z_a(0) = c(t) + v_a(t),    c_j(t) = mean_a dz_a[j](t)

with `c` the individual-independent component and `v_a` the residual. Four readouts are
then computed at every checkpoint:

    trained        z_a(t)                  -- what actually happened
    constant-only  z_a(0) + c(t)           -- the crash the constant component alone causes
    varying-only   z_a(0) + v_a(t)         -- the crash the individual-varying part causes
    nobias         z_a(t) with the readout bias delta zeroed

THE COUNTERFACTUALS ARE THE READOUT; THE ENERGY SPLIT IS CONTEXT. A component can carry
most of an update's energy and none of its effect -- which is exactly what the row-space
norms did in the toy, where the shared subspace held 34% of ||dW|| and moved nothing.
Variance fractions are reported, but the reading is the counterfactual accuracy and rank.

`nobias` is an INTERVENTION, not an attribution: the readout bias is reset to its
pre-injection value and everything else left trained, the exact analog of the toy
counterfactual that closed its question. The bias-share-of-c number is the weaker
attribution statement, and is labelled as such.

WHAT THE ANSWER DECIDES.
  * Constant-carried -> the toy models the SHAPE of the transformer's suppression, and the
    remaining question is only why the transformer's version reverses. This holds whether
    the shift is built from the readout bias or from W acting along a mean key direction:
    both are individual-independent, and both fail to reverse for the same gradient reason
    (the push decays toward zero as B is learned; nothing in B's loss asks it back up).
    Origin matters for the RECOVERY story, not for the modelling question -- a bias-carried
    shift has no reason to reverse, while a W-carried one competes with B's individual
    learning for the same parameters, which is precisely what would explain why the
    transformer recovers and the toy does not.
  * Varying-carried -> the toy's suppression is a degenerate special case, and stage 2
    would be aimed at the wrong thing.

TIMING IS THE POINT. Readouts are reported at every dense checkpoint through the recovery
window, not only at trough and endpoint. If ||c|| SHRINKS between trough and peak, recovery
is the constant component being bought back, and the toy's failure is fully explained by `b`
having nothing to buy it back with. If ||c|| stays flat while accuracy recovers, recovery
comes from the varying component growing to compensate -- a different mechanism, pointing
somewhere else entirely.

Run (one invocation per arm/pretrain_step/seed):
  uv run python -m src.experiments.analyze_shift_decomposition \
      --arm mlp_free --pretrain_step 16000 --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, eval_forward, first_token_positions, get_partition, init_state,
    make_optimizer, pretrain_key, rank_and_loss_within,
)
from src.experiments.analyze_mlpfree import (
    ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for,
)
from src.experiments.mlpfree_dose_injection import DENSE_SCHEDULE, FIG_SCHEDULE
from src.experiments.mlpfree_freeze_injection import ARMS as FREEZE_ARMS
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "shift_decomposition"
# (b, L, V) at L=195, V=2141 is ~214MB per batch at 128; 256 doubles it for no gain, since
# this script also holds three (N, 6, V) blocks at ~103MB each.
BATCH_SIZE = 128


def find_readout_bias_path(params, vocab_size):
    """Locate the readout bias by SHAPE, not by a hardcoded path.

    The head is ClassificationHead -> nn.Dense(vocab_size), and flax defaults
    use_bias=True, so exactly one leaf should be a (vocab_size,) bias. Asserting
    uniqueness rather than assuming a name means a rename in the model factory fails
    loudly here instead of silently zeroing the wrong parameter.
    """
    flat = flatten_dict(params)
    hits = [k for k, v in flat.items()
            if k[-1] == "bias" and getattr(v, "shape", None) == (vocab_size,)]
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one (vocab_size={vocab_size},) 'bias' leaf in the params, "
            f"found {len(hits)}: {hits}")
    return hits[0]


def with_bias_reset(params, bias_path, bias_value):
    flat = flatten_dict(params)
    flat[bias_path] = bias_value
    return unflatten_dict(flat)


def row_logits_all(model, params, ds, vocab_size, want_targets=False):
    """(N, 6, V) value-slot row logits for every individual and attribute.

    At V=2141 and N=2000 this is ~103MB in float32, so it is materialized rather than
    streamed -- which keeps the decomposition arithmetic plain numpy instead of a two-pass
    accumulation whose correctness is harder to check.
    """
    n = ds.eval_inputs.shape[0]
    out = np.zeros((n, NUM_ATTRIBUTES, vocab_size), dtype=np.float32)
    correct_ids = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int32) if want_targets else None
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        rows = np.arange(sl.stop - sl.start)
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            out[sl, k, :] = logits[rows, cols, :]
            if want_targets:
                correct_ids[sl, k] = correct
    return (out, correct_ids) if want_targets else out


def nll_from(z, correct_ids, x_ids):
    """Per-individual NLL on the correct value: full-vocabulary and own-half.

    The console eval lines never printed a loss (only accuracy, rank and the hallucination
    gap) and per-step losses go to wandb, so this is the only route to a loss curve for the
    freeze arms from saved checkpoints. -log p on the correct value IS the cross-entropy for
    that fact, so it is a loss rather than a proxy for one.
    """
    n = z.shape[0]
    full = np.zeros((n, NUM_ATTRIBUTES)); own = np.zeros((n, NUM_ATTRIBUTES))
    for k in range(NUM_ATTRIBUTES):
        rows = z[:, k, :].astype(np.float64)
        c = rows[np.arange(n), correct_ids[:, k]]
        m = rows.max(axis=1)
        full[:, k] = -(c - (m + np.log(np.exp(rows - m[:, None]).sum(axis=1))))
        cand = rows[:, x_ids[k]]
        mo = cand.max(axis=1)
        own[:, k] = -(c - (mo + np.log(np.exp(cand - mo[:, None]).sum(axis=1))))
    return full, own


def metrics_from(z, correct_ids, x_ids):
    """first_acc (full-vocab argmax) and own-half rank, from an (N, 6, V) block."""
    n = z.shape[0]
    acc = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float64)
    rank = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float64)
    for k in range(NUM_ATTRIBUTES):
        rows = z[:, k, :]
        acc[:, k] = (rows.argmax(axis=1) == correct_ids[:, k]).astype(np.float64)
        r, _ = rank_and_loss_within(rows, correct_ids[:, k], x_ids[k])
        rank[:, k] = r
    return acc, rank


def analyze(arm, pretrain_step, seed, condition, checkpoint_dir, total_steps,
            freeze_arm=None, lr_mult=1.0, inject_batch=256, dense=False, fig=False):
    """`freeze_arm` reads the freeze-ablation checkpoints instead of the unfrozen baseline.
    Phase, key_extra and filename prefix all switch together, since the key is what makes
    those distinct checkpoints rather than a naming convention."""
    spec = dict(ARMS[arm])
    if freeze_arm is not None:
        if arm != "mlp_free":
            raise SystemExit("freeze arms exist only for the mlp_free arm")
        spec["phase"] = "mlpfreefreezeinject"
        spec["prefix"] = f"mlpfreefreezeinject-{freeze_arm}"
        spec["key_extra"] = {**spec["key_extra"], "freeze_arm": freeze_arm}
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    # Dose conditions differ only in the inject config, which is already
    # hashed, so lr_mult=1/batch=256 reproduces the baseline key exactly.
    cfg = replace(cfg, inject=replace(cfg.inject, batch_size=inject_batch,
                  **({'checkpoint_steps': DENSE_SCHEDULE} if dense else {}),
                  **({'checkpoint_steps': FIG_SCHEDULE} if fig else {})))
    pop, std_model, data_cfg, _, data_a, _ballast, _b, _c = build(
        cfg, cfg.inject.max_eval_people)
    model = std_model if arm == "standard" else create_mlpfree_model(cfg.model, data_cfg)
    vocab = data_cfg.vocab_size

    pre_key = (pretrain_key(cfg, data_cfg) if arm == "standard"
               else mlpfree_pretrain_key(cfg, data_cfg))
    inject_opt = replace(cfg.inject.opt,
                         peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO * lr_mult)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
        pretrain_key=pre_key, pretrain_step=pretrain_step, seed=cfg.seed,
        **spec["key_extra"])

    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]

    tx = make_optimizer(replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO),
                        cfg.inject.total_steps)
    template = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    sched = FIG_SCHEDULE if fig else (DENSE_SCHEDULE if dense else SCHEDULES[total_steps])
    steps = [int(s) for s in sched.split(",")]

    def path_for(s):
        return os.path.join(
            checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")

    print(f"arm={arm} pretrain_step={pretrain_step} seed={seed} condition={condition} "
          f"inject_total_steps={total_steps}  vocab={vocab}")
    print(f"  pretrain key={pre_key}  inject key={key}")

    p0 = path_for(0)
    if not os.path.exists(p0):
        raise SystemExit(f"step-0 checkpoint missing, nothing to difference against: {p0}")
    params0 = load_params(p0, template.params)
    bias_path = find_readout_bias_path(params0, vocab)
    bias0 = np.asarray(flatten_dict(params0)[bias_path])
    print(f"  readout bias leaf: {bias_path}  shape={bias0.shape}")

    z0, correct_ids = row_logits_all(model, params0, data_a, vocab, want_targets=True)
    acc0, rank0 = metrics_from(z0, correct_ids, x_ids)
    print(f"  step 0 baseline: first_acc={acc0.mean():.4f} rank_own={rank0.mean():.3f}")
    print()

    print(f"  {'step':>5} | {'trained':>16} | {'const-only':>16} | {'vary-only':>16} | "
          f"{'nobias':>16} | {'%energy':>8} {'bias%':>7}")
    print(f"  {'':>5} | {'acc':>7}{'rank':>9} | {'acc':>7}{'rank':>9} | "
          f"{'acc':>7}{'rank':>9} | {'acc':>7}{'rank':>9} | {'const':>8} {'of c':>7}")

    keys = ("steps", "acc_trained", "rank_trained", "acc_const", "rank_const",
            "acc_vary", "rank_vary", "acc_nobias", "rank_nobias",
            "frac_energy_const", "bias_share_of_c", "c_norm", "bias_norm",
            "nll_full", "nll_own")
    out = {k: [] for k in keys}
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING ({p})")
            continue
        params_t = load_params(p, template.params)
        bias_t = np.asarray(flatten_dict(params_t)[bias_path])
        params_nb = with_bias_reset(params_t, bias_path, jnp.array(bias0))

        zt = row_logits_all(model, params_t, data_a, vocab)
        znb = row_logits_all(model, params_nb, data_a, vocab)

        dz = zt - z0
        c = dz.mean(axis=0)                                   # (6, V)
        e_const = float(dz.shape[0] * (c ** 2).sum())
        e_vary = float(((dz - c[None, :, :]) ** 2).sum())
        frac_const = e_const / max(e_const + e_vary, 1e-30)

        a_tr, r_tr = metrics_from(zt, correct_ids, x_ids)
        nf, no = nll_from(zt, correct_ids, x_ids)
        a_cn, r_cn = metrics_from(z0 + c[None, :, :], correct_ids, x_ids)
        a_vr, r_vr = metrics_from(zt - c[None, :, :], correct_ids, x_ids)
        a_nb, r_nb = metrics_from(znb, correct_ids, x_ids)

        # ATTRIBUTION, weaker than the nobias intervention above: how much of the
        # individual-independent shift the readout bias delta alone accounts for. c is
        # per-attribute and db is global, so db should appear in every c_k equally.
        db = bias_t - bias0
        resid = float(((c - db[None, :]) ** 2).sum())
        total = float((c ** 2).sum())
        bias_share = 1.0 - resid / max(total, 1e-30)

        print(f"  {s:>5} | {a_tr.mean():>7.4f}{r_tr.mean():>9.3f} | "
              f"{a_cn.mean():>7.4f}{r_cn.mean():>9.3f} | "
              f"{a_vr.mean():>7.4f}{r_vr.mean():>9.3f} | "
              f"{a_nb.mean():>7.4f}{r_nb.mean():>9.3f} | "
              f"{frac_const:>8.4f} {bias_share:>7.4f}")

        out["steps"].append(s)
        for name, val in (("acc_trained", a_tr), ("rank_trained", r_tr),
                          ("acc_const", a_cn), ("rank_const", r_cn),
                          ("acc_vary", a_vr), ("rank_vary", r_vr),
                          ("acc_nobias", a_nb), ("rank_nobias", r_nb)):
            out[name].append(val)
        out["nll_full"].append(float(nf.mean()))
        out["nll_own"].append(float(no.mean()))
        out["frac_energy_const"].append(frac_const)
        out["bias_share_of_c"].append(bias_share)
        out["c_norm"].append(float(np.sqrt((c ** 2).sum())))
        out["bias_norm"].append(float(np.linalg.norm(db)))
        del zt, znb, dz, c

    print()
    print("  ||c|| by step:   " + "  ".join(
        f"{s}:{v:.3f}" for s, v in zip(out["steps"], out["c_norm"])))
    print("  ||d_bias||:      " + "  ".join(
        f"{s}:{v:.3f}" for s, v in zip(out["steps"], out["bias_norm"])))
    print("  (does ||c|| SHRINK between trough and peak? that is the recovery question)")

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(
        OUT_DIR, f"{arm}{'-' + freeze_arm if freeze_arm else ''}"
        f"{'' if (lr_mult == 1.0 and inject_batch == 256) else f'-lr{lr_mult}-bs{inject_batch}'}"
        f"{'-dense' if dense else ''}{'-fig' if fig else ''}"
        f"-p{pretrain_step}-{condition}-t{total_steps}-seed{seed}.npz")
    np.savez_compressed(
        dest,
        steps=np.array(out["steps"]),
        **{k: np.array(out[k]).astype(np.float32) for k in keys if k != "steps"},
        acc_baseline=acc0.astype(np.float32), rank_baseline=rank0.astype(np.float32),
        arm=arm, pretrain_step=pretrain_step, seed=seed, condition=condition,
        inject_total_steps=total_steps)
    print(f"  saved -> {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--pretrain_step", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--inject_total_steps", type=int, default=None)
    ap.add_argument("--dense_checkpoints", action="store_true")
    ap.add_argument("--fig_checkpoints", action="store_true")
    ap.add_argument("--lr_mult", type=float, default=1.0)
    ap.add_argument("--inject_batch", type=int, default=256)
    ap.add_argument("--freeze_arm", default=None,
                    choices=sorted(FREEZE_ARMS),
                    help="read the freeze-ablation checkpoints for this arm")
    args = ap.parse_args()
    total = args.inject_total_steps or ARMS[args.arm]["default_total_steps"]
    analyze(args.arm, args.pretrain_step, args.seed, args.condition,
            args.checkpoint_dir, total, args.freeze_arm,
            args.lr_mult, args.inject_batch, args.dense_checkpoints, args.fig_checkpoints)


if __name__ == "__main__":
    main()
