"""Does the constant component's 38% fall live in the READOUT or in A's REPRESENTATION?

The shift decomposition (analyze_shift_decomposition.py) established that injection's
individual-independent logit shift `c(t)` peaks at the trough and falls 37.9% by the recovery
peak, and that the readout BIAS carries none of it. But `c(t)` depends on both the readout
weights and A's pre-readout representation, and a logit-level split cannot separate those. This
splits at a different seam: the parameter tree is `{input_layer, backbone, head}`, so
`input_layer + backbone` is the representation and `head` is the readout, and the two can be
crossed.

Four parameter combinations per checkpoint, all exact (the head is a plain `nn.Dense`, so
`z = rep @ K + b` and every combination is a valid forward pass):

    z_00   rep from step 0, readout from step 0     the pre-injection baseline
    z_tt   rep from step t, readout from step t     what actually happened
    z_0t   rep from step 0, readout from step t     the READOUT moving, alone
    z_t0   rep from step t, readout from step 0     the REPRESENTATION moving, alone

and the constant components

    c_full     = mean_a (z_tt - z_00)
    c_readout  = mean_a (z_0t - z_00)
    c_rep      = mean_a (z_t0 - z_00)
    c_cross    = c_full - c_readout - c_rep

`c_cross` is exact, not a residual of convenience: because the head is linear,
`rep_t K_t - rep_0 K_0 = (rep_t - rep_0) K_0 + rep_0 (K_t - K_0) + (rep_t - rep_0)(K_t - K_0)`,
and the third term is precisely what the two single-factor counterfactuals leave out. So the
three reported pieces sum to `c_full` by construction, and a large cross term is itself
informative -- it would mean the two factors only matter jointly.

WHAT THE ANSWER DECIDES (pre-registered).

  * The fall lives in `c_rep` -- A's representation moving. Recovery is A's KEY changing,
    and a toy in which the shared key direction is trainable is the right next model.
  * The fall lives in `c_readout` -- the store genuinely reversing on its own. Then
    something inside the store un-writes the depression, a trainable key direction is the
    WRONG fix, and the line stops here pending a different account.
  * The fall is in `c_cross` -- neither factor alone explains it, and the mechanism is
    irreducibly joint. Report and stop before building.

Run (one invocation per arm/pretrain_step/seed):
  uv run python -m src.experiments.analyze_rep_readout_split \\
      --arm mlp_free --pretrain_step 16000 --seed 0
"""
import argparse
import os
from dataclasses import replace

import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, get_partition, init_state, make_optimizer, pretrain_key,
)
from src.experiments.analyze_mlpfree import (
    ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for,
)
from src.experiments.analyze_shift_decomposition import (
    OUT_DIR as SHIFT_OUT_DIR, metrics_from, row_logits_all,
)
from src.experiments.mlpfree_freeze_injection import ARMS as FREEZE_ARMS
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "rep_readout_split"
REP_KEYS = ("input_layer", "backbone")
READ_KEYS = ("head",)


def splice(rep_params, read_params):
    """Representation parameters from one tree, readout parameters from another.

    Asserts the two trees have exactly the expected top-level structure, so a change to the
    model factory fails here instead of silently producing a half-spliced tree.
    """
    expected = set(REP_KEYS) | set(READ_KEYS)
    for name, tree in (("rep", rep_params), ("read", read_params)):
        got = set(tree.keys())
        if got != expected:
            raise RuntimeError(f"{name} params top-level keys {sorted(got)} != "
                               f"{sorted(expected)}; the model factory changed")
    out = {k: rep_params[k] for k in REP_KEYS}
    out.update({k: read_params[k] for k in READ_KEYS})
    return out


def analyze(arm, pretrain_step, seed, condition, checkpoint_dir, total_steps,
            freeze_arm=None):
    """`freeze_arm` selects the freeze-ablation checkpoints instead of the baseline injection
    ones. Those were written by mlpfree_freeze_injection.py under a different phase, key and
    filename prefix, so all three have to be switched together --
    the key is what makes them distinct checkpoints rather than a filename convention."""
    spec = dict(ARMS[arm])
    if freeze_arm is not None:
        if arm != "mlp_free":
            raise SystemExit("freeze arms exist only for the mlp_free arm")
        spec["phase"] = "mlpfreefreezeinject"
        spec["prefix"] = f"mlpfreefreezeinject-{freeze_arm}"
        spec["key_extra"] = {**spec["key_extra"], "freeze_arm": freeze_arm}
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, std_model, data_cfg, _, data_a, _ballast, _b, _c = build(
        cfg, cfg.inject.max_eval_people)
    model = std_model if arm == "standard" else create_mlpfree_model(cfg.model, data_cfg)
    vocab = data_cfg.vocab_size

    pre_key = (pretrain_key(cfg, data_cfg) if arm == "standard"
               else mlpfree_pretrain_key(cfg, data_cfg))
    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
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
    steps = [int(s) for s in SCHEDULES[total_steps].split(",")]

    def path_for(s):
        return os.path.join(
            checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")

    print(f"arm={arm} pretrain_step={pretrain_step} seed={seed} condition={condition} "
          f"inject_total_steps={total_steps}  vocab={vocab}")
    print(f"  representation = {REP_KEYS}   readout = {READ_KEYS}")

    p0path = path_for(0)
    if not os.path.exists(p0path):
        raise SystemExit(f"step-0 checkpoint missing: {p0path}")
    params0 = load_params(p0path, template.params)
    z00, correct_ids = row_logits_all(model, params0, data_a, vocab, want_targets=True)
    acc0, rank0 = metrics_from(z00, correct_ids, x_ids)
    print(f"  step 0 baseline: first_acc={acc0.mean():.4f} rank_own={rank0.mean():.3f}")
    print()
    print(f"  {'step':>5} | {'||c|| full':>10} {'readout':>9} {'rep':>9} {'cross':>9} | "
          f"{'const-only acc':>30} | {'trained':>8}")
    print(f"  {'':>5} | {'':>10} {'':>9} {'':>9} {'':>9} | "
          f"{'full':>7}{'readout':>8}{'rep':>7}{'cross':>8} | {'acc':>8}")

    keys = ("steps", "c_full", "c_readout", "c_rep", "c_cross",
            "acc_full", "acc_readout", "acc_rep", "acc_cross", "acc_trained",
            "rank_full", "rank_readout", "rank_rep", "rank_cross", "rank_trained")
    out = {k: [] for k in keys}
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING ({p})")
            continue
        params_t = load_params(p, template.params)

        ztt = row_logits_all(model, params_t, data_a, vocab)
        z0t = row_logits_all(model, splice(params0, params_t), data_a, vocab)
        zt0 = row_logits_all(model, splice(params_t, params0), data_a, vocab)

        c_full = (ztt - z00).mean(axis=0)
        c_read = (z0t - z00).mean(axis=0)
        c_rep = (zt0 - z00).mean(axis=0)
        c_cross = c_full - c_read - c_rep

        a_tr, r_tr = metrics_from(ztt, correct_ids, x_ids)
        variants = {}
        for name, cvec in (("full", c_full), ("readout", c_read),
                           ("rep", c_rep), ("cross", c_cross)):
            a, r = metrics_from(z00 + cvec[None, :, :], correct_ids, x_ids)
            variants[name] = (float(a.mean()), float(r.mean()),
                              float(np.sqrt((cvec ** 2).sum())))

        print(f"  {s:>5} | {variants['full'][2]:>10.2f} {variants['readout'][2]:>9.2f} "
              f"{variants['rep'][2]:>9.2f} {variants['cross'][2]:>9.2f} | "
              f"{variants['full'][0]:>7.4f}{variants['readout'][0]:>8.4f}"
              f"{variants['rep'][0]:>7.4f}{variants['cross'][0]:>8.4f} | "
              f"{a_tr.mean():>8.4f}")

        out["steps"].append(s)
        out["acc_trained"].append(float(a_tr.mean()))
        out["rank_trained"].append(float(r_tr.mean()))
        for name in ("full", "readout", "rep", "cross"):
            out[f"acc_{name}"].append(variants[name][0])
            out[f"rank_{name}"].append(variants[name][1])
            out[f"c_{name}"].append(variants[name][2])
        del ztt, z0t, zt0

    st = np.array(out["steps"])
    print()
    for name in ("full", "readout", "rep", "cross"):
        arr = np.array(out[f"c_{name}"])
        i = int(arr.argmax())
        drop = 1.0 - arr[-1] / max(arr[i], 1e-30)
        print(f"  ||c_{name:<8}||  max={arr[i]:>8.2f}@{st[i]:<5d} end={arr[-1]:>8.2f}  "
              f"drop from max={drop:>6.1%}")
    print("  (which of readout / rep / cross carries the trough-to-peak FALL is the answer)")

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(
        OUT_DIR, f"{arm}{'-' + freeze_arm if freeze_arm else ''}-p{pretrain_step}"
        f"-{condition}-t{total_steps}-seed{seed}.npz")
    np.savez_compressed(
        dest, steps=st,
        **{k: np.array(out[k]).astype(np.float32) for k in keys if k != "steps"},
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
    ap.add_argument("--freeze_arm", default=None,
                    choices=sorted(FREEZE_ARMS),
                    help="read the freeze-ablation checkpoints for this arm instead of "
                         "the unfrozen baseline")
    args = ap.parse_args()
    total = args.inject_total_steps or ARMS[args.arm]["default_total_steps"]
    analyze(args.arm, args.pretrain_step, args.seed, args.condition,
            args.checkpoint_dir, total, args.freeze_arm)


if __name__ == "__main__":
    main()
