"""Do attention PATTERNS move during the recovery window? The finer test.

WHY THE EXISTING EVIDENCE IS TOO WEAK FOR THE CONCLUSION IT CARRIES. The earlier analysis ruled
routing out on `attention_to_name_mass`, which is ONE SCALAR PER LAYER: the total mass landing
on name tokens, averaged over heads, people and attributes. A pattern can reorganise
substantially without moving that aggregate at all -- mass can shift between name tokens, or
between name and template positions, with the sum unchanged. That is a weaker basis than the
routing elimination deserves, and the elimination is load-bearing for how a QK/VO freeze arm
would be read.

ALSO MEASURES B'S PATTERNS, which answer a DIFFERENT question. Everything above concerns A.
B introduces individuals whose name tokens were never seen, so each is a new key that has to
be attended to at the right position. If the recall circuit is already general, B's patterns
should resemble A's from step 0 and stay flat -- injection needs no routing work, and the
phenomenon happens entirely downstream of retrieval. If B's start different and converge
toward A's, routing ADAPTED during injection, and whether that adaptation is what disturbs A
becomes an open question.

NOTE ON THE Q/K-FROZEN ARM. Freezing the query and key WEIGHTS does not freeze the patterns:
Q and K are computed from a residual stream that keeps moving as embeddings, LayerNorm and V/O
train. So B's patterns can adapt in that arm too, through the embeddings. Running this on the
Q/K-frozen checkpoints is the discriminating case -- flat patterns there mean the circuit
really was general; moving patterns mean routing adapted via a different parameter group.

WHAT THIS MEASURES INSTEAD. The full per-head attention distribution at the value-slot query
position -- the same column `first_token_positions` returns, i.e. the one whose logits predict
the attribute's first value token -- at the same dense checkpoints as the split runs, through
the recovery window (trough ~50 to peak ~200) and out to 1200.

For layer l, head h, individual a, attribute k, let `p` be the distribution over key positions
and `p0` its step-0 counterpart. Reported per (layer, head):

  * mean total-variation distance  TV = 0.5 * sum_j |p - p0|, in [0, 1] and directly
    interpretable as "fraction of attention mass that moved";
  * the COMMON / INDIVIDUAL-SPECIFIC split of the movement, which is the part that matters:

        d_a   = p_a - p0_a                      per-individual change
        c     = mean_a d_a                      common across individuals
        v_a   = d_a - c                         individual-specific residual

    reported as ||c|| and rms(||v_a||), plus the common FRACTION of the movement's energy.

WHY THE SPLIT IS THE DECISIVE PART. Recovery lives entirely in the constant-across-individuals
logit component. For routing to be the mechanism behind it, pattern change
would have to be roughly UNIFORM across individuals. Individual-specific pattern movement
lands in the varying component and cannot explain the constant one, however large it is.

TWO READINGS, both worth having:
  * Patterns flat under this finer measure too -> the routing elimination holds on much
    stronger evidence, and a positive QK/VO arm would implicate the value/output projections
    specifically.
  * Patterns move substantially in a way the mass summary hid, ESPECIALLY if the movement is
    common across individuals -> the routing hypothesis is live, the QK/VO arm becomes the
    priority, and the earlier conclusion needs revising.

Run:
  uv run python -m src.experiments.analyze_attention_patterns \\
      --arm mlp_free --pretrain_step 16000 --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, init_state, make_optimizer, pretrain_key,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for
from src.experiments.mlpfree_freeze_injection import ARMS as FREEZE_ARMS
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "attention_patterns"
# Distributions are (people, heads, keys) per layer per attribute; 256 people keeps the
# whole comparison in memory while giving a stable mean over individuals.
N_PEOPLE = 256
BATCH = 32


def value_slot_patterns(model, params, ds, n_people=N_PEOPLE, batch_size=BATCH):
    """(n_layers, n_people, n_heads, L) attention distributions at the value-slot query,
    averaged over the six attributes.

    Averaging over attributes before the comparison is deliberate: each attribute's
    retrieval position sits at a different column, so their distributions are not
    commensurable key-by-key. What IS commensurable is each individual's average pattern,
    and the question here is whether that moves.
    """
    n = min(n_people, ds.eval_inputs.shape[0])
    out = None
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        x = jnp.array(ds.eval_inputs[sl])
        _logits, mutated = model.apply(
            {"params": params} if "params" not in params else params,
            x, deterministic=True, mutable=["intermediates"])
        inter = mutated["intermediates"]["backbone"]
        layers = sorted((k for k in inter if k.startswith("block_")),
                        key=lambda s: int(s.split("_")[1]))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        rows = np.arange(sl.stop - sl.start)
        if out is None:
            probe = np.asarray(inter[layers[0]]["CausalSelfAttention_0"]["attn_weights"][0])
            out = np.zeros((len(layers), n, probe.shape[1], probe.shape[3]), dtype=np.float32)
        for li, lname in enumerate(layers):
            attn = np.asarray(inter[lname]["CausalSelfAttention_0"]["attn_weights"][0])
            acc = np.zeros((len(rows), attn.shape[1], attn.shape[3]), dtype=np.float64)
            for k in range(NUM_ATTRIBUTES):
                cols, _ = first_token_positions(mask, targets, k)
                acc += attn[rows, :, cols, :]           # (B, H, L)
            out[li, sl, :, :] = (acc / NUM_ATTRIBUTES).astype(np.float32)
    return out


def compare(p, p0):
    """Per (layer, head): mean TV distance, and the common / individual-specific split."""
    d = p - p0                                            # (L, N, H, K)
    tv = 0.5 * np.abs(d).sum(axis=-1).mean(axis=1)        # (L, H)
    c = d.mean(axis=1)                                    # (L, H, K) common across people
    v = d - c[:, None, :, :]                              # individual-specific residual
    c_norm = np.linalg.norm(c, axis=-1)                   # (L, H)
    v_rms = np.sqrt((v ** 2).sum(axis=-1).mean(axis=1))   # (L, H)
    energy_c = (c ** 2).sum(axis=-1) * d.shape[1]
    energy_v = (v ** 2).sum(axis=(-1, 1))
    frac_common = energy_c / np.maximum(energy_c + energy_v, 1e-30)
    return tv, c_norm, v_rms, frac_common


def analyze(arm, pretrain_step, seed, condition, checkpoint_dir, total_steps, freeze_arm=None):
    spec = dict(ARMS[arm])
    if freeze_arm is not None:
        spec["phase"] = "mlpfreefreezeinject"
        spec["prefix"] = f"mlpfreefreezeinject-{freeze_arm}"
        spec["key_extra"] = {**spec["key_extra"], "freeze_arm": freeze_arm}
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, std_model, data_cfg, _, data_a, _ballast, data_b, _c = build(
        cfg, cfg.inject.max_eval_people)
    model = std_model if arm == "standard" else create_mlpfree_model(cfg.model, data_cfg)
    pre_key = (pretrain_key(cfg, data_cfg) if arm == "standard"
               else mlpfree_pretrain_key(cfg, data_cfg))
    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
        pretrain_key=pre_key, pretrain_step=pretrain_step, seed=cfg.seed, **spec["key_extra"])

    tx = make_optimizer(inject_opt, cfg.inject.total_steps)
    template = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    steps = [int(s) for s in SCHEDULES[total_steps].split(",")]
    path_for = lambda s: os.path.join(
        checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")

    print(f"arm={arm} freeze_arm={freeze_arm} p={pretrain_step} seed={seed} key={key}")
    p0path = path_for(0)
    if not os.path.exists(p0path):
        raise SystemExit(f"step-0 checkpoint missing: {p0path}")
    params0 = load_params(p0path, template.params)
    P0 = value_slot_patterns(model, params0, data_a)
    # B's patterns matter for a DIFFERENT question than A's. B introduces individuals whose
    # name tokens were never seen, so each is a new key that has to be attended to at the
    # right position. If the recall circuit is already general, B's patterns should resemble
    # A's from step 0 and stay put; if routing has to accommodate B, they should start
    # different and converge. That is not answerable from A's patterns at all.
    B0 = value_slot_patterns(model, params0, data_b)
    n_layers, n_people, n_heads, n_keys = P0.shape
    print(f"  patterns: {n_layers} layers x {n_people} people x {n_heads} heads x "
          f"{n_keys} key positions\n")
    print(f"  {'step':>5} | {'A: TV':>8}{'max':>8} | {'A ||c||':>11}"
          f"{'A rms||v||':>14} | {'A com frac':>12} | {'B: TV':>8}{'B com f':>10} | {'TV(A,B)':>8}")

    rec = {k: [] for k in ("steps", "tv_mean", "tv_max", "c_mean", "v_mean", "frac_common",
                           "tv_b", "frac_common_b", "tv_ab")}
    per_head, per_head_frac, per_head_c = [], [], []
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING")
            continue
        params_t = load_params(p, template.params)
        P = value_slot_patterns(model, params_t, data_a)
        Bt = value_slot_patterns(model, params_t, data_b)
        tv, c_norm, v_rms, frac = compare(P, P0)
        tv_b, _cb, _vb, frac_b = compare(Bt, B0)                    # B's own drift
        # A-vs-B similarity: TV between the POPULATION-MEAN patterns, per head. A and B have
        # the same prompt structure, so their mean distributions over key positions are
        # comparable even though the individuals differ.
        ab = 0.5 * np.abs(P.mean(axis=1) - Bt.mean(axis=1)).sum(axis=-1)   # (L, H)
        rec["tv_b"].append(float(tv_b.mean()))
        rec["frac_common_b"].append(float(frac_b.mean()))
        rec["tv_ab"].append(float(ab.mean()))
        print(f"  {s:>5} | {tv.mean():>8.4f}{tv.max():>8.4f} | {c_norm.mean():>11.4f}"
              f"{v_rms.mean():>14.4f} | {frac.mean():>12.4f} | {tv_b.mean():>8.4f}"
              f"{frac_b.mean():>10.4f} | {ab.mean():>8.4f}")
        rec["steps"].append(s); rec["tv_mean"].append(float(tv.mean()))
        rec["tv_max"].append(float(tv.max())); rec["c_mean"].append(float(c_norm.mean()))
        rec["v_mean"].append(float(v_rms.mean())); rec["frac_common"].append(float(frac.mean()))
        per_head.append(tv)
        per_head_frac.append(frac)      # (L, H) -- saved so a single head that moves BOTH
        per_head_c.append(c_norm)       # a lot AND uniformly cannot hide inside the mean
        del P, Bt

    st = np.array(rec["steps"])
    if len(st) >= 2:
        print()
        print(f"  B's OWN pattern drift over injection: {rec['tv_b'][0]:.4f} -> {rec['tv_b'][-1]:.4f}")
        print(f"  TV(A, B) population-mean patterns:    {rec['tv_ab'][0]:.4f} (step 0) -> "
              f"{rec['tv_ab'][-1]:.4f} (end)")
        print("  READING: B's patterns already resembling A's at step 0 AND staying flat means")
        print("    the recall circuit was general before injection and needed no routing work;")
        print("    B starting different and converging toward A means routing ADAPTED.")
    w = [i for i, s in enumerate(st) if 50 <= s <= 200]
    if len(w) >= 2:
        print(f"\n  RECOVERY WINDOW (steps {st[w[0]]}-{st[w[-1]]}): "
              f"TV moves {rec['tv_mean'][w[0]]:.4f} -> {rec['tv_mean'][w[-1]]:.4f}, "
              f"delta={rec['tv_mean'][w[-1]] - rec['tv_mean'][w[0]]:+.4f}")
        print(f"  common fraction of that movement: {rec['frac_common'][w[-1]]:.4f}")
        print("  (routing can only explain the CONSTANT logit component if pattern change is")
        print("   roughly uniform across individuals, i.e. common fraction near 1)")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{arm}{'-' + freeze_arm if freeze_arm else ''}"
    dest = os.path.join(OUT_DIR, f"{tag}-p{pretrain_step}-{condition}-t{total_steps}-seed{seed}.npz")
    np.savez_compressed(dest, steps=st, per_head_tv=np.array(per_head),
                        per_head_frac_common=np.array(per_head_frac),
                        per_head_c_norm=np.array(per_head_c),
                        **{k: np.array(rec[k], dtype=np.float32) for k in rec if k != "steps"},
                        arm=arm, freeze_arm=str(freeze_arm), pretrain_step=pretrain_step,
                        seed=seed, condition=condition, inject_total_steps=total_steps)
    print(f"  saved -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--inject_total_steps", type=int, default=None)
    ap.add_argument("--freeze_arm", default=None,
                    choices=sorted(FREEZE_ARMS))
    a = ap.parse_args()
    analyze(a.arm, a.pretrain_step, a.seed, a.condition, a.checkpoint_dir,
            a.inject_total_steps or ARMS[a.arm]["default_total_steps"], a.freeze_arm)


if __name__ == "__main__":
    main()
