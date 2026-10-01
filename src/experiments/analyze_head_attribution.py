"""Which attention heads write the constant shift `c`?

WHY THIS RATHER THAN ANOTHER FREEZE ARM. Freezing conflates two roles a component may play:
it is where the shift on A gets *written*, or it is where B's facts get *stored*. A component
doing the second throttles B, and since the shift is *caused* by B's learning, a throttled arm
shows no crash and no recovery -- exactly what a component doing the first would produce. That
is what makes the attention arm uninterpretable: its B reaches ceiling at step
830 against baseline's ~135. **This script measures the write instead of removing the writer.**
It runs on existing checkpoints with no training, so B learns exactly as in baseline and the
confound cannot arise.

THE DECOMPOSITION IS EXACT, NOT AN APPROXIMATION. The MLP-free block is `x = x + Attn(LN(x))`
and attention's output is a single `out` Dense over concatenated heads, so the residual stream
before the final LayerNorm is exactly

    res = emb + sum_l ( sum_h head_{l,h} W_out[h] + b_out_l )

LayerNorm is affine once its scale is fixed -- `LN(sum_i c_i) = sum_i (c_i - mean(c_i))/s * g + beta`
with `s` computed from the full residual -- so every component's logit contribution is exact
given `s`. Summing them reproduces the model's own logits, and the script ASSERTS that to 1e-3
rather than trusting the derivation.

WHAT IS REPORTED. Per layer and per head, the head's share of `c(t) = mean_a [z_a(t) - z_a(0)]`,
both as a projection share `<c_h, c> / ||c||^2` (signed, sums to 1 over all components, so
cancellation is visible) and as `||c_h||`. Sorted, with cumulative shares.

PRE-REGISTERED EXPECTATION, recorded before running so a flat result is not read as a failed
measurement. The freeze arms found no single parameter group necessary for recovery, which is
weak prior evidence for a DISTRIBUTED write. If the share is spread roughly evenly across all
heads, that is the finding -- the mechanism lives in attention's OV path distributed rather than
in a circuit -- and head patching would isolate nothing. Only a concentrated distribution makes
Step 2 worth running.

FRAMING. This attributes what heads WRITE (their OV contribution), not where they attend.
The pattern measurement and the Q/K arm both rule out routing. A result here is
"these heads' OV contribution carries the constant shift", never a routing circuit.

Run (one invocation per seed):
  uv run python -m src.experiments.analyze_head_attribution \\
      --arm mlp_free --pretrain_step 16000 --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

# JAX defaults float32 matmuls to TF32 on GPU -- 10 mantissa bits, ~3e-4 relative. That makes
# the MODEL's forward pass the imprecise side of this comparison: the reconstruction check
# failed at 1.29e-02 on the cluster (max logit 27.8, so 4.6e-4 relative) while passing at
# 3e-06 on CPU, identically across seeds, and unchanged by moving the attribution to float64.
# Analysis wants an accurate read of fixed weights, so ask for real float32 throughout.
jax.config.update("jax_default_matmul_precision", "highest")

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, get_partition, init_state, make_optimizer,
    pretrain_key,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "head_attribution"
# Smaller than the shift-decomposition script's 128: this materialises per-head residual
# contributions at (B, 6, n_heads_total, model_dim) before summing over the batch.
BATCH_SIZE = 64


def _get(tree, *path):
    """Navigate a params/intermediates tree, failing with the available keys rather than a
    bare KeyError -- module autonaming is the thing most likely to drift here."""
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"{'/'.join(map(str, path))}: missing {p!r}; "
                           f"available at this level: {sorted(node.keys())}")
        node = node[p]
    return node


def value_slot_columns(mask, targets):
    """(B, NUM_ATTRIBUTES) column of each attribute's first value token."""
    cols = np.zeros((mask.shape[0], NUM_ATTRIBUTES), dtype=np.int64)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(mask, targets, k)
    return cols


def component_contributions(model, params, inputs, cols, n_layers, n_heads, model_dim):
    """Per-head residual contributions and the final-LayerNorm gain, at the value slots only.

    Returns (heads, other, mul) with
      heads (B, A, n_layers * n_heads, E) -- each head's contribution to the residual stream
      other (B, A, E)                     -- embeddings plus every attention out-bias
      mul   (B, A)                        -- the final LayerNorm's scalar gain

    FLOAT64, AND ONLY AT THE VALUE SLOTS. Both together, because the second is what makes the
    first affordable. A trained model's heads write large, heavily cancelling vectors, so
    summing 64 of them in float32 loses far more precision than the size of the sum suggests:
    the first version reconstructed the logits only to 1.29e-02, identically across three
    seeds. Restricting to the six positions that are actually read drops the work ~32x, which
    buys float64 for free.

    `mul` is READ OFF THE MODEL'S OWN LAYERNORM OUTPUT rather than recomputed. Recomputing it
    as sqrt(np.var(res) + eps) failed reconstruction at 1.29e-02, identically across three
    seeds: flax uses the fast variance `max(0, mean(x^2) - mean(x)^2)`, whose float32
    cancellation differs from numpy's two-pass form. Solving for the gain that the model's own
    LayerNorm actually applied is exact whatever arithmetic flax uses, and cannot drift with a
    flax version.
    """
    _, aux = model.apply({"params": params}, inputs, deterministic=True,
                         capture_intermediates=True, mutable=["intermediates"])
    inter = aux["intermediates"]

    emb = np.asarray(_get(inter, "input_layer", "__call__")[0])
    B, L, E = emb.shape
    A = cols.shape[1]
    bi = np.arange(B)[:, None]
    head_dim = model_dim // n_heads
    heads = np.zeros((B, A, n_layers * n_heads, E), dtype=np.float64)
    other = emb[bi, cols].astype(np.float64)

    for i in range(n_layers):
        blk = _get(inter, "backbone", f"block_{i}", "CausalSelfAttention_0")
        v = np.asarray(_get(blk, "value", "__call__")[0])          # (B, L, qkv)
        attn = np.asarray(_get(blk, "attn_weights")[0])            # (B, H, L, L)
        pblk = _get(params, "backbone", f"block_{i}", "CausalSelfAttention_0", "out")
        w_out = np.asarray(pblk["kernel"]).astype(np.float64)      # (qkv, E)
        other += np.asarray(pblk["bias"]).astype(np.float64)

        v = v.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3).astype(np.float64)
        # gather the query rows first: only the value slots are ever read
        aq = attn.transpose(0, 2, 1, 3)[bi, cols].astype(np.float64)   # (B, A, H, L)
        ov = np.einsum("bahk,bhkd->bahd", aq, v)                       # (B, A, H, D)
        for h in range(n_heads):
            heads[:, :, i * n_heads + h, :] = (
                ov[:, :, h, :] @ w_out[h * head_dim:(h + 1) * head_dim])

    # Solve for the LayerNorm gain the model actually applied. With
    #   lnout = (res - mean(res)) * mul * gamma + beta
    # and `a = (res - mean(res)) * gamma`, least squares over features gives
    #   mul = <a, lnout - beta> / <a, a>
    # -- no division by gamma, so near-zero scale entries are harmless.
    res = np.asarray(
        _get(inter, "backbone", f"block_{n_layers - 1}", "__call__")[0])[bi, cols].astype(
            np.float64)
    lnout = np.asarray(
        _get(inter, "backbone", "LayerNorm_0", "__call__")[0])[bi, cols].astype(np.float64)
    ln = _get(params, "backbone", "LayerNorm_0")
    a = (res - res.mean(axis=-1, keepdims=True)) * np.asarray(ln["scale"]).astype(np.float64)
    mul = (a * (lnout - np.asarray(ln["bias"]))).sum(-1) / np.maximum((a * a).sum(-1), 1e-30)
    return heads, other, mul


def logit_map(params, vocab):
    """(E, V) matrix taking a centred, scale-divided residual component to logits, plus the
    constant offset every component-free path contributes."""
    ln = _get(params, "backbone", "LayerNorm_0")
    gamma = np.asarray(ln["scale"]).astype(np.float64)
    beta = np.asarray(ln["bias"]).astype(np.float64)
    hd = _get(params, "head", "Dense_0")
    w = np.asarray(hd["kernel"]).astype(np.float64)
    b = np.asarray(hd["bias"]).astype(np.float64)
    assert w.shape[1] == vocab, f"head kernel {w.shape} does not end in vocab {vocab}"
    return gamma[:, None] * w, beta @ w + b


def centred(x, mul):
    """(x - mean_features(x)) * mul, the linear part of LayerNorm applied to one component."""
    return (x - x.mean(axis=-1, keepdims=True)) * mul[..., None]


def mean_head_logits(model, params, ds, vocab, n_layers, n_heads, model_dim, verify=False,
                     readout=None):
    """Mean over individuals of every component's logit contribution at the value slots.

    Returns (heads (6, H, V), other (6, V)). The mean is taken over individuals BEFORE the
    projection to logit space, which is exact because the projection is linear and identical
    for every individual -- and turns a 5.4e10-flop-per-batch matmul into a 4.2e8 one.
    """
    n = ds.eval_inputs.shape[0]
    H = n_layers * n_heads
    acc_h = np.zeros((NUM_ATTRIBUTES, H, model_dim), dtype=np.float64)
    acc_o = np.zeros((NUM_ATTRIBUTES, model_dim), dtype=np.float64)
    # `readout` pins the logit map to another checkpoint's head+final-LN. With it at step 0,
    # c_h is the REPRESENTATION-side attribution only: what changed about what the head writes,
    # with the readout held fixed. Without it, a head is also credited with the readout's own
    # change wherever its residual is large, which conflates the two channels 3.16 separated --
    # and 3.16 found the readout channel never reverses, so they behave differently.
    wg, offset = logit_map(params, vocab) if readout is None else readout

    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        x = jnp.array(ds.eval_inputs[sl])
        cols = value_slot_columns(np.asarray(ds.eval_mask[sl]), np.asarray(ds.eval_targets[sl]))
        heads, other, mul = component_contributions(
            model, params, x, cols, n_layers, n_heads, model_dim)
        norm_h = centred(heads, mul[..., None])                    # (B, A, H, E)
        acc_h += norm_h.sum(axis=0)
        acc_o += centred(other, mul).sum(axis=0)

        if verify and readout is None and start == 0:
            _verify(model, params, x, cols, norm_h, centred(other, mul), wg, offset)
            verify = False

    return (acc_h / n) @ wg, (acc_o / n) @ wg + offset


def _verify(model, params, x, cols, norm_h, norm_o, wg, offset):
    """Summing the components must reproduce the model's own logits. This is the check that
    makes the whole attribution trustworthy: it catches any reshape, head-slice or LayerNorm
    error at once, and every number downstream is meaningless without it.

    The cancellation ratio is printed alongside, because that is what decides whether a failure
    here is a bug or a precision limit: the components are far larger than their sum, so the
    sum is a difference of large numbers."""
    real = np.asarray(model.apply({"params": params}, x, deterministic=True))
    bi = np.arange(cols.shape[0])[:, None]
    recon = (norm_h.sum(axis=2) + norm_o) @ wg + offset
    worst = float(np.abs(recon - real[bi, cols]).max())
    biggest = float(np.abs(norm_h @ wg).max())
    print(f"  reconstruction check: max |sum(components) - logits| = {worst:.2e}")
    print(f"    max |logit| {np.abs(real[bi, cols]).max():.1f}, max |single head's logit "
          f"contribution| {biggest:.1f} -> cancellation ratio "
          f"{biggest / max(float(np.abs(real[bi, cols]).max()), 1e-30):.1f}x")
    print("    (a failure near 5e-4 x max|logit| with a low cancellation "
          "ratio means the model's own matmul precision, not this "
          "decomposition -- see the header comment.)")
    assert worst < 1e-3, (
        f"component decomposition does not reproduce the logits (max err {worst:.2e}); "
        f"every attribution below would be meaningless")


def analyze(arm, pretrain_step, seed, condition, checkpoint_dir, total_steps,
            fixed_readout=False):
    if arm != "mlp_free":
        # With MLPs present the residual also carries per-block MLP outputs, which this script
        # does not capture, so the decomposition would be incomplete. The reconstruction check
        # would catch it -- but failing here says why instead of just that.
        raise SystemExit("mlp_free only: the exact residual decomposition assumes "
                         "x = x + Attn(LN(x)) with no MLP term")
    spec = dict(ARMS[arm])
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, std_model, data_cfg, _, data_a, _b, _c, _d = build(cfg, cfg.inject.max_eval_people)
    model = std_model if arm == "standard" else create_mlpfree_model(cfg.model, data_cfg)
    vocab = data_cfg.vocab_size
    n_layers, n_heads, model_dim = cfg.model.num_layers, cfg.model.num_heads, cfg.model.model_dim

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

    def path_for(s):
        return os.path.join(
            checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")

    print(f"arm={arm} pretrain_step={pretrain_step} seed={seed} condition={condition}")
    print(f"  device={jax.devices()[0].platform} "
          f"matmul_precision={jax.config.jax_default_matmul_precision}")
    print(f"  {n_layers} layers x {n_heads} heads = {n_layers * n_heads} heads, "
          f"model_dim={model_dim}, vocab={vocab}")
    print(f"  inject key={key}")

    p0 = path_for(0)
    if not os.path.exists(p0):
        raise SystemExit(f"step-0 checkpoint missing: {p0}")
    params0 = load_params(p0, template.params)
    print(f"  step 0 (readout held {'at step 0' if fixed_readout else 'at each step'}):")
    ro = logit_map(params0, vocab) if fixed_readout else None
    h0, o0 = mean_head_logits(model, params0, data_a, vocab, n_layers, n_heads, model_dim,
                              verify=True, readout=ro)

    H = n_layers * n_heads
    out = {"steps": [], "c_head": [], "c_other": [], "c_total": []}
    print(f"\n  {'step':>5} {'||c||':>9} {'top1':>7} {'top4':>7} {'top8':>7} {'top16':>7} "
          f"{'even':>7} | {'argmax head':>14}")
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING ({p})")
            continue
        ht, ot = mean_head_logits(model, load_params(p, template.params), data_a, vocab,
                                  n_layers, n_heads, model_dim, readout=ro)
        c_head = ht - h0                                   # (6, H, V)
        c_other = ot - o0                                  # (6, V)
        c = c_head.sum(axis=1) + c_other                   # (6, V) == the c of 3.15
        cn2 = float((c ** 2).sum())
        share = np.einsum("ahv,av->h", c_head, c) / max(cn2, 1e-30)
        other_share = float((c_other * c).sum()) / max(cn2, 1e-30)
        order = np.argsort(-share)
        cum = np.cumsum(share[order])
        out["steps"].append(s)
        out["c_head"].append(c_head.astype(np.float32))
        out["c_other"].append(c_other.astype(np.float32))
        out["c_total"].append(c.astype(np.float32))
        top = order[0]
        q = [cum[min(k, H - 1)] for k in (0, 3, 7, 15)]
        print(f"  {s:>5} {np.sqrt(cn2):>9.2f} {q[0]:>7.3f} {q[1]:>7.3f} {q[2]:>7.3f} "
              f"{q[3]:>7.3f} {min(16, H) / H:>7.3f} | L{top // n_heads}H{top % n_heads} "
              f"{share[top]:>6.3f} | {other_share:>6.3f} {share.sum() + other_share:>6.3f}")

    print(f"\n  'even' is what top-16-of-{H} would be if every head contributed equally.")
    print("  Concentrated => Step 2 (head patching) is worth running.")
    print("  Flat        => the write is distributed across the OV path; patching isolates")
    print("                 nothing, and that is the finding (see this file's docstring).")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "-fixedro" if fixed_readout else ""
    stem = f"{arm}{tag}-p{pretrain_step}-{condition}-t{total_steps}-seed{seed}"
    meta = dict(steps=np.array(out["steps"]), num_layers=n_layers, num_heads=n_heads,
                arm=arm, condition=condition, seed=seed, pretrain_step=pretrain_step,
                inject_total_steps=total_steps, fixed_readout=fixed_readout)

    # TWO FILES, and the split is deliberate. `c_head` is (steps, 6, H, V) -- 31.8MB per seed
    # at 11 steps, 64 heads and vocab 1881, against 8-156KB for every other analysis npz in
    # this repo. It is kept so a later question about *which values* a head depresses does not
    # need a re-run, but it stays out of git (three seeds would be ~95MB in every clone
    # forever). The summary is what travels: norms and signed shares per head per step, a few
    # tens of KB, and enough for every figure and table.
    ch = np.stack(out["c_head"])                       # (steps, 6, H, V)
    co = np.stack(out["c_other"])
    ct = np.stack(out["c_total"])
    np.savez_compressed(os.path.join(OUT_DIR, f"{stem}.npz"),
                        c_head=ch, c_other=co, c_total=ct, **meta)

    cn2 = (ct ** 2).sum(axis=(1, 2))                                   # (steps,)
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}-summary.npz"),
        c_norm=np.sqrt(cn2),
        head_norm=np.sqrt((ch ** 2).sum(axis=(1, 3))),                 # (steps, H)
        head_norm_attr=np.sqrt((ch ** 2).sum(axis=3)),                 # (steps, 6, H)
        head_share=np.einsum("sahv,sav->sh", ch, ct) / np.maximum(cn2, 1e-30)[:, None],
        other_share=np.einsum("sav,sav->s", co, ct) / np.maximum(cn2, 1e-30),
        **meta)
    print()
    print(f"  wrote {os.path.join(OUT_DIR, stem)}.npz (full, gitignored) "
          f"and -summary.npz (tracked)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free", choices=sorted(ARMS))
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--fixed_readout", action="store_true",
                    help="hold the readout at step 0, giving the representation-side "
                         "attribution alone (3.16 separates the two channels)")
    a = ap.parse_args()
    analyze(a.arm, a.pretrain_step, a.seed, a.condition, a.checkpoint_dir, a.total_steps,
            a.fixed_readout)


if __name__ == "__main__":
    main()
