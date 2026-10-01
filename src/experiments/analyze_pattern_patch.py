"""Is delta produced by attention PATTERN change, or by what attention writes? (patching)

Zucchet et al. (arXiv 2503.21676) locate the extraction circuit -- which selects and relays
information according to the requested attribute type -- in the FINAL attention layer. An earlier
measurement found delta entering overwhelmingly at block 7: 40-50% of the total, 21.7 at the trough
against 6.6 for the next largest. Same location. That invites a reading delta has not been tested
against: that it is not a push against A's values but the attribute-type SELECTION being biased
toward B's region, which would account for delta being individual-blind, entering last, and having
no alignment with B's mean representation (cos -> 0.00 while ||delta|| stays 10.3), since selection
is not similarity.

THE INTERVENTION. At the trough checkpoint, run A's prompts with every weight at its trough value
but block j's attention pattern replaced by the STEP-0 pattern, and remeasure delta.

  delta collapses  -> it is produced by pattern change at that block. Routing, strictly, and the
                      first positive evidence for it in this project.
  delta survives   -> it lives in what attention writes (the value/output path), not where it
                      looks.
  partial          -> the fraction is the result; it is reported, not rounded.

Blocks are patched individually and cumulatively from block 7 downward, so "block 7 alone" and
"block 7 plus what feeds it" are distinguishable.

NO DOSE CONFOUND. Nothing is trained, so B's acquisition is untouched by construction. That is
what the freeze arms could never provide.

HOW THE SUBSTITUTION IS DONE. `nn.softmax` is called exactly once in the whole model
(backbone.py:123, inside CausalSelfAttention), so the call is intercepted at runtime and the
n-th call within one `apply` is block n. This touches none of the model code and -- more
importantly -- requires no reimplementation of LayerNorm or RoPE, which is where a silent error
would otherwise enter. Three things are asserted rather than assumed: the number of intercepted
calls equals the number of blocks, an empty patch set reproduces the unmodified model, and a
patched block's sown `attn_weights` equal the pattern that was injected.

Run:
  uv run python -m src.experiments.analyze_pattern_patch --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

import src.model.backbone as backbone
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, get_partition, init_state, make_optimizer,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "pattern_patch"
BATCH_SIZE = 32


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


class PatternPatch:
    """Replace block j's post-softmax attention pattern, for j in `overrides`.

    Blocks execute in order within one `apply`, and `nn.softmax` is called once per block, so the
    n-th interception is block n. `calls` is checked against the block count by the caller.
    """

    def __init__(self, overrides):
        self.overrides = overrides
        self.calls = 0
        self._orig = backbone.nn.softmax

    def __enter__(self):
        def patched(x, axis=-1, **kw):
            i = self.calls
            self.calls += 1
            if i in self.overrides:
                p = jnp.asarray(self.overrides[i])
                if p.shape != x.shape:
                    raise ValueError(f"block {i}: pattern {p.shape} vs softmax input {x.shape}")
                return p
            return self._orig(x, axis=axis, **kw)
        backbone.nn.softmax = patched
        return self

    def __exit__(self, *a):
        backbone.nn.softmax = self._orig
        return False


def run(model, params, x, overrides, n_layers):
    """(readout at every position, sown attention patterns) with `overrides` applied."""
    with PatternPatch(overrides) as pp:
        _, aux = model.apply({"params": params}, x, deterministic=True,
                             capture_intermediates=True, mutable=["intermediates"])
        if pp.calls != n_layers:
            raise AssertionError(f"intercepted {pp.calls} softmax calls, expected {n_layers}; "
                                 f"the call-order assumption is broken")
    it = aux["intermediates"]
    read = np.asarray(_get(it, "backbone", "LayerNorm_0", "__call__")[0])
    pats = [np.asarray(_get(it, "backbone", f"block_{i}", "CausalSelfAttention_0",
                            "attn_weights")[0]) for i in range(n_layers)]
    return read, pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--patch_step", type=int, default=50, help="checkpoint to patch (the trough)")
    ap.add_argument("--n_items", type=int, default=512,
                    help="A individuals; delta is a mean, so its sampling noise is "
                         "rms||eps||/sqrt(n) = 0.45 at 512 against ||delta|| of 18.8")
    a = ap.parse_args()
    if a.arm != "mlp_free":
        raise SystemExit("mlp_free only")

    spec = dict(ARMS[a.arm])
    cfg = _cfg_for(a.arm, a.pretrain_step, a.seed, a.condition, a.total_steps)
    pop, _, data_cfg, _, data_a, _b, _c, _d = build(cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)
    n_layers = cfg.model.num_layers

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
        pretrain_key=mlpfree_pretrain_key(cfg, data_cfg), pretrain_step=a.pretrain_step,
        seed=cfg.seed, **spec["key_extra"])
    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_opt, cfg.inject.total_steps), cfg.seed + 20)

    def path_for(s):
        return os.path.join(a.checkpoint_dir,
                            f"{spec['prefix']}-p{a.pretrain_step}-{key}-step{s:04d}.msgpack")

    for s in (0, a.patch_step):
        if not os.path.exists(path_for(s)):
            raise SystemExit(f"checkpoint missing: {path_for(s)}")
    p0 = load_params(path_for(0), template.params)
    pt = load_params(path_for(a.patch_step), template.params)

    n = min(a.n_items, data_a.eval_inputs.shape[0])
    m, t = np.asarray(data_a.eval_mask)[:n], np.asarray(data_a.eval_targets)[:n]
    cols = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int64)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(m, t, k)

    configs = [("none", set())]
    configs += [(f"blk{j}", {j}) for j in range(n_layers)]
    configs += [(f"blk{j}-{n_layers-1}", set(range(j, n_layers))) for j in range(n_layers - 1, -1, -1)]

    E = cfg.model.model_dim
    # blk7 and blk7-7 are the same override set; the duplicate is kept deliberately, since the
    # two must produce identical numbers and disagreement would mean the accumulation is wrong.
    acc = {name: np.zeros((NUM_ATTRIBUTES, E), dtype=np.float64) for name, _ in configs}
    acc0 = np.zeros((NUM_ATTRIBUTES, E), dtype=np.float64)
    checked = False
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        x = jnp.array(data_a.eval_inputs[sl])
        c = cols[sl]
        bi = np.arange(c.shape[0])[:, None]
        r0, pats0 = run(model, p0, x, {}, n_layers)
        acc0 += r0[bi, c].sum(0)
        for name, blocks in configs:
            ov = {j: pats0[j] for j in blocks}
            r, pats = run(model, pt, x, ov, n_layers)
            acc[name] += r[bi, c].sum(0)
            if not checked and blocks:
                j = max(blocks)
                if not np.allclose(pats[j], pats0[j], atol=1e-6):
                    raise AssertionError(f"{name}: block {j}'s sown pattern is not the injected "
                                         f"one; the interception is not taking effect")
        if not checked:
            # 1. the interception harness with an empty override set must be a no-op: compare
            #    against an apply with no interceptor installed at all
            _, ref_aux = model.apply({"params": pt}, x, deterministic=True,
                                     capture_intermediates=True, mutable=["intermediates"])
            ref = np.asarray(_get(ref_aux["intermediates"], "backbone", "LayerNorm_0",
                                  "__call__")[0])
            rp, pats_t = run(model, pt, x, {}, n_layers)
            assert np.allclose(ref, rp, atol=1e-6), (
                "the interception harness perturbs the forward pass even with nothing patched")
            # 2. injecting the model's OWN patterns must also be a no-op. This is the check that
            #    the injected array actually reaches the attention computation: an interception
            #    that silently did nothing would pass check 1 too.
            rid, _ = run(model, pt, x, {j: pats_t[j] for j in range(n_layers)}, n_layers)
            assert np.allclose(ref, rid, atol=1e-6), (
                "injecting the model's own patterns changed the output; the substitution is "
                "not landing where it is assumed to")
            print(f"  harness checks passed: {n_layers} softmax calls intercepted, empty patch "
                  f"is a no-op, self-patch is a no-op, and a step-0 patch changes the pattern")
            checked = True

    d0 = acc0 / n
    base = acc["none"] / n - d0
    bn = np.linalg.norm(base, axis=-1)
    print(f"\npatch at step {a.patch_step}, {n} individuals, {n_layers} blocks")
    print(f"  unpatched ||delta|| per attribute: " + " ".join(f"{v:.2f}" for v in bn)
          + f"   mean {bn.mean():.3f}")
    print()
    print(f"  {'patch':>12} {'||delta||':>10} {'frac of base':>13} | per attribute")
    out = {}
    for name, _ in configs:
        d = acc[name] / n - d0
        dn = np.linalg.norm(d, axis=-1)
        out[name] = dn
        print(f"  {name:>12} {dn.mean():>10.3f} {dn.mean()/bn.mean():>13.3f} | "
              + " ".join(f"{v:>6.2f}" for v in dn))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}-p{a.pretrain_step}-{a.condition}-step{a.patch_step}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        names=np.array([nm for nm, _ in configs]),
        delta_norm=np.stack([out[nm] for nm, _ in configs]),
        delta=np.stack([acc[nm] / n - d0 for nm, _ in configs]).astype(np.float32),
        base_norm=bn, n_items=n, patch_step=a.patch_step, seed=a.seed,
        pretrain_step=a.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
