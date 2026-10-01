"""Do heads go input-insensitive at the trough? (step 2 follow-up)

For each of the 64 heads, split what it writes into the residual at A's value slots into a
CONSTANT part (the mean across A's prompts) and a VARYING part (the per-prompt remainder) --
the same delta/eps split used on the residual itself, applied per head.

  A head doing normal input-dependent work     -> large varying part
  A head writing something close to constant   -> small varying part, large constant

THE PREDICTION BEING TESTED. If injection made heads partly input-insensitive and the constant
they write is the push, the varying part should DROP at the trough and RETURN by step 200. If
the varying part is flat while the OV pieces rotate 60-72 degrees (which they do), the
input-insensitivity story is wrong and the rotation is the whole thing.

NO MODEL CHANGE, AND AN EXACT FALSIFIER. With the MLP disabled the block is exactly
`x_out = x_in + Attn(LN(x_in))`, and BOTH x_in and x_out are already sown, so the block's total
attention write is `x_out - x_in` with nothing reimplemented. The per-head split is rebuilt from
the sown attention pattern plus the value/out kernels, and then CHECKED against that difference:

    sum_h (attn_h @ v_h) @ W_out[h*D:(h+1)*D] + b_out  ==  x_out - x_in

Reimplementing LayerNorm is exactly where a silent error would enter (RoPE is not a risk here:
it touches q and k, never v), so the check is the point, not a formality. It is asserted every
batch, and the worst error is printed.

MEMORY. Storing every head's write for every prompt is 3.2GB, so the split is accumulated in one
pass as sum(w) and sum(||w||^2), giving var = mean||w||^2 - ||mean w||^2 exactly.

Run:
  uv run python -m src.experiments.analyze_head_write_variance --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, init_state, make_optimizer,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, _cfg_for
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

jax.config.update("jax_default_matmul_precision", "highest")

OUT_DIR = "head_write_variance"
BATCH_SIZE = 16
LN_EPS = 1e-6          # flax nn.LayerNorm default


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def layernorm(x, scale, bias):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + LN_EPS) * scale + bias


def head_writes(params, aux, j, n_heads):
    """(per-head write, total attention write, out bias) for block j."""
    it = aux["intermediates"]
    x_in = np.asarray(_get(it, "input_layer", "__call__")[0] if j == 0
                      else _get(it, "backbone", f"block_{j-1}", "__call__")[0])
    x_out = np.asarray(_get(it, "backbone", f"block_{j}", "__call__")[0])
    attn = np.asarray(_get(it, "backbone", f"block_{j}", "CausalSelfAttention_0",
                           "attn_weights")[0])                       # (B,H,Lq,Lk)

    p = params["backbone"][f"block_{j}"]
    y = layernorm(x_in, np.asarray(p["LayerNorm_0"]["scale"]),
                  np.asarray(p["LayerNorm_0"]["bias"]))
    a = p["CausalSelfAttention_0"]
    Wv, Wo = np.asarray(a["value"]["kernel"]), np.asarray(a["out"]["kernel"])
    bo = np.asarray(a["out"]["bias"])

    B, L, E = y.shape
    D = E // n_heads
    v = (y @ Wv).reshape(B, L, n_heads, D)
    ctx = np.einsum("bhqk,bkhd->bqhd", attn, v)
    # head h occupies rows [h*D:(h+1)*D] of W_out, matching the reshape in backbone.py
    per_head = np.einsum("bqhd,hde->bqhe", ctx, Wo.reshape(n_heads, D, E))
    return per_head, x_out - x_in, bo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--steps", default="0,50,200")
    ap.add_argument("--n_items", type=int, default=512)
    a = ap.parse_args()
    if a.arm != "mlp_free":
        raise SystemExit("mlp_free only")

    spec = dict(ARMS[a.arm])
    cfg = _cfg_for(a.arm, a.pretrain_step, a.seed, a.condition, a.total_steps)
    pop, _, data_cfg, _, data_a, _b, _c, _d = build(cfg, cfg.inject.max_eval_people)
    model = create_mlpfree_model(cfg.model, data_cfg)
    n_layers, n_heads = cfg.model.num_layers, cfg.model.num_heads

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
        pretrain_key=mlpfree_pretrain_key(cfg, data_cfg), pretrain_step=a.pretrain_step,
        seed=cfg.seed, **spec["key_extra"])
    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_opt, cfg.inject.total_steps), cfg.seed + 20)

    n = min(a.n_items, data_a.eval_inputs.shape[0])
    m, t = np.asarray(data_a.eval_mask)[:n], np.asarray(data_a.eval_targets)[:n]
    cols = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int64)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(m, t, k)

    steps = [int(s) for s in a.steps.split(",")]
    E = cfg.model.model_dim
    out = {}
    print(f"n={n} blocks={n_layers} heads={n_heads} dim={E} steps={steps}")

    for s in steps:
        path = os.path.join(a.checkpoint_dir,
                            f"{spec['prefix']}-p{a.pretrain_step}-{key}-step{s:04d}.msgpack")
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint missing: {path}")
        params = load_params(path, template.params)

        ssum = np.zeros((n_layers, n_heads, NUM_ATTRIBUTES, E), dtype=np.float64)
        sq = np.zeros((n_layers, n_heads, NUM_ATTRIBUTES), dtype=np.float64)
        worst = 0.0
        for start in range(0, n, BATCH_SIZE):
            sl = slice(start, min(start + BATCH_SIZE, n))
            c = cols[sl]
            bi = np.arange(c.shape[0])[:, None]
            _, aux = model.apply({"params": params}, jnp.array(data_a.eval_inputs[sl]),
                                 deterministic=True, capture_intermediates=True,
                                 mutable=["intermediates"])
            for j in range(n_layers):
                per_head, total, bo = head_writes(params, aux, j, n_heads)
                # THE FALSIFIER: the per-head split must rebuild the block's own write.
                err = np.abs(per_head.sum(2) + bo - total).max()
                worst = max(worst, float(err))
                if err > 2e-3:
                    raise AssertionError(
                        f"step {s} block {j}: per-head writes rebuild the block write to only "
                        f"{err:.3e}; the LayerNorm or head-slice reconstruction is wrong")
                w = per_head[bi, c]                      # (b, A, H, E)
                ssum[j] += w.transpose(2, 1, 0, 3).sum(2)
                sq[j] += (w ** 2).sum(-1).transpose(2, 1, 0).sum(2)
        mean = ssum / n
        const = np.linalg.norm(mean, axis=-1)                         # (L,H,A)
        vary = np.sqrt(np.maximum(sq / n - (mean ** 2).sum(-1), 0.0))  # (L,H,A)
        out[s] = (const, vary)
        print(f"\nstep {s}: reconstruction max error {worst:.2e} (per-head sum == block write)")
        print(f"  {'block':>6} {'const':>8} {'vary':>8} {'vary/const':>11} | per-head vary")
        for j in range(n_layers):
            print(f"  {j:>6} {const[j].mean():>8.3f} {vary[j].mean():>8.3f} "
                  f"{vary[j].mean()/max(const[j].mean(),1e-9):>11.3f} | "
                  + " ".join(f"{v:>5.2f}" for v in vary[j].mean(-1)))

    if len(steps) >= 3:
        s0, st, sp = steps[0], steps[1], steps[2]
        print(f"\nVARYING PART relative to step {s0} -- the prediction is a dip at {st} "
              f"and a return by {sp}")
        print(f"  {'block':>6} " + " ".join(f"{'h'+str(h):>7}" for h in range(n_heads))
              + f" | {'block mean':>10}")
        for tag, s in ((f"{st}/{s0}", st), (f"{sp}/{s0}", sp)):
            print(f"  --- {tag}")
            for j in range(n_layers):
                r = out[s][1][j].mean(-1) / np.maximum(out[s0][1][j].mean(-1), 1e-9)
                print(f"  {j:>6} " + " ".join(f"{v:>7.3f}" for v in r)
                      + f" | {r.mean():>10.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}-p{a.pretrain_step}-{a.condition}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(steps),
        const=np.stack([out[s][0] for s in steps]).astype(np.float32),
        vary=np.stack([out[s][1] for s in steps]).astype(np.float32),
        n_items=n, seed=a.seed, pretrain_step=a.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
