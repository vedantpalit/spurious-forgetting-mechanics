"""Is A's residual displacement common across individuals? (measurement A)

Descriptive, forward passes on existing MLP-free p16000 checkpoints, nothing trained.

WHY. Measurement 1 killed the readout-side account: A's unembedding rows move 0.51% of their
norm at the trough while A's residual moves 120% of its own, A's rows are the MOST stable in the
matrix at that step (0.66x the non-value rate), and the displacement is monotone straight
through the recovery. So the shift is representation-side.

THE HYPOTHESIS THIS TESTS. The shift is blind to the input -- individual, template and position
all fail to predict it -- while its effect is graded by a property of the target, the value
token's cross-attribute sharing count. The model does not see the target at inference, so a
uniform cause is producing a token-graded effect. That is consistent only if the displacement is
COMMON across A's residuals and value tokens differ in how exposed their readout directions are
to it: a common `delta` changes value v's logit by <delta, u_v>.

    dh_a = delta + eps_a      delta = mean over individuals, eps_a = per-individual remainder

TWO CONFOUNDS DESIGNED OUT.

  * delta is computed PER ATTRIBUTE, not pooled. A's residuals at the value slot differ
    systematically by attribute, so a pooled mean would mix six different contexts -- inflating
    ||delta|| where they happen to align and cancelling it where they do not. Individual-blindness
    was established within attribute, so the decomposition has to be within attribute too. The
    pooled number is reported alongside, as a contrast rather than the readout.

  * The split is taken on the POST-LayerNorm readout state, because that is the vector the
    unembedding rows are dotted with and therefore the one whose common component would produce
    a logit shift. LayerNorm is nonlinear, so the displacement of LN(h) is not LN of the
    displacement of h; taking the split on the pre-LN residual would measure a different object.
    Per-layer pre-LN residuals are reported too, for the depth profile only.

WHAT DECIDES. ||delta|| against rms||eps_a||: what fraction of the displacement is common. And
whether ||delta|| turns over during the recovery window while the individual part does not --
the readout rows showed no reversal at all, so a reversing common component would be the first
quantity here that behaves like the accuracy curve, measured on the correct side of the product.

PRE-REGISTERED FAILURE MODE. ||delta|| small relative to rms||eps|| means the shift is not
common, and the individual-blind result needs re-examining rather than a variant of this test.

Run:
  uv run python -m src.experiments.analyze_residual_split --seed 0
"""
import argparse
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, first_token_positions, get_partition, init_state, make_optimizer,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, SCHEDULES, _cfg_for
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "residual_split"
BATCH_SIZE = 64


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def states(model, params, ds, cols, n_layers):
    """(readout, layers) at the value slots.

    readout (N, A, E)            -- post-final-LayerNorm, the vector u_v is dotted with
    layers  (N, A, n_layers+1, E) -- pre-LN residual after each block, entry 0 = embeddings
    """
    n = ds.eval_inputs.shape[0]
    E = None
    read, lay = None, None
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        c = cols[sl]
        bi = np.arange(c.shape[0])[:, None]
        _, aux = model.apply({"params": params}, jnp.array(ds.eval_inputs[sl]),
                            deterministic=True, capture_intermediates=True,
                            mutable=["intermediates"])
        it = aux["intermediates"]
        r = np.asarray(_get(it, "backbone", "LayerNorm_0", "__call__")[0])[bi, c]
        blocks = [np.asarray(_get(it, "input_layer", "__call__")[0])[bi, c]]
        for i in range(n_layers):
            blocks.append(
                np.asarray(_get(it, "backbone", f"block_{i}", "__call__")[0])[bi, c])
        b = np.stack(blocks, axis=2)
        if read is None:
            E = r.shape[-1]
            read = np.zeros((n, NUM_ATTRIBUTES, E), dtype=np.float32)
            lay = np.zeros((n, NUM_ATTRIBUTES, n_layers + 1, E), dtype=np.float32)
        read[sl], lay[sl] = r, b
    return read, lay


def split(dh):
    """(delta per attribute, rms||eps||) from dh of shape (N, A, E) -- or (N, A, L, E)."""
    delta = dh.mean(axis=0)                       # (A, ...) common part, per attribute
    eps = dh - delta[None]
    return delta, np.sqrt((eps ** 2).sum(-1)).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--fig_checkpoints", action="store_true",
                    help="the FIG_SCHEDULE run of mlpfree_dose_injection (t1200)")
    a = ap.parse_args()
    if a.arm != "mlp_free":
        raise SystemExit("mlp_free only")

    spec = dict(ARMS[a.arm])
    cfg = _cfg_for(a.arm, a.pretrain_step, a.seed, a.condition, a.total_steps)
    if a.fig_checkpoints:
        from src.experiments.mlpfree_dose_injection import FIG_SCHEDULE
        cfg = replace(cfg, inject=replace(cfg.inject, checkpoint_steps=FIG_SCHEDULE))
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

    cols = np.zeros((data_a.eval_inputs.shape[0], NUM_ATTRIBUTES), dtype=np.int64)
    correct = np.zeros_like(cols, dtype=np.int32)
    m, t = np.asarray(data_a.eval_mask), np.asarray(data_a.eval_targets)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], correct[:, k] = first_token_positions(m, t, k)

    steps = [int(s) for s in cfg.inject.checkpoint_steps.split(",")]

    def path_for(s):
        return os.path.join(a.checkpoint_dir,
                            f"{spec['prefix']}-p{a.pretrain_step}-{key}-step{s:04d}.msgpack")

    if not os.path.exists(path_for(0)):
        raise SystemExit(f"step-0 checkpoint missing, and it is the baseline: {path_for(0)}")
    r0, l0 = states(model, load_params(path_for(0), template.params), data_a, cols, n_layers)
    print(f"N={r0.shape[0]} attrs={NUM_ATTRIBUTES} dim={r0.shape[-1]} layers={n_layers}")
    print(f"  mean ||readout state|| at step 0: {np.linalg.norm(r0, axis=-1).mean():.3f}")
    print()
    print("  READOUT STATE (post-LayerNorm), split per attribute")
    print(f"  {'step':>5} | {'||delta||':>9} {'rms||eps||':>10} {'ratio':>6} | "
          f"{'pooled ||d||':>12} | per-attribute ||delta||")

    out = {k: [] for k in ("steps", "delta", "eps", "delta_layer", "eps_layer", "pooled")}
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING")
            continue
        r, l = states(model, load_params(p, template.params), data_a, cols, n_layers)
        d, e = split(r - r0)                                   # (A, E), (A,)
        dl, el = split(l - l0)                                 # (A, L+1, E), (A, L+1)
        pooled = np.linalg.norm((r - r0).reshape(-1, r.shape[-1]).mean(0))
        dn = np.linalg.norm(d, axis=-1)
        out["steps"].append(s)
        out["delta"].append(d.astype(np.float32))
        out["eps"].append(e.astype(np.float32))
        out["delta_layer"].append(np.linalg.norm(dl, axis=-1).astype(np.float32))
        out["eps_layer"].append(el.astype(np.float32))
        out["pooled"].append(float(pooled))
        print(f"  {s:>5} | {dn.mean():>9.3f} {e.mean():>10.3f} {dn.mean()/max(e.mean(),1e-9):>6.2f} "
              f"| {pooled:>12.3f} | " + " ".join(f"{v:>6.2f}" for v in dn))

    print()
    print("  PER LAYER (pre-LayerNorm residual), ||delta|| / rms||eps||, attribute-averaged")
    dl = np.stack(out["delta_layer"]); el = np.stack(out["eps_layer"])
    print(f"  {'step':>5} " + " ".join(f'{"L"+str(i):>7}' for i in range(n_layers + 1)))
    for i, s in enumerate(out["steps"]):
        print(f"  {s:>5} " + " ".join(
            f"{dl[i,:,j].mean()/max(el[i,:,j].mean(),1e-9):>7.3f}" for j in range(n_layers + 1)))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}{'-fig' if a.fig_checkpoints else ''}-p{a.pretrain_step}-{a.condition}-t{a.total_steps}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(out["steps"]), delta=np.stack(out["delta"]),
        eps_rms=np.stack(out["eps"]), delta_layer_norm=dl, eps_layer_rms=el,
        pooled_delta_norm=np.array(out["pooled"]),
        readout0_norm=np.linalg.norm(r0, axis=-1).astype(np.float32),
        correct_id=correct, seed=a.seed, pretrain_step=a.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz "
          f"(delta per attribute per step, so the alignment test needs no re-run)")


if __name__ == "__main__":
    main()
