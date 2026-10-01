"""Which weights produce delta, and which produce the part that recovers? (steps 1-2)

Splice named parameters from the step-0 checkpoint into the patched checkpoint, run forward, and
remeasure delta. Nothing is trained, so B's acquisition is untouched by construction -- the
confound that makes freeze arms unreadable. Unlike `analyze_pattern_patch`, this needs no
interception at all: a spliced params tree is just another params tree.

  --patch qk        query and key kernels, per block
  --patch ov        value and out projections, per block
  --patch ln        the block's pre-attention LayerNorm, per block
  --patch embed     the input embedding                     (not per block)
  --patch finalln   the final LayerNorm before the head     (not per block)
  --patch head      the unembedding -- a NULL CONTROL, see below

EXHAUSTIVE, AND ONE GROUP IS A NULL CONTROL. The six groups cover all 61 parameter leaves, so
the decomposition can be stated as complete rather than as "OV is 84% and we did not look at the
rest". But `head` sits DOWNSTREAM of the measurement: delta is read at the final LayerNorm's
output and the head is applied after it, so patching the head cannot change delta and must give
exactly zero. It is run as a check that the readout is tapped where this code assumes, not as a
component -- and its positive control is inverted to match, requiring that zeroing it does NOT
move the readout. The groups that can affect delta are qk, ov, ln, embed and finalln, 59 leaves.

STEP 1 IS NOT A CHECK, IT IS A DECOMPOSITION. Restoring QK weights is a DIFFERENT intervention
from restoring the attention pattern, and the difference is the point. Pattern patching imposes
`softmax(f(x_0; W_QK^0))`; the QK weight patch gives `softmax(f(x_trough; W_QK^0))`. The two agree
only if the block's INPUT is unchanged, and it is not. So comparing them decomposes block 7's
pattern change into two sources:

    QK patch reproduces the pattern patch  ->  the pattern change came from QK weights moving
    QK patch does nothing                  ->  it came from the block's input drifting

Either is a result. Recording it as a confirmation would throw the informative half away.

REPORTING. The removed piece is the DIFFERENCE VECTOR `delta_none - delta_patched`, never the
shrinkage of the total. Near-orthogonal pieces barely shorten the total: block 7's pattern-produced
component is 46% of ||delta|| but removing it drops the norm only 17.7%, and reporting the ratio
understated it by more than half. The same error is guarded a second way -- the saved vectors let
a piece that ROTATES between checkpoints rather than shrinking be seen, which a norm difference
near zero would hide entirely.

VERIFICATION, EVERY RUN (`verify`). A small measured effect and a splice that landed nowhere
produce the same number, and the QK result has exactly that shape, so it is established by
assertion rather than by inspection: identical key sets, shapes and dtypes between the two
checkpoints; after splicing, the intended leaves BIT-IDENTICAL to step 0 and every other leaf
BIT-IDENTICAL to the trough; the set of leaves actually changed equal to the intended leaves
that moved between checkpoints. Then two positive controls -- zeroing the group through the
same code path must move the readout substantially (a self-splice is a no-op whether or not the
path is right, so it cannot show this), and splicing EVERY leaf must reconstruct the step-0
model and give delta exactly zero. Per-leaf weight movement is printed so that "patched, but
these weights barely moved" stays distinguishable from "patching did nothing".
`tests/test_weight_patch_splice.py` pins that the verification itself can fail.

PER ATTRIBUTE, ALWAYS. Attributes differ from 37% to 63% in how much of their delta is
pattern-produced, while the survivor falls 49-61% in all six. Whether the per-block OV
contributions are likewise uniform across attributes is a stronger statement than any pooled
number, and it is free once the per-attribute deltas are saved.

Run (one invocation per patch group, seed and checkpoint):
  uv run python -m src.experiments.analyze_weight_patch --patch ov --patch_step 50 --seed 0
"""
import argparse
import os

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.analyze_arms import ARM_SPECS, load_arm
from src.experiments.knowledge_injection import first_token_positions, get_partition
from src.train import eval_forward

OUT_DIR = "weight_patch"
BATCH_SIZE = 32

# The per-block groups (qk, ov) are only exact in the MLP-free stack, where the residual is
# embedding plus attention outputs and nothing else. The global groups are the same leaves in
# every arm, so they run anywhere -- which is what the ballast contrast needs for `head`.
MLP_FREE_ONLY = {"qk", "ov"}

# Verified against a constructed params tree, not assumed: query/key/value are use_bias=False
# so they have no bias leaf; `out` has one.
GROUPS = {
    "qk": [("CausalSelfAttention_0", "query", "kernel"),
           ("CausalSelfAttention_0", "key", "kernel")],
    "ov": [("CausalSelfAttention_0", "value", "kernel"),
           ("CausalSelfAttention_0", "out", "kernel"),
           ("CausalSelfAttention_0", "out", "bias")],
    "ln": [("LayerNorm_0", "scale"), ("LayerNorm_0", "bias")],
}

# Groups whose parameters do not live under a block, so there is no per-block or cumulative
# variant -- they are patched whole or not at all. Together with GROUPS these cover all 61
# leaves, so the decomposition can be stated as exhaustive rather than as "OV is 84% and we
# did not look at the rest".
GLOBAL_GROUPS = {
    "embed":   [("input_layer", "Embed_0", "embedding")],
    "finalln": [("backbone", "LayerNorm_0", "scale"), ("backbone", "LayerNorm_0", "bias")],
    "head":    [("head", "Dense_0", "kernel"), ("head", "Dense_0", "bias")],
}

# `head` is DOWNSTREAM of the measurement. delta is read at the final LayerNorm's output
# (see `readout`), and the head is applied after it, so patching the head cannot change delta
# and its result must be exactly zero. That makes it a null control rather than a component:
# a nonzero answer would mean the readout is not being tapped where this code assumes. Its
# positive control is inverted accordingly -- zeroing it must NOT move the readout.
DOWNSTREAM = {"head"}


def all_groups():
    return sorted(GROUPS) + sorted(GLOBAL_GROUPS)


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def group_keys(group, blocks):
    """The flat parameter paths `group` names. `blocks` is ignored for a global group."""
    if group in GLOBAL_GROUPS:
        return list(GLOBAL_GROUPS[group])
    return [("backbone", f"block_{j}") + leaf
            for j in sorted(blocks) for leaf in GROUPS[group]]


def splice_keys(pt, p0, keys):
    """`pt` with exactly `keys` taken from `p0`. Raises if a path matches nothing.

    A path that matches nothing is THE failure mode here: the splice would be a silent no-op
    and a small measured effect would mean "nothing was patched" rather than "patching did
    nothing". Those are opposite conclusions, so this raises rather than warns.
    """
    ft, f0 = flatten_dict(pt), flatten_dict(p0)
    for k in keys:
        if k not in ft:
            near = sorted({"/".join(x[2:]) for x in ft if x[:2] == k[:2]})
            raise KeyError(f"{'/'.join(k)} is not in the params tree, so the splice would be "
                           f"a silent no-op. Under {'/'.join(k[:2])} there is: {near}")
        ft[k] = f0[k]
    return unflatten_dict(ft)


def splice(pt, p0, group, blocks):
    """`pt` with `group`'s leaves for every block in `blocks` taken from `p0`."""
    keys = group_keys(group, blocks)
    return splice_keys(pt, p0, keys), len(keys)


def _flat(t):
    return {k: np.asarray(v) for k, v in flatten_dict(t).items()}


def structural_check(p0, pt):
    """Identical trees, or fail loudly. A renamed or missing path must not pass silently."""
    f0, ft = _flat(p0), _flat(pt)
    if set(f0) != set(ft):
        raise SystemExit("the two checkpoints have different parameter trees: "
                         + str(sorted("/".join(k) for k in set(f0) ^ set(ft))))
    bad = [("/".join(k), f0[k].shape, ft[k].shape, f0[k].dtype, ft[k].dtype)
           for k in f0 if f0[k].shape != ft[k].shape or f0[k].dtype != ft[k].dtype]
    if bad:
        raise SystemExit(f"shape/dtype mismatch between checkpoints: {bad}")
    return f0, ft


def verify(model, p0, pt, group, n_layers, ds, cols, n_ver, r0):
    """Prove the splice patches exactly what it claims, and that patching CAN move the output.

    Value equality is asserted on both sides -- spliced leaves bit-identical to step 0, every
    other leaf bit-identical to the trough -- because a shape check cannot distinguish a
    correct splice from one that landed nowhere. Then two positive controls, in the same
    spirit as the pattern harness's self-patch check:

      1. splicing a ZEROED version of the group through the same code path must move the
         readout substantially. This is what proves the paths reach the forward pass; a
         self-splice (identical values) is a no-op whether or not the path is right.
      2. splicing EVERY leaf from step 0 must reconstruct the step-0 model exactly, so delta
         is zero by definition. A nonzero result means the splice is incomplete somewhere.
    """
    f0, ft = structural_check(p0, pt)
    intended = set(group_keys(group, range(n_layers)))
    if not intended:
        raise SystemExit(f"group {group!r} names no parameters")

    hyb = _flat(splice_keys(pt, p0, sorted(intended)))
    wrong_side = [("/".join(k), "should equal step0" if k in intended else "should equal trough")
                  for k in ft
                  if not np.array_equal(hyb[k], (f0 if k in intended else ft)[k])]
    if wrong_side:
        raise SystemExit(f"spliced tree has {len(wrong_side)} leaves on the wrong side: "
                         f"{wrong_side[:8]}")

    # The set of leaves the splice actually CHANGED must be exactly the intended leaves that
    # differ between the checkpoints. Intended leaves that are already bit-identical splice to
    # a no-op legitimately -- that is a fact about the weights, not a bug, and it is reported
    # below rather than hidden.
    changed = {k for k in ft if not np.array_equal(hyb[k], ft[k])}
    expected = {k for k in intended if not np.array_equal(f0[k], ft[k])}
    if changed != expected:
        raise SystemExit(f"changed set != intended-and-moved set; symmetric difference: "
                         f"{sorted('/'.join(k) for k in changed ^ expected)}")

    print(f"  [verify] tree: {len(ft)} leaves, identical key sets, shapes and dtypes")
    print(f"  [verify] splice: {len(intended)} intended leaves, all bit-identical to step 0; "
          f"{len(ft) - len(intended)} others all bit-identical to the trough")
    print(f"  [verify] leaves the splice changed: {len(changed)} "
          f"(= intended leaves that moved between checkpoints)")

    # Normalised by max(||W(0)||, ||W(t)||), not by ||W(0)||: a bias that pretrained to near
    # zero would otherwise divide by ~0 and report a meaningless 1e+11. This form is bounded
    # by 1 and reads as "what fraction of the larger weight the movement is".
    print(f"  [verify] weight movement ||W(t)-W(0)|| / max(||W(0)||,||W(t)||), patched leaves:")
    mv = lambda k: (np.linalg.norm(ft[k] - f0[k])
                    / max(np.linalg.norm(f0[k]), np.linalg.norm(ft[k]), 1e-12))
    if group in GLOBAL_GROUPS:
        for k in GLOBAL_GROUPS[group]:
            print(f"    {'/'.join(k):>34}  {mv(k):.2e}")
    else:
        for j in range(n_layers):
            parts = [f"{'/'.join(leaf[1:]):>12}={mv(('backbone', f'block_{j}') + leaf):.2e}"
                     for leaf in GROUPS[group]]
            print(f"    block_{j}  " + "  ".join(parts))
    frozen = sorted("/".join(k) for k in intended if np.array_equal(f0[k], ft[k]))
    if frozen:
        print(f"  [verify] NOTE {len(frozen)} intended leaves did not move at all between "
              f"checkpoints, so they splice to a no-op legitimately: {frozen}")

    rt = readout(model, pt, ds, cols, n_ver)
    same, _ = splice(pt, pt, group, set(range(n_layers)))
    if not np.allclose(readout(model, same, ds, cols, n_ver), rt, atol=1e-6):
        raise SystemExit("self-splice changed the output; the splice is not landing as assumed")
    print("  [verify] self-splice (trough into trough) is a no-op")

    # Positive control 1 -- a deliberately destructive splice through the same code path.
    fz = dict(flatten_dict(pt))
    for k in intended:
        fz[k] = np.zeros_like(np.asarray(fz[k]))
    rz = readout(model, splice_keys(pt, unflatten_dict(fz), sorted(intended)), ds, cols, n_ver)
    moved = np.linalg.norm(rz - rt, axis=-1).mean() / np.linalg.norm(rt, axis=-1).mean()
    if group in DOWNSTREAM:
        # Inverted on purpose: this group sits AFTER the readout tap, so zeroing it must not
        # move the readout. A nonzero answer means delta is not being read where assumed.
        if moved > 1e-6:
            raise SystemExit(f"zeroing every {group} leaf moved the readout {moved:.2e}, but "
                             f"{group} is downstream of the readout tap and cannot affect it")
        print(f"  [verify] {group} is downstream of the readout tap: zeroing all "
              f"{len(intended)} of its leaves moves the readout {moved:.1e}, as it must")
    elif moved < 0.10:
        raise SystemExit(f"zeroing every {group} leaf moved the readout only {moved:.1%}; the "
                         f"splice is probably not reaching the forward pass")
    else:
        print(f"  [verify] zeroing all {len(intended)} {group} leaves moves the readout "
              f"{moved:.1%} -- the patched paths reach the forward pass")

    # Positive control 2 -- reconstruct step 0 leaf by leaf; delta must be exactly zero.
    all_keys = sorted(flatten_dict(pt))
    pall = splice_keys(pt, p0, all_keys)
    fa = _flat(pall)
    if any(not np.array_equal(fa[k], f0[k]) for k in all_keys):
        raise SystemExit("full-tree splice did not reproduce the step-0 tree")
    dall = np.abs(readout(model, pall, ds, cols, n_ver) - r0).max()
    if dall > 1e-5:
        raise SystemExit(f"full-tree splice gives max|delta| = {dall:.3e}, not 0; the splice "
                         f"is incomplete somewhere")
    print(f"  [verify] full-tree splice of all {len(all_keys)} leaves gives "
          f"max|delta| = {dall:.2e} (zero by construction)")


def accuracy_stats(model, params, ds, cols, n, own_ids):
    """A's first-token accuracy per attribute, (A,) over the full vocabulary and (A,) with
    the argmax restricted to A's own value half.

    The second is the within-half read: a per-class bias that lifts the other half whole
    cannot change it, so the gap between the two is the between-half suppression and the
    fall in the second is damage the bias account does not cover.
    """
    t = np.asarray(ds.eval_targets)[:n]
    hit_full = np.zeros(NUM_ATTRIBUTES)
    hit_own = np.zeros(NUM_ATTRIBUTES)
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        rows = np.arange(logits.shape[0])
        for k in range(NUM_ATTRIBUTES):
            c = cols[sl, k]
            z = logits[rows, c, :]
            correct = t[sl][rows, c]
            hit_full[k] += (z.argmax(-1) == correct).sum()
            own = np.asarray(own_ids[k])
            hit_own[k] += (own[z[:, own].argmax(-1)] == correct).sum()
    return hit_full / n, hit_own / n


def readout(model, params, ds, cols, n):
    """Mean post-LayerNorm readout state at the value slots, (A, E)."""
    acc = None
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        c = cols[sl]
        bi = np.arange(c.shape[0])[:, None]
        _, aux = model.apply({"params": params}, jnp.array(ds.eval_inputs[sl]),
                             deterministic=True, capture_intermediates=True,
                             mutable=["intermediates"])
        # The backbone's return value: the final LayerNorm's output where there is one, the
        # last block's residual in the no_final_norm arm (which has no LayerNorm_0 to tap).
        r = np.asarray(_get(aux["intermediates"], "backbone", "__call__")[0])[bi, c]
        acc = r.sum(0) if acc is None else acc + r.sum(0)
    return acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="ov", choices=all_groups())
    ap.add_argument("--arm", default="mlp_free", choices=sorted(ARM_SPECS))
    ap.add_argument("--pretrain_step", type=int, default=None,
                    help="defaults to the arm's own (16000 for the 8-layer arms, 8000 for 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--patch_step", type=int, default=50)
    ap.add_argument("--patch_steps", default=None,
                    help="comma list; with --score accuracy, every step is scored in one job")
    ap.add_argument("--score", default="readout", choices=("readout", "accuracy"),
                    help="readout: the delta decomposition (the original measurement). "
                         "accuracy: A's accuracy under (body_0, group_t) and (body_t, group_0), "
                         "which is what the ballast contrast needs -- the head sits after the "
                         "readout tap, so on delta it is a structural zero, but on accuracy "
                         "it is the readout's share of the damage")
    ap.add_argument("--n_items", type=int, default=512)
    ap.add_argument("--n_verify", type=int, default=64,
                    help="items used for the splice verification forward passes; the checks "
                         "are about parameter identity and whether a patch can move the "
                         "output at all, neither of which needs the full eval set")
    ap.add_argument("--verify_only", action="store_true",
                    help="run the splice checks and exit, without the measurement")
    a = ap.parse_args()
    if a.patch in MLP_FREE_ONLY and a.arm != "mlp_free":
        raise SystemExit(f"--patch {a.patch} is exact only in the MLP-free stack; use --arm "
                         f"mlp_free. The global groups (embed, finalln, head) run on any arm.")

    ctx = load_arm(a.arm, a.seed, a.condition, a.total_steps, a.checkpoint_dir, a.pretrain_step)
    model, n_layers, data_a, pop, cfg = ctx.model, ctx.n_layers, ctx.data_a, ctx.pop, ctx.cfg
    path_for = ctx.path_for
    steps = ([int(s) for s in a.patch_steps.split(",")] if a.patch_steps else [a.patch_step])
    if a.score == "readout" and len(steps) != 1:
        raise SystemExit("--score readout takes one --patch_step; use --patch_steps with "
                         "--score accuracy")

    for s in [0] + steps:
        if not os.path.exists(path_for(s)):
            raise SystemExit(f"checkpoint missing: {path_for(s)}")
    p0 = ctx.load(0)
    pt = ctx.load(steps[0])

    n = min(a.n_items, data_a.eval_inputs.shape[0])
    m, t = np.asarray(data_a.eval_mask)[:n], np.asarray(data_a.eval_targets)[:n]
    cols = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int64)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(m, t, k)

    r0 = readout(model, p0, data_a, cols, n)
    if a.score == "accuracy":
        # Verify the splice once, on the first requested step, then score every step.
        nv = min(a.n_verify, n)
        verify(model, p0, pt, a.patch, n_layers, data_a, cols, nv,
               readout(model, p0, data_a, cols, nv))
        if a.verify_only:
            print("\n  --verify_only: checks passed, no measurement run")
            return
        halves = get_partition(cfg, pop)
        own_ids = [np.asarray(pop.attr_first_token_ids[k][halves[k][0]])
                   for k in range(NUM_ATTRIBUTES)]
        blocks = {0} if a.patch in GLOBAL_GROUPS else set(range(n_layers))
        acc0, own0 = accuracy_stats(model, p0, data_a, cols, n, own_ids)
        print(f"\n  score=accuracy patch={a.patch} arm={a.arm} seed={a.seed} n={n}  "
              f"A(0): full {acc0.mean():.4f}  own-half {own0.mean():.4f}")
        print(f"\n  {'step':>5} | {'A(t)':>7} {'body_0+grp_t':>13} {'body_t+grp_0':>13} | "
              f"{'R_grp':>6} {'R_body':>7} | {'own(t)':>7} {'own b0+g_t':>10} {'own bt+g0':>10}")
        rows = []
        for s in steps:
            ps = pt if s == steps[0] else ctx.load(s)
            # splice(dst, src, ...) is dst with the group's leaves taken from src.
            grp_t = splice(p0, ps, a.patch, blocks)[0]     # body at 0, group at t
            grp_0 = splice(ps, p0, a.patch, blocks)[0]     # body at t, group at 0
            acc_t, own_t = accuracy_stats(model, ps, data_a, cols, n, own_ids)
            acc_g, own_g = accuracy_stats(model, grp_t, data_a, cols, n, own_ids)
            acc_b, own_b = accuracy_stats(model, grp_0, data_a, cols, n, own_ids)
            dmg = acc0.mean() - acc_t.mean()
            r_grp = (acc0.mean() - acc_g.mean()) / dmg if dmg > 1e-9 else float("nan")
            r_body = (acc0.mean() - acc_b.mean()) / dmg if dmg > 1e-9 else float("nan")
            rows.append((s, acc_t, acc_g, acc_b, own_t, own_g, own_b, r_grp, r_body))
            print(f"  {s:>5} | {acc_t.mean():>7.4f} {acc_g.mean():>13.4f} {acc_b.mean():>13.4f} "
                  f"| {r_grp:>6.3f} {r_body:>7.3f} | {own_t.mean():>7.4f} {own_g.mean():>10.4f} "
                  f"{own_b.mean():>10.4f}")
        os.makedirs(OUT_DIR, exist_ok=True)
        stem = f"{a.arm}-{a.patch}-acc-p{ctx.pretrain_step}-{a.condition}-seed{a.seed}"
        np.savez_compressed(
            os.path.join(OUT_DIR, f"{stem}.npz"),
            steps=np.array([r[0] for r in rows]),
            acc0=acc0, own0=own0,
            acc_t=np.stack([r[1] for r in rows]), acc_group_t=np.stack([r[2] for r in rows]),
            acc_group_0=np.stack([r[3] for r in rows]),
            own_t=np.stack([r[4] for r in rows]), own_group_t=np.stack([r[5] for r in rows]),
            own_group_0=np.stack([r[6] for r in rows]),
            r_group=np.array([r[7] for r in rows]), r_body=np.array([r[8] for r in rows]),
            patch=a.patch, arm=a.arm, n_items=n, seed=a.seed,
            pretrain_step=ctx.pretrain_step, condition=a.condition, key=ctx.key)
        print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")
        return

    if a.patch in GLOBAL_GROUPS:
        # No per-block or cumulative variants: these parameters do not live under a block.
        configs = [("none", set()), ("all", {0})]
        print(f"patch={a.patch} ({len(GLOBAL_GROUPS[a.patch])} leaves, not per-block) "
              f"step={a.patch_step} seed={a.seed} n={n}")
    else:
        configs = [("none", set())]
        configs += [(f"blk{j}", {j}) for j in range(n_layers)]
        configs += [(f"blk{j}-{n_layers-1}", set(range(j, n_layers)))
                    for j in range(n_layers - 1, -1, -1)]
        print(f"patch={a.patch} ({len(GROUPS[a.patch])} leaves/block, "
              f"{len(GROUPS[a.patch]) * n_layers} spliced for all blocks) "
              f"step={a.patch_step} seed={a.seed} n={n}")
    nv = min(a.n_verify, n)
    verify(model, p0, pt, a.patch, n_layers, data_a, cols, nv,
           readout(model, p0, data_a, cols, nv))
    if a.verify_only:
        print("\n  --verify_only: checks passed, no measurement run")
        return

    deltas = {}
    for name, blocks in configs:
        pp, _ = splice(pt, p0, a.patch, blocks) if blocks else (pt, 0)
        deltas[name] = readout(model, pp, data_a, cols, n) - r0

    base = deltas["none"]
    bn = np.linalg.norm(base, axis=-1)
    print(f"\n  unpatched ||delta||: " + " ".join(f"{v:.2f}" for v in bn)
          + f"   mean {bn.mean():.3f}")
    print(f"\n  {'patch':>12} {'||delta||':>10} | {'||removed||':>12} {'as % of base':>13} | "
          f"per-attribute ||removed||")
    for name, _ in configs:
        d = deltas[name]
        rem = base - d
        rn = np.linalg.norm(rem, axis=-1)
        print(f"  {name:>12} {np.linalg.norm(d, axis=-1).mean():>10.3f} | "
              f"{rn.mean():>12.3f} {rn.mean()/bn.mean():>12.1%} | "
              + " ".join(f"{v:>6.2f}" for v in rn))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = (f"{a.arm}-{a.patch}-p{ctx.pretrain_step}-{a.condition}-"
            f"step{a.patch_step}-seed{a.seed}")
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        names=np.array([nm for nm, _ in configs]),
        delta=np.stack([deltas[nm] for nm, _ in configs]).astype(np.float32),
        removed=np.stack([base - deltas[nm] for nm, _ in configs]).astype(np.float32),
        base_norm=bn, patch=a.patch, patch_step=a.patch_step, n_items=n, seed=a.seed,
        pretrain_step=ctx.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
