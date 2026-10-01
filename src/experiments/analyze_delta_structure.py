"""What are delta and eps made of, and where do they enter? (measurements 2-4)

Lookups on existing MLP-free p16000 checkpoints. Forward passes only, nothing trained, no
mechanism asserted.

Measurement A established that A's residual displacement splits into a common part delta, which
peaks at the accuracy trough and reverses 37%, and an individual part eps, which grows
monotonically. This characterises both objects before asking where they come from.

  2. IS delta NEW? Project delta onto the principal directions of A's step-0 representations.
     Lying along directions the representations already had variance in means the shift amplifies
     existing structure; orthogonal to all of them means a genuinely new direction is being
     written.

  3. IS eps ACTUALLY INDIVIDUAL? The mean pairwise cosine among the eps_a. Mutually orthogonal
     means genuinely individual. Substantial shared structure means calling it individual is
     wrong and the delta/eps split is not capturing the real decomposition -- which would
     invalidate measurement A's reading rather than refine it. This is the single number the
     whole framing rests on.

     Computed as (||sum of unit eps||^2 - N) / (N(N-1)), which is the exact mean over ordered
     pairs without forming the N x N matrix.

  4. WHERE DOES EACH ENTER? The residual is a sum, so delta at layer L is delta at L-1 plus what
     block L-1 added. Per-layer CONTRIBUTIONS to each, not the cumulative ratio measurement A
     already reported. Same layers for delta and eps means one process with two components and
     the split is descriptive; different layers means two separate things.

Everything is reported per attribute, never pooled: A's residuals at the value slot differ
systematically by attribute, and pooling them cost 31% of the common component in measurement A.

Run:
  uv run python -m src.experiments.analyze_delta_structure --seed 0
"""
import argparse
import os

import jax.numpy as jnp
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.analyze_arms import ARM_SPECS, load_arm
from src.experiments.knowledge_injection import first_token_positions

# ARMS. Originally MLP-free only. The per-layer increments `delta_layer[l+1] - delta_layer[l]`
# are exact in every arm -- each block's output minus its input, summing to the layer-L
# displacement with no remainder -- so the script runs on the standard 4- and 8-layer arms
# too. What changes is the reading: in a standard block the increment is attention PLUS the
# MLP, not attention alone. That is the decomposition the ballast contrast asks for.

OUT_DIR = "delta_structure"
BATCH_SIZE = 64
N_PC = 20


def _get(tree, *path):
    node = tree
    for p in path:
        if p not in node:
            raise KeyError(f"missing {p!r}; available: {sorted(node.keys())}")
        node = node[p]
    return node


def states(model, params, ds, cols, n_layers):
    """(readout (N,A,E), layers (N,A,L+1,E)) at the value slots."""
    n = ds.eval_inputs.shape[0]
    read = lay = None
    for start in range(0, n, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n))
        c = cols[sl]
        bi = np.arange(c.shape[0])[:, None]
        _, aux = model.apply({"params": params}, jnp.array(ds.eval_inputs[sl]),
                            deterministic=True, capture_intermediates=True,
                            mutable=["intermediates"])
        it = aux["intermediates"]
        r = np.asarray(_get(it, "backbone", "__call__")[0])[bi, c]   # backbone output = post-final-LN, or last residual if no final LN
        blocks = [np.asarray(_get(it, "input_layer", "__call__")[0])[bi, c]]
        for i in range(n_layers):
            blocks.append(np.asarray(_get(it, "backbone", f"block_{i}", "__call__")[0])[bi, c])
        b = np.stack(blocks, axis=2)
        if read is None:
            read = np.zeros((n, NUM_ATTRIBUTES, r.shape[-1]), dtype=np.float32)
            lay = np.zeros((n, NUM_ATTRIBUTES, n_layers + 1, r.shape[-1]), dtype=np.float32)
        read[sl], lay[sl] = r, b
    return read, lay


def mean_pairwise_cos(v):
    """Mean cosine over ordered pairs of the rows of v, without forming the N x N matrix.

    sum_{a!=b} <u_a, u_b> = ||sum_a u_a||^2 - N for unit u, so the mean is that over N(N-1).
    Rows of norm 0 are dropped rather than producing a division by zero.
    """
    n = np.linalg.norm(v, axis=-1)
    keep = n > 1e-12
    if keep.sum() < 2:
        return float("nan")
    u = v[keep] / n[keep][:, None]
    m = u.shape[0]
    return float((np.linalg.norm(u.sum(0)) ** 2 - m) / (m * (m - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free", choices=sorted(ARM_SPECS))
    ap.add_argument("--pretrain_step", type=int, default=None,
                    help="defaults to the arm's own (16000 for the 8-layer arms, 8000 for 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    a = ap.parse_args()

    ctx = load_arm(a.arm, a.seed, a.condition, a.total_steps, a.checkpoint_dir, a.pretrain_step)
    model, data_a, n_layers = ctx.model, ctx.data_a, ctx.n_layers
    path_for, steps = ctx.path_for, ctx.steps

    cols = np.zeros((data_a.eval_inputs.shape[0], NUM_ATTRIBUTES), dtype=np.int64)
    m, t = np.asarray(data_a.eval_mask), np.asarray(data_a.eval_targets)
    for k in range(NUM_ATTRIBUTES):
        cols[:, k], _ = first_token_positions(m, t, k)

    if not os.path.exists(path_for(0)):
        raise SystemExit(f"step-0 checkpoint missing, and it is the baseline: {path_for(0)}")
    r0, l0 = states(model, ctx.load(0), data_a, cols, n_layers)
    N = r0.shape[0]

    # principal directions of A's step-0 readout states, per attribute
    pcs, evr = [], []
    for k in range(NUM_ATTRIBUTES):
        h = r0[:, k].astype(np.float64)
        h = h - h.mean(0)
        _, s, vt = np.linalg.svd(h, full_matrices=False)
        pcs.append(vt[:N_PC])
        evr.append((s ** 2 / (s ** 2).sum())[:N_PC])
    pcs = np.stack(pcs)                       # (A, N_PC, E)
    evr = np.stack(evr)
    print(f"N={N} attrs={NUM_ATTRIBUTES} layers={n_layers}")
    print(f"  step-0 representations: top-{N_PC} PCs explain "
          f"{evr.sum(1).mean():.1%} of A's readout variance (per-attribute mean)")
    print(f"  PC1 alone: {evr[:,0].mean():.1%}")
    print()
    print(f"  {'step':>5} | {'delta in top-20 PCs':>19} {'eps in top-20 PCs':>18} | "
          f"{'mean pairwise cos(eps)':>22}")

    out = {k: [] for k in ("steps", "delta_layer", "eps_layer_rms", "eps_cos_layer",
                           "delta_pc", "eps_pc", "eps_cos",
                           "delta_share", "eps_share", "eps_pc_sq", "eps_sq_norm")}
    for s in steps:
        p = path_for(s)
        if not os.path.exists(p):
            print(f"  {s:>5}: MISSING")
            continue
        r, l = states(model, ctx.load(s), data_a, cols, n_layers)
        dr = r - r0
        dl = l - l0
        # Free the two (N, A, L+1, E) blocks as soon as the difference exists, and never
        # materialise eps at layer resolution: at 2000 x 6 x 9 x 512 float32 that is 221MB per
        # array, and holding l, l0, dl and e_lay together took the job over an 8GB limit. The
        # per-layer quantities are all reductions, so they are computed one slice at a time.
        del l
        d_read = dr.mean(0)                                   # (A, E)
        e_read = dr - d_read[None]                            # (N, A, E)
        d_lay = dl.mean(0)                                    # (A, L+1, E)
        e_lay_rms = np.zeros((NUM_ATTRIBUTES, n_layers + 1), dtype=np.float32)
        ecos_l = np.zeros((NUM_ATTRIBUTES, n_layers + 1))
        for k in range(NUM_ATTRIBUTES):
            for j in range(n_layers + 1):
                e = dl[:, k, j] - d_lay[k, j]                  # (N, E)
                e_lay_rms[k, j] = np.sqrt((e.astype(np.float64) ** 2).sum(-1)).mean()
                ecos_l[k, j] = mean_pairwise_cos(e.astype(np.float64))
        del dl

        # share of each object's energy inside the step-0 principal subspace
        dpc = np.einsum("ake,ae->ak", pcs, d_read.astype(np.float64))
        d_share = (dpc ** 2).sum(1) / np.maximum((d_read.astype(np.float64) ** 2).sum(1), 1e-30)
        epc = np.einsum("ake,nae->nak", pcs, e_read.astype(np.float64))
        e_share = (epc ** 2).sum(2).mean(0) / np.maximum(
            (e_read.astype(np.float64) ** 2).sum(2).mean(0), 1e-30)
        ecos = np.array([mean_pairwise_cos(e_read[:, k].astype(np.float64))
                         for k in range(NUM_ATTRIBUTES)])

        out["steps"].append(s)
        out["delta_layer"].append(d_lay.astype(np.float32))
        out["eps_layer_rms"].append(e_lay_rms)
        out["eps_cos_layer"].append(ecos_l)
        out["delta_pc"].append(dpc)
        # Mean ABSOLUTE projection, kept for continuity with the existing npz. It is NOT an
        # energy and squaring it does not give one: recovering a fraction from it needs a
        # distributional assumption (a 2/pi Gaussian factor), which is why the exact shares
        # below are saved alongside.
        out["eps_pc"].append(np.abs(epc).mean(0))
        # The exact energy fractions. These were already computed here and only printed; not
        # saving them forced every downstream consumer to re-derive them from `eps_pc` under
        # an assumption the data does not need.
        out["delta_share"].append(d_share)
        out["eps_share"].append(e_share)
        out["eps_pc_sq"].append((epc ** 2).mean(0))
        out["eps_sq_norm"].append((e_read.astype(np.float64) ** 2).sum(2).mean(0))
        out["eps_cos"].append(ecos)
        print(f"  {s:>5} | {d_share.mean():>19.4f} {e_share.mean():>18.4f} | "
              f"{ecos.mean():>22.4f}")

    dl_ = np.stack(out["delta_layer"])                        # (S, A, L+1, E)
    print()
    print("  PER-LAYER CONTRIBUTION (block l adds entry l+1 minus entry l), attribute-averaged")
    print(f"  {'step':>5} " + " ".join(f'{"blk"+str(i):>7}' for i in range(n_layers)))
    for i, s in enumerate(out["steps"]):
        inc = np.linalg.norm(np.diff(dl_[i], axis=1), axis=-1)     # (A, L)
        print(f"  d {s:>3} " + " ".join(f"{v:>7.3f}" for v in inc.mean(0)))
    el_ = np.stack(out["eps_layer_rms"])
    for i, s in enumerate(out["steps"]):
        inc = np.diff(el_[i], axis=1)
        print(f"  e {s:>3} " + " ".join(f"{v:>7.3f}" for v in inc.mean(0)))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}-p{ctx.pretrain_step}-{a.condition}-t{a.total_steps}-seed{a.seed}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        steps=np.array(out["steps"]), delta_layer=dl_,
        eps_layer_rms=el_, eps_cos_layer=np.stack(out["eps_cos_layer"]),
        delta_pc=np.stack(out["delta_pc"]), eps_pc=np.stack(out["eps_pc"]),
        eps_cos=np.stack(out["eps_cos"]), pcs=pcs.astype(np.float32),
        delta_share=np.stack(out["delta_share"]), eps_share=np.stack(out["eps_share"]),
        eps_pc_sq=np.stack(out["eps_pc_sq"]), eps_sq_norm=np.stack(out["eps_sq_norm"]),
        explained_var=evr.astype(np.float32),
        seed=a.seed, pretrain_step=ctx.pretrain_step, condition=a.condition, arm=a.arm,
        key=ctx.key)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
