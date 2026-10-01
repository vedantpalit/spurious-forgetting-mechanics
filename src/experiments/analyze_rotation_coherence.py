"""Do the per-block OV pieces rotate together, or independently? (step 2 follow-up)

Reads `weight_patch/*.npz` only -- no checkpoints, no forward passes.

WHAT THE ROTATION IS. Each block's OV-attributable piece P_j = delta_none - delta_patched keeps
its NORM between the trough (step 50) and the recovery peak (step 200) -- all eight change by
less than 0.21 while ||delta|| itself falls 6.97 -- but turns 60-72 degrees. Read on norms alone
the blocks look inert during recovery. They are not; they reorient.

THE TEST. An orthogonal map preserves inner products, so a SINGLE rigid rotation carrying the
whole set from step 50 to step 200 exists exactly when the two cosine Grams are equal. Gram
preserved => the configuration moved as one object, which is a coherent thing to look for a
cause of. Gram scrambled => the pieces moved relative to each other, and the fall in the
cumulative is cancellation between pieces that no longer line up.

That is the whole content of the test, and it is worth stating what it is NOT: 8 vectors in 512
dimensions with matching Grams ALWAYS admit some rigid map, so fitting one proves nothing on its
own. Only the Gram comparison carries information.

THREE READINGS, all reported:
  1. Gram preservation -- max and mean |cos_ij(200) - cos_ij(50)| over pairs, and the correlation
     between the two Grams' off-diagonals.
  2. Coherence of the SUM -- ||sum_j P_j|| / sum_j ||P_j||, at each step. If the pieces are
     mutually aligned this is near 1; if mutually orthogonal, near 1/sqrt(8) = 0.354; if they
     actively cancel, below that. A fall in this ratio IS the cancellation account, measured.
  3. Alignment of the DISPLACEMENTS D_j = P_j(200) - P_j(50). A common additive shift shows as
     mutually aligned D_j; independent motion shows as mutually orthogonal ones.

Per attribute throughout, then averaged, because delta itself is per-attribute.

Run: uv run python -m src.experiments.analyze_rotation_coherence
"""
import argparse
import glob
import os

import numpy as np

OUT_DIR = "weight_patch"
ATTRS = ["birthdate", "birthplace", "university", "major", "company", "work_location"]


def load(patch, step, arm, pretrain_step, condition):
    pat = os.path.join(OUT_DIR, f"{arm}-{patch}-p{pretrain_step}-{condition}-step{step}-seed*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no files matching {pat}")
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["names"]]
        idx = [names.index(f"blk{j}") for j in range(8)]
        out.append(d["removed"][idx].astype(np.float64))    # (8, A, E)
    return np.stack(out), files                              # (S, 8, A, E)


def unit(v, axis=-1):
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12, None)


def gram(P):
    """Cosine Gram over the block axis. P is (8, E) -> (8, 8)."""
    U = unit(P)
    return U @ U.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="ov")
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--steps", default="50,200")
    a = ap.parse_args()
    s0, s1 = [int(s) for s in a.steps.split(",")]

    Pe, fe = load(a.patch, s0, a.arm, a.pretrain_step, a.condition)
    Pl, fl = load(a.patch, s1, a.arm, a.pretrain_step, a.condition)
    S, B, A, E = Pe.shape
    print(f"patch={a.patch}  {S} seeds  {B} blocks  {A} attributes  dim {E}")
    print(f"  step {s0}: {len(fe)} files   step {s1}: {len(fl)} files")

    iu = np.triu_indices(B, k=1)

    print(f"\n1. GRAM PRESERVATION -- is one rigid rotation enough?")
    print(f"   {'attribute':>14} {'mean|dcos|':>11} {'max|dcos|':>10} {'corr(G50,G200)':>15} "
          f"{'mean cos@'+str(s0):>13} {'mean cos@'+str(s1):>13}")
    dg_all, c0_all, c1_all = [], [], []
    for k in range(A):
        dg, cc, m0, m1 = [], [], [], []
        for s in range(S):
            G0, G1 = gram(Pe[s, :, k]), gram(Pl[s, :, k])
            d = np.abs(G1[iu] - G0[iu])
            dg.append(d)
            cc.append(np.corrcoef(G0[iu], G1[iu])[0, 1])
            m0.append(G0[iu].mean()); m1.append(G1[iu].mean())
        dg = np.concatenate(dg)
        dg_all.append(dg); c0_all.append(np.mean(m0)); c1_all.append(np.mean(m1))
        print(f"   {ATTRS[k]:>14} {dg.mean():>11.3f} {dg.max():>10.3f} {np.mean(cc):>15.3f} "
              f"{np.mean(m0):>13.3f} {np.mean(m1):>13.3f}")
    print(f"   {'ALL':>14} {np.concatenate(dg_all).mean():>11.3f} "
          f"{np.concatenate(dg_all).max():>10.3f} {'':>15} "
          f"{np.mean(c0_all):>13.3f} {np.mean(c1_all):>13.3f}")
    print(f"   A rigid rotation requires mean|dcos| ~ 0. Each block's own turn is 60-72 deg,")
    print(f"   so a preserved Gram means they turned TOGETHER.")

    print(f"\n2. COHERENCE OF THE SUM -- ||sum_j P_j|| / sum_j ||P_j||")
    print(f"   orthogonal pieces give 1/sqrt(8) = {1/np.sqrt(B):.3f}; aligned pieces give ~1")
    print(f"   {'attribute':>14} {'@'+str(s0):>8} {'@'+str(s1):>8} {'change':>8}")
    for k in range(A):
        r0 = np.mean([np.linalg.norm(Pe[s, :, k].sum(0)) / np.linalg.norm(Pe[s, :, k], axis=-1).sum()
                      for s in range(S)])
        r1 = np.mean([np.linalg.norm(Pl[s, :, k].sum(0)) / np.linalg.norm(Pl[s, :, k], axis=-1).sum()
                      for s in range(S)])
        print(f"   {ATTRS[k]:>14} {r0:>8.3f} {r1:>8.3f} {r1-r0:>+8.3f}")

    print(f"\n3. DISPLACEMENTS D_j = P_j({s1}) - P_j({s0}) -- do the blocks move the same way?")
    print(f"   {'attribute':>14} {'mean pair cos(D)':>17} {'mean ||D||':>11} "
          f"{'mean cos(D_j,P_j({}))'.format(s0):>22}")
    for k in range(A):
        mc, nd, dp = [], [], []
        for s in range(S):
            D = Pl[s, :, k] - Pe[s, :, k]
            G = gram(D)
            mc.append(G[iu].mean())
            nd.append(np.linalg.norm(D, axis=-1).mean())
            dp.append(np.mean(np.sum(unit(D) * unit(Pe[s, :, k]), axis=-1)))
        print(f"   {ATTRS[k]:>14} {np.mean(mc):>17.3f} {np.mean(nd):>11.3f} {np.mean(dp):>22.3f}")

    print(f"\n4. PER-BLOCK SELF-ROTATION cos(P_j({s0}), P_j({s1})), for reference")
    sc = np.zeros((A, B))
    for k in range(A):
        for j in range(B):
            sc[k, j] = np.mean([np.dot(unit(Pe[s, j, k]), unit(Pl[s, j, k])) for s in range(S)])
    print(f"   {'attribute':>14} " + " ".join(f"{'blk'+str(j):>7}" for j in range(B)))
    for k in range(A):
        print(f"   {ATTRS[k]:>14} " + " ".join(f"{v:>7.3f}" for v in sc[k]))
    print(f"   {'mean':>14} " + " ".join(f"{v:>7.3f}" for v in sc.mean(0)))


if __name__ == "__main__":
    main()
