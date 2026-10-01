"""Is the head attribution measuring behaviour, or a gauge direction?

`analyze_head_attribution` scores each head by `<c_h, c> / ||c||^2` with `c` the full-vocabulary
logit shift. That basis has two defects, and this script measures how much they matter rather
than arguing about it. It reads the saved `c_head` arrays, so nothing is re-run.

DEFECT 1: THE SOFTMAX GAUGE. Adding a constant to every logit in a row changes nothing the model
does. `c` is not centred over the vocabulary, so any uniform-over-vocab component inflates
`||c||` while being behaviourally inert -- and a head is credited for writing in that direction.
Centring `c` over the vocabulary axis removes exactly the inert part.

DEFECT 2: THE SUPPORT. `||c||` runs over all ~1881 tokens, most of which never compete for the
value slot. Accuracy and rank depend on the gap among the population's OWN-HALF value tokens.
Restricting to those is the basis closest to behaviour.

Three bases are reported side by side:
    full      -- as published, the whole vocabulary, uncentred
    centred   -- vocabulary-mean removed, so the softmax gauge is gone
    values    -- own-half value tokens only, centred within that set

WHAT THE ANSWER DECIDES. If the head ranking is stable across all three, the concentration is a
fact about the mechanism and `||c||`'s defects did not drive it. If it moves, the published
ranking was partly an artifact of the basis and the behavioural one is the one to report.

Run:
  uv run python -m src.experiments.summarise_head_attribution --seed 0
"""
import argparse
import glob
import os

import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.knowledge_injection import build, get_partition
from src.experiments.analyze_mlpfree import _cfg_for

OUT_DIR = "head_attribution"


def own_half_ids(arm, pretrain_step, seed, condition, total_steps):
    """Token ids of the value half population A draws from, per attribute."""
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, _, _, _, _, _, _, _ = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    return [np.asarray(pop.attr_first_token_ids[k][halves[k][0]])
            for k in range(NUM_ATTRIBUTES)]


def shares(c_head, c_total, ids=None, centre=False):
    """(steps, H) signed share of each head, in one basis, plus ||c|| in that basis.

    Accumulated per attribute rather than stacked: the own-half value pools are RAGGED (the
    post-filter pools differ per attribute, 127-135 values), so there is no rectangular array
    to restrict to. Centring, where asked, is within each attribute's restricted set, since the
    softmax gauge is per row.
    """
    ch, ct = c_head.astype(np.float64), c_total.astype(np.float64)
    n_steps, n_attr, n_head, _ = ch.shape
    num = np.zeros((n_steps, n_head))
    den = np.zeros(n_steps)
    for k in range(n_attr):
        h = ch[:, k] if ids is None else ch[:, k][..., ids[k]]
        t = ct[:, k] if ids is None else ct[:, k][..., ids[k]]
        if centre:
            h = h - h.mean(axis=-1, keepdims=True)
            t = t - t.mean(axis=-1, keepdims=True)
        num += np.einsum("shv,sv->sh", h, t)
        den += (t ** 2).sum(axis=-1)
    den = np.maximum(den, 1e-30)
    return num / den[:, None], np.sqrt(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--topk", type=int, default=4)
    a = ap.parse_args()

    path = os.path.join(
        OUT_DIR, f"{a.arm}-p{a.pretrain_step}-{a.condition}-t{a.total_steps}-seed{a.seed}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing. It is the FULL array, which is gitignored and stays "
                         f"on the cluster; run this where analyze_head_attribution ran. "
                         f"Present: {sorted(glob.glob(os.path.join(OUT_DIR, '*.npz')))}")
    d = np.load(path)
    ch, ct, steps = d["c_head"], d["c_total"], d["steps"]
    nh = int(d["num_heads"])
    ids = own_half_ids(a.arm, a.pretrain_step, a.seed, a.condition, a.total_steps)
    print(f"own-half value tokens per attribute: {[len(i) for i in ids]} "
          f"of vocab {ct.shape[-1]}")

    bases = {
        "full": shares(ch, ct),
        "centred": shares(ch, ct, centre=True),
        "values": shares(ch, ct, ids=ids, centre=True),
    }

    # How much of ||c||^2 is the behaviourally inert uniform-over-vocab direction?
    ctf = ct.astype(np.float64)
    gauge = (ctf.mean(axis=-1) ** 2 * ctf.shape[-1]).sum(axis=1)
    print(f"\nfraction of ||c||^2 in the softmax-gauge (uniform-over-vocab) direction:")
    print("  " + "  ".join(f"{s}:{g:.3f}" for s, g in
                           zip(steps, gauge / np.maximum((ctf ** 2).sum(axis=(1, 2)), 1e-30))))

    print(f"\ntop-{a.topk} cumulative share, and the top head, per basis:")
    print(f"  {'step':>5} | " + " | ".join(f"{b:>22}" for b in bases))
    for i, s in enumerate(steps):
        if s == 0:
            continue
        cells = []
        for b, (sh, _) in bases.items():
            o = np.argsort(-sh[i])
            cells.append(f"{np.cumsum(sh[i][o])[a.topk - 1]:>6.3f} "
                         f"top L{o[0] // nh}H{o[0] % nh} {sh[i][o[0]]:>6.3f}")
        print(f"  {s:>5} | " + " | ".join(f"{c:>22}" for c in cells))

    print("\nrank correlation of per-head share between bases (Spearman, over heads):")
    for i, s in enumerate(steps):
        if s not in (50, 200, 1200):
            continue
        def rk(x):
            r = np.empty_like(x); r[np.argsort(x)] = np.arange(len(x)); return r
        f, c, v = (rk(bases[b][0][i]) for b in ("full", "centred", "values"))
        print(f"  step {s:>4}: full-centred {np.corrcoef(f, c)[0, 1]:.3f}   "
              f"full-values {np.corrcoef(f, v)[0, 1]:.3f}   "
              f"centred-values {np.corrcoef(c, v)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
