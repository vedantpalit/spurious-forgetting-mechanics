"""Offline analysis for the MLP-free ablation, run over saved injection checkpoints.

Both arms go through THIS code path, deliberately. The standard arm's mean rank is not in
its logs (`evaluate_and_log`'s console `_fmt` prints only `rank_own_top1`), so it has to be
recomputed from checkpoints regardless; recomputing the MLP-free arm the same way -- rather
than printing it live for one arm and recomputing it for the other -- means the two arms'
numbers cannot differ because of how they were produced.

Computes, per (arm, pretrain_step, seed, injection step):
  * A full-vocabulary first-token accuracy   -- the suppression-sensitive readout
  * A own-half MEAN RANK                     -- the erosion-sensitive readout
  * both, decomposed per attribute
  * fact-level probability on the correct value, per individual, full-vocab AND own-half
    -- deletion moves mass to zero, coarsening slides the whole distribution left, and a
    scalar mean averages away exactly that distinction
  * attention-to-name-token mass per layer   -- a stability check, from the sow added to
    CausalSelfAttention (at a subset of steps; see ATTENTION_STEPS)

Per-person arrays are kept at (N, 6) so nothing is collapsed before it is saved.

Run (one invocation per arm/checkpoint/seed):
  uv run python -m src.experiments.analyze_mlpfree --arm mlp_free --pretrain_step 16000 --seed 0
  uv run python -m src.experiments.analyze_mlpfree --arm standard --pretrain_step 16000 --seed 0
Then:
  uv run python -m src.experiments.analyze_mlpfree --summarize
"""
import argparse
import glob
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.data.biography import NUM_ATTRIBUTES
from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, eval_forward, first_token_positions, get_partition,
    init_state, make_optimizer, pretrain_key,
)
from src.experiments.analyze_demo_injection import per_person_first_correct
from src.experiments.analyze_name_overlap import per_person_rank_own
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key
from src.model.factory import create_model

OUT_DIR = "mlpfree_analysis"
PRETRAIN_PEAK_LR = 5e-4
PRETRAIN_TOTAL_STEPS = 16000

# Per-arm differences, in one place so the two cannot drift apart.
ARMS = {
    "standard": dict(
        mlp_coefficient=4, phase="scale8inject", prefix="scale8inject",
        default_total_steps=1200, key_extra={},
    ),
    "mlp_free": dict(
        mlp_coefficient=0, phase="mlpfreeinject", prefix="mlpfreeinject",
        default_total_steps=1200, key_extra={"arm": "mlp_free"},
    ),
}

# The dense schedule follows the run length. Run length is ONLY run length here:
# injection uses a constant LR (InjectConfig overrides OptConfig's cosine default), so
# the 1200- and 3000-step MLP-free runs are the same trajectory measured to different
# horizons -- confirmed empirically, they agree to four decimals at every shared step.
# Both are cross-arm comparable; 1200 matches the standard arm's checkpoint steps exactly.
SCHEDULES = {
    1200: "0,10,25,50,100,200,400,600,800,1000,1200",
    3000: "0,10,25,50,100,200,400,600,800,1000,1200,1600,2000,2400,3000",
}

# Attention capture is far more expensive than the accuracy/rank passes (the sown tensors
# are (B, H, L, L) with L=195), so it runs on a subsample at a subset of steps: enough to
# answer "does attention move during injection" without building a patching apparatus.
ATTENTION_STEPS = (0, 50, 200, 1200)
ATTENTION_PEOPLE = 256
ATTENTION_BATCH = 32


def _cfg_for(arm, pretrain_step, seed, condition, total_steps):
    spec = ARMS[arm]
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", "512", "--model.num_heads", "8", "--model.num_layers", "8",
        "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(spec["mlp_coefficient"]),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", condition, "--inject.seed", str(seed),
        "--inject.total_steps", str(total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10",
        "--inject.checkpoint_steps", SCHEDULES[total_steps],
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ]
    return parse_config(KIConfig, description=f"MLP-free analysis ({arm})")


def _inject_key(arm, cfg, data_cfg, pre_key, pretrain_step):
    spec = ARMS[arm]
    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=pretrain_step, seed=cfg.seed, **spec["key_extra"])


def per_person_prob_correct(model, params, ds, x_ids, batch_size=256):
    """Per-person, per-attribute probability on the correct value, two ways:
    full-vocabulary softmax, and softmax restricted to the population's own half.

    Neither exists in src/ today. The own-half one is nearly free -- `rank_and_loss_within`
    already computes the own-half loss per row and every caller throws it away -- but the
    full-vocab one is what distinguishes deletion (mass to zero) from coarsening (the whole
    distribution sliding left), so both are returned. (N, 6) each.
    """
    n = ds.eval_inputs.shape[0]
    full = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float64)
    own = np.zeros((n, NUM_ATTRIBUTES), dtype=np.float64)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        rows = np.arange(sl.stop - sl.start)
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[rows, cols, :]
            correct_logit = row_logits[rows, correct]

            m = row_logits.max(axis=1)
            lse_full = m + np.log(np.exp(row_logits - m[:, None]).sum(axis=1))
            full[sl, k] = np.exp(correct_logit - lse_full)

            cand = row_logits[:, x_ids[k]]
            mo = cand.max(axis=1)
            lse_own = mo + np.log(np.exp(cand - mo[:, None]).sum(axis=1))
            own[sl, k] = np.exp(correct_logit - lse_own)
    return full, own


def attention_to_name_mass(model, params, ds, pad_id, prompt_len,
                            n_people=ATTENTION_PEOPLE, batch_size=ATTENTION_BATCH):
    """Mean attention mass, per layer, from each attribute's retrieval position onto the
    NAME tokens of the prompt.

    Retrieval position: the column whose logits predict attribute k's first value token,
    i.e. exactly the column `first_token_positions` returns. Name tokens: prompt positions
    1..prompt_len-1 that are not pad -- read off the input ids directly rather than
    reconstructed from per-person name lengths, so it cannot drift from the actual data.

    Returns (n_layers,) mean mass, averaged over heads, people and attributes.

    Caveat on the absolute value, found while validating this on a small model: it came
    back ~0.73-0.83 at random init, which looks implausibly high until you notice that
    attribute order is permuted per biography, so the earliest attribute's retrieval
    position sits only ~20 columns in -- and under causal masking ~14 of those ~20 visible
    keys ARE the name. The absolute number therefore mixes attributes whose retrieval
    positions see very different amounts of context, and is not interpretable on its own.
    What this measures is whether the mass MOVES across injection; the level is not a
    claim.
    """
    n = min(n_people, ds.eval_inputs.shape[0])
    sums, counts = None, 0
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        x = jnp.array(ds.eval_inputs[sl])
        _logits, mutated = model.apply(
            {"params": params} if "params" not in params else params,
            x, deterministic=True, mutable=["intermediates"])
        inter = mutated["intermediates"]["backbone"]
        layer_names = sorted((k for k in inter if k.startswith("block_")),
                              key=lambda s: int(s.split("_")[1]))

        inputs = np.asarray(ds.eval_inputs[sl])
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        rows = np.arange(sl.stop - sl.start)

        # (B, L) boolean over key positions: real name tokens only.
        is_name = np.zeros_like(inputs, dtype=bool)
        is_name[:, 1:prompt_len] = inputs[:, 1:prompt_len] != pad_id

        if sums is None:
            sums = np.zeros(len(layer_names), dtype=np.float64)
        for li, lname in enumerate(layer_names):
            attn = np.asarray(
                inter[lname]["CausalSelfAttention_0"]["attn_weights"][0])  # (B, H, L, L)
            for k in range(NUM_ATTRIBUTES):
                cols, _correct = first_token_positions(mask, targets, k)
                q = attn[rows, :, cols, :]              # (B, H, L) -- keys for that query
                # Weight by rows only. Multiplying by NUM_ATTRIBUTES here as well (as an
                # earlier version did) while incrementing `counts` once per batch turned
                # the per-attribute average into a SUM over attributes, inflating every
                # reported mass by exactly 6x -- visible because an attention mass must lie
                # in [0, 1] and the values came back as high as 4.14.
                sums[li] += float((q * is_name[:, None, :]).sum(axis=-1).mean() * len(rows))
        counts += len(rows) * NUM_ATTRIBUTES
    return sums / counts


def analyze_run(arm, pretrain_step, seed, condition, checkpoint_dir, do_attention,
                 total_steps=None):
    spec = ARMS[arm]
    total_steps = total_steps or spec["default_total_steps"]
    cfg = _cfg_for(arm, pretrain_step, seed, condition, total_steps)
    pop, std_model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    model = std_model if arm == "standard" else create_mlpfree_model(cfg.model, data_cfg)

    pre_key = (pretrain_key(cfg, data_cfg) if arm == "standard"
                else mlpfree_pretrain_key(cfg, data_cfg))
    key = _inject_key(arm, cfg, data_cfg, pre_key, pretrain_step)

    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]

    tx = make_optimizer(replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO),
                         cfg.inject.total_steps)
    template = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    steps = [int(s) for s in SCHEDULES[total_steps].split(",")]
    print(f"arm={arm} pretrain_step={pretrain_step} seed={seed} condition={condition} "
          f"inject_total_steps={total_steps}")
    print(f"  pretrain key={pre_key}  inject key={key}")

    out = {"steps": [], "first_acc": [], "rank_own": [], "prob_full": [], "prob_own": [],
           "attn_steps": [], "attn_name_mass": []}
    for s in steps:
        path = os.path.join(checkpoint_dir, f"{spec['prefix']}-p{pretrain_step}-{key}-step{s:04d}.msgpack")
        if not os.path.exists(path):
            print(f"  step {s:>5}: MISSING ({path})")
            continue
        params = load_params(path, template.params)

        fc = per_person_first_correct(model, params, data_a)
        rk = per_person_rank_own(model, params, data_a, x_ids)
        pf, po = per_person_prob_correct(model, params, data_a, x_ids)
        out["steps"].append(s)
        out["first_acc"].append(fc)
        out["rank_own"].append(rk)
        out["prob_full"].append(pf)
        out["prob_own"].append(po)
        print(f"  step {s:>5}: first_acc={fc.mean():.4f}  rank_own_mean={rk.mean():.3f}  "
              f"p_correct(full)={pf.mean():.4f}  p_correct(own)={po.mean():.4f}")

        if do_attention and s in ATTENTION_STEPS:
            mass = attention_to_name_mass(model, params, data_a, pop.pad_id, pop.prompt_len)
            out["attn_steps"].append(s)
            out["attn_name_mass"].append(mass)
            print(f"           attention-to-name mass by layer: "
                  f"{np.round(mass, 4).tolist()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(
        OUT_DIR, f"{arm}-p{pretrain_step}-{condition}-t{total_steps}-seed{seed}.npz")
    # Compact dtypes: these are (S, N, 6) = 15 x 2000 x 6 per array, and at float64 the
    # 20 runs come to ~60MB of git history for no gained precision. first_acc is 0/1,
    # rank is a small integer (own half is <= 111 candidates), probabilities are fine in
    # float32.
    np.savez_compressed(
        dest,
        steps=np.array(out["steps"]),
        first_acc=np.array(out["first_acc"]).astype(np.uint8),      # (S, N, 6)
        rank_own=np.array(out["rank_own"]).astype(np.uint16),       # (S, N, 6)
        prob_full=np.array(out["prob_full"]).astype(np.float32),    # (S, N, 6)
        prob_own=np.array(out["prob_own"]).astype(np.float32),      # (S, N, 6)
        attn_steps=np.array(out["attn_steps"]),
        attn_name_mass=np.array(out["attn_name_mass"]) if out["attn_name_mass"]
            else np.zeros((0, 0)),
        arm=arm, pretrain_step=pretrain_step, seed=seed, condition=condition,
        inject_total_steps=total_steps)
    print(f"  saved -> {dest}")
    return dest


def summarize():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    if not files:
        raise SystemExit(f"No npz files in {OUT_DIR}/")
    groups = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        # Older files predate the inject_total_steps field; those were the 3000-step
        # MLP-free runs and the 1200-step standard ones.
        t = (int(d["inject_total_steps"]) if "inject_total_steps" in d.files
             else (1200 if str(d["arm"]) == "standard" else 3000))
        gk = (str(d["arm"]), int(d["pretrain_step"]), str(d["condition"]), t)
        groups.setdefault(gk, []).append(d)

    for gk in sorted(groups, key=lambda t: (t[3], t[0], t[1])):
        arm, pstep, cond, tsteps = gk
        ds = groups[gk]
        steps = ds[0]["steps"]
        fa = np.stack([d["first_acc"].mean(axis=(1, 2)) for d in ds])   # (seeds, S)
        rk = np.stack([d["rank_own"].mean(axis=(1, 2)) for d in ds])
        print(f"\n=== {arm}  pretrain_step={pstep}  {cond}  ({len(ds)} seeds) ===")
        print(f"  {'step':>6} {'first_acc':>18} {'rank_own_mean':>20}")
        for i, s in enumerate(steps):
            print(f"  {s:>6} {fa[:, i].mean():>10.4f} +/-{fa[:, i].std():<6.4f} "
                  f"{rk[:, i].mean():>12.3f} +/-{rk[:, i].std():<6.3f}")
        base, trough = fa[:, 0].mean(), fa.mean(axis=0).min()
        peak_after = fa.mean(axis=0)[np.argmin(fa.mean(axis=0)):].max()
        print(f"  baseline={base:.4f}  trough={trough:.4f}  "
              f"crash_depth={base - trough:.4f}  "
              f"recovery_fraction={(peak_after - trough) / max(base - trough, 1e-9):.3f}")
        print(f"  rank_own_mean: step0={rk[:, 0].mean():.3f}  "
              f"end={rk[:, -1].mean():.3f}  damage={rk[:, -1].mean() - rk[:, 0].mean():+.3f}")
        if ds[0]["attn_name_mass"].size:
            am = np.stack([d["attn_name_mass"] for d in ds]).mean(axis=0)  # (A, layers)
            asteps = ds[0]["attn_steps"]
            print(f"  attention-to-name mass by layer:")
            for j, s in enumerate(asteps):
                print(f"    step {s:>5}: {np.round(am[j], 4).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--pretrain_step", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--inject_total_steps", type=int, default=None,
                     choices=sorted(SCHEDULES),
                     help="defaults to the arm's matched length (1200)")
    ap.add_argument("--no_attention", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return
    if args.arm is None or args.pretrain_step is None or args.seed is None:
        raise SystemExit("--arm, --pretrain_step and --seed are required (or --summarize)")
    analyze_run(args.arm, args.pretrain_step, args.seed, args.condition,
                 args.checkpoint_dir, not args.no_attention, args.inject_total_steps)


if __name__ == "__main__":
    main()
