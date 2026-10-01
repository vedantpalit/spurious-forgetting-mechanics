"""Shared pieces for the MLP-free ablation.

Mirrors the role `scale8_common.py` plays for the scale check -- one place for the model
factory, the params-derived verification, the arm-discriminated checkpoint key, and the load
diagnostics, so the pretrain, injection, and diagnostic drivers cannot silently diverge.

Two things here exist specifically to make failure modes loud rather than silent:

  * `verify_no_mlp` proves from the *constructed weights* that no MLP tensors exist, in the
    same spirit as `scale8_common.verify_architecture` -- a config saying `mlp_coefficient=0`
    is not evidence that the model built has no MLP.
  * `mlpfree_pretrain_key` adds `arm="mlp_free"` to the key. Without it the MLP-free arm
    would hash to the *same* key as the standard arm (the flag that removes the MLP lives on
    the flax module, deliberately not on `ModelConfig` -- see the note in backbone.py) and
    the two arms would overwrite each other's checkpoints.
"""
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from src.data.config import DataConfig
from src.model.backbone import TransformerBackbone
from src.model.config import ModelConfig
from src.model.factory import SequenceModel, _get_dtype
from src.model.heads import ClassificationHead
from src.model.layers import EmbeddingInput
from src.experiments.ckpt import experiment_key, file_hash

# Same architecture as the scale check, so the only difference between arms is the MLP.
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8

# Tensor-name fragments that only ever appear in an MLP sublayer of a TransformerBlock.
# In this backbone a block's submodules are named positionally by flax:
#   LayerNorm_0, CausalSelfAttention_0  (attention half, always present)
#   LayerNorm_1, Dense_0, Dense_1       (MLP half, absent when mlp_enabled=False)
_MLP_ONLY_BLOCK_KEYS = ("LayerNorm_1", "Dense_0", "Dense_1")


def create_mlpfree_model(model_config: ModelConfig, data_config: DataConfig) -> SequenceModel:
    """Same composition as `create_model`, with attention-only blocks.

    Requires `mlp_coefficient == 0`: the config must not claim an MLP width the model does
    not have. That also makes the checkpoint key differ from the standard arm's automatically,
    on top of the explicit `arm=` discriminator in `mlpfree_pretrain_key`.
    """
    if model_config.mlp_coefficient != 0:
        raise ValueError(
            f"create_mlpfree_model requires mlp_coefficient=0 (got "
            f"{model_config.mlp_coefficient}). The config must state that there is no MLP, "
            f"so it cannot disagree with the model and so the checkpoint key differs from "
            f"the standard arm's.")

    dtype = _get_dtype(model_config.dtype)
    backbone = TransformerBackbone(
        num_heads=model_config.num_heads,
        model_dim=model_config.model_dim,
        num_layers=model_config.num_layers,
        mlp_dim=0,
        dropout_rate=model_config.dropout_rate,
        positional_encoding=model_config.positional_encoding,
        max_seq_len=model_config.max_seq_len,
        activation=model_config.activation,
        dtype=dtype,
        mlp_enabled=False,
    )
    if data_config.task_type != "biography":
        raise ValueError(f"Unknown task_type: {data_config.task_type}")
    return SequenceModel(
        input_layer=EmbeddingInput(vocab_size=data_config.vocab_size,
                                    model_dim=model_config.model_dim, dtype=dtype),
        backbone=backbone,
        head=ClassificationHead(vocab_size=data_config.vocab_size, dtype=dtype),
        dropout_rate=model_config.dropout_rate,
        dtype=dtype,
    )


def verify_no_mlp(params, verbose=True) -> int:
    """Prove from the constructed weights that no MLP tensors exist. Returns the count of
    MLP-only tensors found (0 for a correct MLP-free model) and raises if it is nonzero.

    Deliberately params-derived, not config-derived: `scale8_common.verify_architecture`
    exists because a config flag can be silently dropped, and the same applies here.
    """
    backbone_params = params["backbone"]
    offenders = []
    for block_name, block in backbone_params.items():
        if not block_name.startswith("block_"):
            continue
        for sub in block:
            if sub in _MLP_ONLY_BLOCK_KEYS:
                offenders.append(f"{block_name}/{sub}")

    if verbose:
        blocks = [k for k in backbone_params if k.startswith("block_")]
        example = sorted(backbone_params[blocks[0]].keys()) if blocks else []
        print("=== MLP-free verification (derived from actual constructed weights) ===")
        print(f"  blocks found: {len(blocks)}")
        print(f"  submodules in block_0: {example}")
        print(f"  MLP-only tensors found: {len(offenders)}")

    if offenders:
        raise ValueError(
            f"Model still has {len(offenders)} MLP tensor group(s): {offenders[:6]}"
            f"{' ...' if len(offenders) > 6 else ''}. mlp_enabled was not applied.")
    if verbose:
        print("  OK -- no MLP sublayers exist in the constructed model.\n")
    return len(offenders)


def load_ratio(pop, halves, num_a, num_ballast, support_size):
    """Individuals per value, per attribute, per half -- the load diagnostic that runs at
    every gate whether it passes or fails.

    Computed per half, not pooled: A draws only from X and ballast only from Y, so the two
    populations' loads land on disjoint value sets and do NOT add. What doubling the
    population changes is the total fact count, not this ratio -- which is exactly why the
    A-alone diagnostic localizes a per-value ceiling against a total-load ceiling.

    Uses post-filter pool counts (nominal pool size is not effective pool size).
    Returns (rows, summary) where rows are per-attribute dicts.
    """
    rows = []
    for k, (x_idx, y_idx) in enumerate(halves):
        n_x, n_y = len(x_idx), len(y_idx)
        rows.append({
            "attribute": k,
            "pool_post_filter": int(pop.num_values_per_attr[k]),
            "half_X": n_x,
            "half_Y": n_y,
            "A_per_value": num_a / n_x if n_x else float("nan"),
            "ballast_per_value": num_ballast / n_y if n_y else float("nan"),
        })
    a_ratios = [r["A_per_value"] for r in rows]
    summary = {
        "A_per_value_min": float(np.min(a_ratios)),
        "A_per_value_mean": float(np.mean(a_ratios)),
        "A_per_value_max": float(np.max(a_ratios)),
        "support_size": support_size,
        "total_facts_A_union_D": (num_a + num_ballast) * len(rows),
    }
    return rows, summary


def print_load_ratio(pop, halves, num_a, num_ballast, support_size):
    rows, summary = load_ratio(pop, halves, num_a, num_ballast, support_size)
    print("=== Load ratio (post-filter pools, per half -- halves are disjoint, so these "
          "do NOT add) ===")
    print(f"  {'attr':>5} {'pool':>6} {'|X|':>5} {'|Y|':>5} {'A/value':>9} {'D/value':>9}")
    for r in rows:
        print(f"  {r['attribute']:>5} {r['pool_post_filter']:>6} {r['half_X']:>5} "
              f"{r['half_Y']:>5} {r['A_per_value']:>9.1f} {r['ballast_per_value']:>9.1f}")
    print(f"  A individuals-per-value: min={summary['A_per_value_min']:.1f} "
          f"mean={summary['A_per_value_mean']:.1f} max={summary['A_per_value_max']:.1f}  "
          f"(support_size={support_size})")
    print(f"  Total facts stored (A+D) x {len(rows)} attributes = "
          f"{summary['total_facts_A_union_D']:,}")
    print("  Reference points: the hand-built toy plateaued at ~31 individuals-per-value "
          "and converged at ~7.8. The standard 8-layer model reaches ceiling at this same "
          "ratio, so a ratio near 30 is not prohibitive for a model WITH MLPs.\n")
    return rows, summary


def per_attribute_first_acc(state, ds, batch_size=256):
    """First-token accuracy for attribute k, separately for each k -- NOT pooled.

    This does not exist anywhere in the training path: `evaluate_and_log` pools every
    attribute into `{ds}/first_token_accuracy` via METRIC_CATEGORIES, and
    `population_rank_metrics` concatenates across k before aggregating. The *machinery*
    exists (`first_token_positions` selects attribute k's first value-token position from
    the +k role mask), so this is cheap -- one extra forward pass over the eval set -- but
    it is new wiring, not a call to something already logged.

    Needed because the load-ratio table predicts an ordering across attributes, and testing
    it *within* one run controls for architecture, optimizer, and data pipeline in a way the
    between-model comparison to the toy cannot.
    """
    from src.data.biography import NUM_ATTRIBUTES
    from src.experiments.knowledge_injection import eval_forward, first_token_positions

    n = ds.eval_inputs.shape[0]
    hits = [[] for _ in range(NUM_ATTRIBUTES)]
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(state.apply_fn, state.params,
                                          jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[np.arange(sl.stop - sl.start), cols, :]
            hits[k].append(row_logits.argmax(axis=-1) == correct)
    return np.array([float(np.concatenate(h).mean()) for h in hits])


def _spearman(a, b):
    """Rank correlation without a scipy dependency on the cluster path."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# Display heuristic only -- the human decides. Printed alongside the raw slope so it can be
# overridden by eye; it gates nothing automatically.
_CLIMBING_EPS = 0.005  # accuracy per 1000 steps


class GateProbe:
    """Accumulates the gate's per-attribute accuracy trajectory and reports, at each probe
    step, the three things the gate reading needs: per-attribute accuracy against the load
    ratio, the slope (flat vs still-climbing), and the occupancy the whole thing is
    happening at.
    """

    def __init__(self, load_rows, occupancy_pct, n_params):
        self.load_rows = load_rows
        self.occupancy_pct = occupancy_pct
        self.n_params = n_params
        self.history = []  # (step, aggregate_acc, per_attr_acc)

    def probe(self, step, state, ds_a):
        per_attr = per_attribute_first_acc(state, ds_a)
        agg = float(per_attr.mean())
        self.history.append((step, agg, per_attr))

        ratios = [r["A_per_value"] for r in self.load_rows]
        rho = _spearman(ratios, per_attr)

        print(f"\n=== Gate probe, step {step} "
              f"(occupancy {self.occupancy_pct:.4%}, {self.n_params:,} params) ===")
        print(f"  {'attr':>5} {'pool':>6} {'A/value':>9} {'first_acc':>10}")
        for r, acc in zip(self.load_rows, per_attr):
            print(f"  {r['attribute']:>5} {r['pool_post_filter']:>6} "
                  f"{r['A_per_value']:>9.1f} {acc:>10.4f}")
        print(f"  aggregate (mean over attributes) = {agg:.4f}")
        print(f"  Spearman(load ratio, accuracy) over 6 attributes = {rho:+.3f}  "
              f"[n=6: directional, not precise. Negative = the predicted dose-response, "
              f"i.e. lower load -> higher accuracy.]")

        if len(self.history) >= 2:
            (s0, a0, _), (s1, a1, _) = self.history[-2], self.history[-1]
            slope_1k = (a1 - a0) / max(s1 - s0, 1) * 1000
            print(f"  slope since step {s0}: {slope_1k:+.4f} accuracy per 1000 steps")
            if slope_1k > _CLIMBING_EPS:
                print(f"  -> STILL CLIMBING (> {_CLIMBING_EPS} per 1k). Let it run; do NOT "
                      f"call a structural plateau on a curve that still has slope.")
            else:
                print(f"  -> FLAT (<= {_CLIMBING_EPS} per 1k). If also far from ceiling, this "
                      f"is the structural-plateau signature. Display heuristic -- confirm by eye "
                      f"against the trajectory below.")
        if len(self.history) >= 3:
            traj = "  ".join(f"{s}:{a:.4f}" for s, a, _ in self.history)
            print(f"  trajectory: {traj}")
        print()
        return per_attr


def mlpfree_pretrain_key(cfg, data_cfg):
    """`knowledge_injection.pretrain_key` plus `arm="mlp_free"`.

    Deliberately a separate function rather than a parameter on the shared one: the shared
    key function is used by every existing driver, and changing its signature or its hashed
    contents would re-key checkpoints that already exist on the cluster.
    """
    pretrain_for_key = replace(cfg.pretrain, max_eval_people=0, eval_interval=0, target_acc=0.0)
    partition_hash = (file_hash(cfg.partition_path)
                       if cfg.partition_path and os.path.exists(cfg.partition_path) else None)
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="pretrain",
        pretrain=pretrain_for_key, seed=cfg.seed,
        population=[cfg.num_a, cfg.num_ballast, cfg.num_b, cfg.num_c],
        c_half=cfg.c_half, partition_seed=cfg.partition_seed, partition_hash=partition_hash,
        arm="mlp_free")
