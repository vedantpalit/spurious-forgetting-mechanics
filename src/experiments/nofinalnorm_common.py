"""Shared pieces for the no-final-norm ablation: the direct test of the toy's recovery term.

WHAT IS BEING TESTED. In the K=1 toy (paper Section 3) the recovery of the old population comes
from one place: the parameter-free normalizer in front of the softmax. Its Jacobian gives every
injected individual's error signal an extra term, (M_b / ||h_b||) h_hat_b, whose coefficient is
that individual's confidence margin. That term builds the shared shift on A while B is wrong
and withdraws it once B is right. With a linear readout the term does not exist and the crash
never reverses (0 of 96 runs).

The transformer's final LayerNorm is that normalizer. So the prediction is: an otherwise
identical 8-layer model whose backbone ends at the last block's residual, with no LayerNorm
before the unembedding, crashes when B is injected and DOES NOT RECOVER -- delta (the common
part of A's displacement) should not reverse, eps should be unchanged. If A recovers anyway,
the toy's account of recovery is wrong for the transformer.

WHAT IS NOT REMOVED. The pre-norm inside every block. The model is not trainable at this
depth without them, and the toy's K-stack found that per-block normalizers generate no
anti-state term (autodiff to 1e-13): the one term is generated at the final normalizer.
Removing the block norms would change trainability and the comparison at once.

Mirrors mlpfree_common.py in structure and in the two things that make failure loud:

  * `verify_no_final_norm` proves from the CONSTRUCTED WEIGHTS that the backbone's final
    LayerNorm does not exist, and that every block's own LayerNorms still do.
  * `nofinalnorm_pretrain_key` adds `arm="no_final_norm"` to the key. `final_norm` lives on
    the flax module, deliberately not on `ModelConfig` (see backbone.py), so without the
    discriminator this arm would hash to the same key as the standard arm and overwrite it.
"""
import os
from dataclasses import replace

import jax

from src.data.config import DataConfig
from src.model.backbone import TransformerBackbone
from src.model.config import ModelConfig
from src.model.factory import SequenceModel, _get_dtype
from src.model.heads import ClassificationHead
from src.model.layers import EmbeddingInput
from src.experiments.ckpt import experiment_key, file_hash

# Same architecture as the standard 8-layer arm (scale8_common), so the only difference
# between the arms is the final LayerNorm.
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
MLP_COEFFICIENT = 4
ARM = "no_final_norm"

# flax names the backbone's own LayerNorm positionally: it is the only LayerNorm that is a
# direct child of the backbone (the blocks' LayerNorms live under block_i/).
_FINAL_NORM_KEY = "LayerNorm_0"


def create_nofinalnorm_model(model_config: ModelConfig, data_config: DataConfig) -> SequenceModel:
    """Same composition as `create_model`, with `final_norm=False` on the backbone."""
    if model_config.mlp_coefficient != MLP_COEFFICIENT:
        raise ValueError(
            f"create_nofinalnorm_model expects mlp_coefficient={MLP_COEFFICIENT} (got "
            f"{model_config.mlp_coefficient}): this arm changes the final norm and nothing "
            f"else, so it must match the standard arm's MLP width.")
    dtype = _get_dtype(model_config.dtype)
    backbone = TransformerBackbone(
        num_heads=model_config.num_heads,
        model_dim=model_config.model_dim,
        num_layers=model_config.num_layers,
        mlp_dim=model_config.model_dim * model_config.mlp_coefficient,
        dropout_rate=model_config.dropout_rate,
        positional_encoding=model_config.positional_encoding,
        max_seq_len=model_config.max_seq_len,
        activation=model_config.activation,
        dtype=dtype,
        final_norm=False,
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


def verify_no_final_norm(params, verbose=True):
    """Prove from the constructed weights that the backbone has no final LayerNorm and that
    every block still has its pre-norms. Raises otherwise.

    Params-derived, not config-derived, for the same reason as `verify_no_mlp`: a flag that
    was silently dropped would leave the config saying one thing and the model doing another.
    """
    backbone_params = params["backbone"]
    blocks = sorted(k for k in backbone_params if k.startswith("block_"))
    top_level_norms = [k for k in backbone_params if k.startswith("LayerNorm")]
    blocks_missing_norms = [b for b in blocks
                            if not any(k.startswith("LayerNorm") for k in backbone_params[b])]
    if verbose:
        print("=== No-final-norm verification (derived from actual constructed weights) ===")
        print(f"  blocks found: {len(blocks)}")
        print(f"  LayerNorms that are direct children of the backbone: {top_level_norms}")
        print(f"  blocks without any LayerNorm: {blocks_missing_norms}")
    if top_level_norms:
        raise ValueError(f"Backbone still has a final LayerNorm: {top_level_norms}. "
                         f"final_norm=False was not applied.")
    if blocks_missing_norms:
        raise ValueError(f"Blocks lost their pre-norms: {blocks_missing_norms}. Only the "
                         f"final LayerNorm is meant to be removed.")
    if verbose:
        print("  OK -- no final LayerNorm; every block keeps its pre-norms.\n")


def nofinalnorm_pretrain_key(cfg, data_cfg):
    """`knowledge_injection.pretrain_key` plus `arm="no_final_norm"`. A separate function
    for the same reason as `mlpfree_pretrain_key`: the shared key function is used by every
    existing driver and must not change."""
    pretrain_for_key = replace(cfg.pretrain, max_eval_people=0, eval_interval=0, target_acc=0.0)
    partition_hash = (file_hash(cfg.partition_path)
                       if cfg.partition_path and os.path.exists(cfg.partition_path) else None)
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="pretrain",
        pretrain=pretrain_for_key, seed=cfg.seed,
        population=[cfg.num_a, cfg.num_ballast, cfg.num_b, cfg.num_c],
        c_half=cfg.c_half, partition_seed=cfg.partition_seed, partition_hash=partition_hash,
        arm=ARM)


def n_params(params):
    return int(sum(x.size for x in jax.tree.leaves(params)))
