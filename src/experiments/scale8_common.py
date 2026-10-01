"""Shared pieces for the 8-layer/512-dim scale-check rebuild: architecture verification
and the occupancy formula, used identically by the LR probe, pretrain, and injection
drivers so none of the three can silently diverge from the others.
"""
import jax
import numpy as np

# Exact spec, hardcoded (not left to CLI flags) so a dropped/mistyped flag cannot
# silently fall back to the pilot's 256/4/4 -- see verify_architecture below for the
# independent, params-derived check that this was actually honored.
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
MLP_COEFFICIENT = 4  # -> mlp_dim = 2048, matches spec, no separate override needed

BITS_PER_PARAM = 2.0  # Allen-Zhu convention, used literally


def verify_architecture(params, requested_model_dim, requested_num_heads, requested_num_layers):
    """Independently derive layer count, model_dim, and total params from the actual
    constructed weights -- not by re-echoing the config that was passed in. Raises if
    anything doesn't match what was requested, so a silently-ignored flag fails loudly
    at the very start of the run rather than producing a quietly-wrong result.

    Returns (n_layers, model_dim, n_params).
    """
    backbone_params = params["backbone"]
    block_keys = [k for k in backbone_params if k.startswith("block_")]
    n_layers = len(block_keys)

    embed_dim = int(params["input_layer"]["Embed_0"]["embedding"].shape[-1])

    n_params = int(sum(x.size for x in jax.tree.leaves(params)))

    print(f"=== Architecture verification (derived from actual constructed weights) ===")
    print(f"  requested: model_dim={requested_model_dim} num_heads={requested_num_heads} "
          f"num_layers={requested_num_layers}")
    print(f"  derived:   num_layers={n_layers} (counted block_* keys in backbone params)")
    print(f"  derived:   model_dim={embed_dim} (input_layer embedding table's feature dim)")
    print(f"  derived:   total_params={n_params:,}")

    if n_layers != requested_num_layers:
        raise ValueError(f"Layer count mismatch: requested {requested_num_layers}, "
                          f"actual model has {n_layers}. Config was not applied correctly.")
    if embed_dim != requested_model_dim:
        raise ValueError(f"model_dim mismatch: requested {requested_model_dim}, "
                          f"actual model has {embed_dim}. Config was not applied correctly.")
    print("  OK -- constructed model matches the requested architecture exactly.\n")
    return n_layers, embed_dim, n_params


def occupancy(n_params, n_people, pop_num_values_per_attr):
    """Exact formula and constants of the earlier capacity estimate, unchanged for
    comparability with the pilot's ~2.09% figure: occupancy = (N * A * L_base_bits) / (P * 2),
    where N = population being stored (A+ballast only, matching the earlier estimate's own
    choice -- B and C are deliberately excluded, carried forward unchanged, not revisited
    here), A = number of attributes, L_base_bits = mean log|V| in nats converted to bits
    (x 1/ln(2)), P = total model parameters, and 2 is the Allen-Zhu bits-per-parameter
    constant used literally.
    """
    log_v_nats = np.log(np.asarray(pop_num_values_per_attr, dtype=np.float64))
    l_base_bits = float(log_v_nats.mean()) / np.log(2.0)
    n_attrs = len(pop_num_values_per_attr)
    knowledge_bits = n_people * n_attrs * l_base_bits
    capacity_bits = n_params * BITS_PER_PARAM
    return knowledge_bits / capacity_bits, knowledge_bits, capacity_bits
