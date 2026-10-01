"""Transformer backbone with RoPE and KV cache support."""
from typing import Any, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


def compute_freqs_cis(
    dim: int, length: int, start_pos: int = 0, theta: float = 10000.0, dtype=jnp.float32
):
    """Compute sin/cos frequencies for rotary position embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(length) + start_pos
    freqs = jnp.outer(t, freqs)
    return jnp.sin(freqs).astype(dtype), jnp.cos(freqs).astype(dtype)


def apply_rotary_emb(x, sin, cos):
    """Apply rotary position embeddings.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        sin, cos: (seq_len, head_dim // 2)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return jnp.stack([y1, y2], axis=-1).reshape(x.shape)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and KV cache."""
    num_heads: int
    qkv_features: int
    dropout_rate: float
    positional_encoding: str = "rope"
    max_seq_len: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        x,
        mask=None,
        deterministic=True,
        init_cache=False,
        cache=None,
        decode_step=None,
        max_seq_len=None,
    ):
        B, L, E = x.shape
        head_dim = self.qkv_features // self.num_heads
        x = x.astype(self.dtype)

        q = nn.Dense(self.qkv_features, use_bias=False, dtype=self.dtype, name="query")(x)
        k = nn.Dense(self.qkv_features, use_bias=False, dtype=self.dtype, name="key")(x)
        v = nn.Dense(self.qkv_features, use_bias=False, dtype=self.dtype, name="value")(x)

        q = q.reshape(B, L, self.num_heads, head_dim)
        k = k.reshape(B, L, self.num_heads, head_dim)
        v = v.reshape(B, L, self.num_heads, head_dim)

        # Apply positional encoding
        if self.positional_encoding == "rope":
            start_pos = decode_step if decode_step is not None else 0
            sin, cos = compute_freqs_cis(head_dim, L, start_pos, dtype=self.dtype)
            q = apply_rotary_emb(q, sin, cos)
            k = apply_rotary_emb(k, sin, cos)
        elif self.positional_encoding == "learned":
            start_pos = decode_step if decode_step is not None else 0
            pos_emb = self.param(
                "pos_emb_attn",
                nn.initializers.normal(0.02),
                (self.max_seq_len, head_dim),
            )
            positions = jnp.arange(L) + start_pos
            pe = pos_emb[positions]  # (L, head_dim)
            q = q + pe[None, :, None, :]
            k = k + pe[None, :, None, :]

        # KV cache handling
        new_cache = None
        if init_cache:
            if max_seq_len is None:
                max_seq_len = L
            k_cache = jnp.zeros((B, max_seq_len, self.num_heads, head_dim), dtype=k.dtype)
            v_cache = jnp.zeros((B, max_seq_len, self.num_heads, head_dim), dtype=v.dtype)
            k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, 0, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, 0, 0, 0))
            new_cache = (k_cache, v_cache)
        elif cache is not None:
            k_cache, v_cache = cache
            if decode_step is not None:
                k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, decode_step, 0, 0))
                v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, decode_step, 0, 0))
                new_cache = (k_cache, v_cache)
                k = k_cache
                v = v_cache

        # Attention computation
        q = jnp.swapaxes(q, 1, 2)  # (B, H, L, D)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        scale = head_dim ** -0.5
        sim = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale

        if cache is not None and decode_step is not None:
            max_len = k.shape[2]
            keep = jnp.arange(max_len) <= decode_step
            sim = jnp.where(keep[None, None, None, :], sim, -1e9)
        else:
            # Causality is always enforced; an explicit mask can only restrict
            # attention further (e.g. padding), never re-enable future positions.
            keep = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None, :, :]
            if mask is not None:
                keep = keep & mask.astype(bool)
            sim = jnp.where(keep, sim, -1e9)

        attn = nn.softmax(sim, axis=-1).astype(self.dtype)
        # Sown for the attention-stability check (logging attention-to-name-tokens as a check,
        # not as patching infrastructure). Sowing the post-softmax, pre-dropout distribution:
        # that is the actual attention pattern, and dropout is 0 by house convention anyway.
        # `sow` into a non-mutable collection is a no-op -- it adds no parameters and does not
        # touch `attn` -- so this is inert unless a caller passes mutable=["intermediates"].
        # Pinned by tests/test_mlpfree_noop.py.
        self.sow("intermediates", "attn_weights", attn)
        if not deterministic and self.dropout_rate > 0.0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)

        out = jnp.matmul(attn, v)
        out = jnp.swapaxes(out, 1, 2).reshape(B, L, self.qkv_features)
        out = nn.Dense(E, dtype=self.dtype, name="out")(out)

        if init_cache or cache is not None:
            return out, new_cache
        return out


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block.

    `mlp_enabled=False` gives an attention-only block (`x = x + Attn(LN(x))`) for the
    MLP-free ablation: the pre-MLP LayerNorm and both Dense layers are not constructed at
    all, rather than constructed and zeroed. Skipping rather than zeroing keeps the
    parameter count honest -- occupancy divides by the real count, and ~16.8M dead MLP
    parameters would misreport it by 2.6x at 8L/512 -- and lets a params-derived check
    assert the MLP tensors *do not exist*, which is strictly stronger than asserting some
    tensors are numerically zero. The two are numerically identical here regardless: this
    block has unscaled residuals and the pre-MLP LayerNorm feeds nothing but the MLP, so a
    forced-zero MLP output leaves every downstream LayerNorm seeing the same residual
    stream. The flag lives here rather than on `ModelConfig` because `experiment_key` hashes
    `dataclasses.asdict(ModelConfig)`, so adding a config field -- even one defaulting to
    True -- would re-key every existing checkpoint.
    """
    num_heads: int
    model_dim: int
    mlp_dim: int
    dropout_rate: float
    positional_encoding: str = "rope"
    max_seq_len: int = 512
    activation: str = "gelu"
    dtype: Any = jnp.float32
    mlp_enabled: bool = True

    @nn.compact
    def __call__(
        self, x, mask=None, deterministic=True,
        init_cache=False, cache=None, decode_step=None, max_seq_len=None,
    ):
        # Attention block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        attn_args = dict(
            mask=mask, deterministic=deterministic,
            init_cache=init_cache, cache=cache,
            decode_step=decode_step, max_seq_len=max_seq_len,
        )
        attn_out = CausalSelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            dropout_rate=self.dropout_rate,
            positional_encoding=self.positional_encoding,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )(y, **attn_args)

        new_cache = None
        if init_cache or cache is not None:
            attn_out, new_cache = attn_out

        x = x + attn_out

        # MLP block
        if self.mlp_enabled:
            y = nn.LayerNorm(dtype=self.dtype)(x)
            if self.activation == "gelu":
                act_fn = nn.gelu
            elif self.activation == "relu":
                act_fn = nn.relu
            else:
                act_fn = nn.gelu

            y = nn.Dense(self.mlp_dim, dtype=self.dtype)(y)
            y = act_fn(y)
            y = nn.Dense(self.model_dim, dtype=self.dtype)(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

            x = x + y

        if init_cache or cache is not None:
            return x, new_cache
        return x


class TransformerBackbone(nn.Module):
    """Stack of Transformer blocks with final LayerNorm.

    `final_norm=False` ends the backbone at the last block's residual instead of
    `LayerNorm(x)`, for the no-final-norm ablation: the head then reads an un-normalized
    residual stream. The per-block pre-norms are untouched. As with `mlp_enabled`, the
    LayerNorm is not constructed rather than constructed and bypassed, so a params-derived
    check can assert it does not exist, and the flag lives here rather than on
    `ModelConfig` so that existing checkpoints keep their keys.
    """
    num_heads: int
    model_dim: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    positional_encoding: str = "rope"
    max_seq_len: int = 512
    activation: str = "gelu"
    dtype: Any = jnp.float32
    mlp_enabled: bool = True  # False -> attention-only blocks; see TransformerBlock
    final_norm: bool = True   # False -> no LayerNorm after the last block; see docstring

    @nn.compact
    def __call__(
        self, x, mask=None, deterministic=True,
        init_cache=False, cache=None, decode_step=None, max_seq_len=None,
    ):
        new_caches = []
        for i in range(self.num_layers):
            layer_cache = cache[i] if cache is not None else None
            block_out = TransformerBlock(
                num_heads=self.num_heads,
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                positional_encoding=self.positional_encoding,
                max_seq_len=self.max_seq_len,
                activation=self.activation,
                dtype=self.dtype,
                mlp_enabled=self.mlp_enabled,
                name=f"block_{i}",
            )(
                x, mask=mask, deterministic=deterministic,
                init_cache=init_cache, cache=layer_cache,
                decode_step=decode_step, max_seq_len=max_seq_len,
            )

            if init_cache or cache is not None:
                x, new_cache = block_out
                new_caches.append(new_cache)
            else:
                x = block_out

        if self.final_norm:
            x = nn.LayerNorm(dtype=self.dtype)(x)

        if init_cache or cache is not None:
            return x, new_caches
        return x
