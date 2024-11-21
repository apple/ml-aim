# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from aim.v1.jax.layers import (
    Attention,
    PatchEmbed,
    ViTPreprocessor,
    AttentionPoolingClassifier,
)

__all__ = [
    "Identity",
    "TextPreprocessor",
    "ExtractEOS",
    "RMSNorm",
    "SwiGLUFFN",
    "AttentionPoolingClassifier",
    "ViTPreprocessor",
    "PatchEmbed",
    "Attention",
]


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class TextPreprocessor(nn.Module):
    vocab_size: int
    embed_dim: int
    max_context_length: int = 77
    eos_token_id: int = 49407

    @nn.compact
    def __call__(self, input_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        _, N = input_ids.shape
        max_len = min(N, self.max_context_length)
        eos_token_mask = input_ids == self.eos_token_id

        tokens = nn.Embed(self.vocab_size, self.embed_dim, name="text_embedding")(
            input_ids
        )
        positional_embedding = self.param(
            "positional_embedding",
            nn.initializers.zeros_init(),
            (self.max_context_length, self.embed_dim),
        )
        tokens = tokens[:, :max_len] + positional_embedding[None, :max_len]
        return tokens, eos_token_mask


class ExtractEOS(nn.Module):
    @nn.compact
    def __call__(self, tokens: jax.Array, eos_token_mask: jax.Array) -> jax.Array:
        B, N, D = tokens.shape
        assert eos_token_mask.shape == (B, N)
        eos_token = tokens.at[jnp.arange(B), eos_token_mask.argmax(axis=-1)].get()
        return eos_token


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones_init(), (dim,))
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        output = output * scale
        return output

    def _norm(self, x: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps)


class SwiGLUFFN(nn.Module):
    in_features: int
    hidden_features: int
    use_bias: bool = True
    norm_layer: Optional[Callable[..., nn.Module]] = None
    # unused, but passed to `ffn_target` in `Block`
    act_layer: Any = None
    drop: Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x1 = nn.Dense(self.hidden_features, use_bias=self.use_bias, name="fc1")(x)
        x2 = nn.Dense(self.hidden_features, use_bias=self.use_bias, name="fc3")(x)
        x = nn.silu(x1) * x2
        if self.norm_layer is not None:
            x = self.norm_layer(name="norm_layer")(x)
        x = nn.Dense(self.in_features, use_bias=self.use_bias, name="fc2")(x)
        return x
