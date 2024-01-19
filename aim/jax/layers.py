# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

__all__ = [
    "LayerNorm",
    "PatchEmbed",
    "ViTPreprocessor",
    "Attention",
    "MLP",
    "Block",
    "AverageLayers",
    "AttentionPoolingClassifier",
]


LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-6)  # already has `epsilon=1e-6`


class PatchEmbed(nn.Module):
    img_size: Union[int, Tuple[int, int]] = 224
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    in_chans: int = 3  # unused, for API compatibility
    norm_layer: Optional[Callable[..., nn.Module]] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        n = x.shape[0]
        x = x.transpose(0, 2, 3, 1)  # (N C H W) -> (N H W C)

        x = nn.Conv(
            self.embed_dim,
            kernel_size=self._patch_size,
            strides=self._patch_size,
            name="proj",
        )(x)
        x = x.reshape(n, -1, self.embed_dim)  # (N, H * W, F)
        if self.norm_layer is not None:
            x = self.norm_layer(name="norm")(x)
        return x

    @property
    def _patch_size(self) -> Tuple[int, int]:
        return (
            (self.patch_size, self.patch_size)
            if isinstance(self.patch_size, int)
            else self.patch_size
        )

    @property
    def grid_size(self) -> Tuple[int, int]:
        ims = (
            (self.img_size, self.img_size)
            if isinstance(self.img_size, int)
            else tuple(self.img_size)
        )
        ps = self._patch_size
        return ims[0] // ps[0], ims[1] // ps[1]

    @property
    def num_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]


class SinCosPosEmbed(nn.Module):
    cls_token: bool = False

    @nn.compact
    def __call__(self, h: int, w: int, embed_dim: int) -> jax.Array:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = jnp.arange(h, dtype=float)
        grid_w = jnp.arange(w, dtype=float)
        grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")
        grid = jnp.stack(grid, axis=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = jnp.concatenate([jnp.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jax.Array) -> jax.Array:
        omega = jnp.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = pos[:, None] * omega[None, :]  # (M, D/2), outer product

        emb_sin = jnp.sin(out)  # (M, D/2)
        emb_cos = jnp.cos(out)  # (M, D/2)

        emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb


class ViTPreprocessor(nn.Module):
    patchifier: PatchEmbed
    drop_patches: bool = False
    cls_token: bool = False

    @nn.compact
    def __call__(self, data: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        tokens = self.patchifier(data)
        if self.cls_token:
            cls_token = self.param(
                "cls_token",
                nn.initializers.zeros_init(),
                (1, 1, self.patchifier.embed_dim),
            )
            num_tokens = tokens.shape[0]
            _, num_queries, dim = cls_token.shape
            cls_token = jnp.broadcast_to(cls_token, (num_tokens, num_queries, dim))
            tokens = jnp.concatenate([cls_token, tokens], axis=1)

        B, N, D = tokens.shape

        h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        pos_embed = SinCosPosEmbed(self.cls_token)(h, w, embed_dim)

        tokens = tokens + pos_embed[None, :N]

        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, :1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = jnp.concatenate([cls_token, tokens], axis=1)

        return tokens


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False, **_: Any) -> jax.Array:
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.qk_scale or head_dim**-0.5
        attn = (q @ k.swapaxes(-2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop, deterministic=not training, name="attn_drop")(
            attn
        )

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = nn.Dense(self.dim, use_bias=self.use_bias, name="proj")(x)
        x = nn.Dropout(self.proj_drop, deterministic=not training, name="proj_drop")(x)
        return x


class MLP(nn.Module):
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable[[jax.Array], jax.Array] = nn.activation.gelu
    use_bias: bool = True
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        x = nn.Dense(hidden_features, use_bias=self.use_bias, name="fc1")(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop, deterministic=not training, name="drop1")(x)
        x = nn.Dense(out_features, use_bias=self.use_bias, name="fc2")(x)
        x = nn.Dropout(self.drop, deterministic=not training, name="drop2")(x)
        return x


class Block(nn.Module):
    dim: int
    attn_target: Callable[[bool, ...], nn.Module]
    mlp_hidden_dim: Optional[int] = None
    act_layer: Callable[[jax.Array], jax.Array] = nn.activation.gelu
    norm_layer: Callable[..., nn.Module] = LayerNorm
    ffn_dropout_rate: float = 0.0
    use_bias: bool = True
    name: Optional[str] = None

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # pre-norm
        x = x + self.attn_target(self.use_bias, name="attn")(
            self.norm_layer(name="norm_1")(x), mask=mask
        )

        mlp = MLP(
            in_features=self.dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.ffn_dropout_rate,
            use_bias=self.use_bias,
            name="mlp",
        )
        x = x + mlp(self.norm_layer(name="norm_2")(x))
        return x


class AverageLayers(nn.Module):
    layers: Sequence[int]
    reduce: bool = False

    @nn.compact
    def __call__(self, _: jax.Array, layer_features: List[jax.Array]) -> jax.Array:
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        feats = jnp.stack(layer_features, axis=-1).mean(axis=-1)

        return feats.mean(axis=1) if self.reduce else feats

    @property
    def max_block_id(self) -> int:
        return max(self.layers)


class AttentionPoolingClassifier(nn.Module):
    dim: int
    out_features: int
    num_heads: int = 12
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    num_queries: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, **_: Any) -> jax.Array:
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = self.qk_scale or head_dim**-0.5

        x = nn.BatchNorm(
            epsilon=1e-6,
            momentum=0.9,
            use_running_average=False,
            use_scale=False,
            use_bias=False,
            name="bn",
        )(x)

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_queries, self.dim),
        )
        cls_token = jnp.broadcast_to(cls_token, (B, self.num_queries, self.dim))

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).transpose(0, 2, 1, 3)
        k = (
            nn.Dense(self.dim, use_bias=self.qkv_bias, name="k")(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose(0, 2, 1, 3)
        )

        q = q * scale
        v = (
            nn.Dense(self.dim, use_bias=self.qkv_bias, name="v")(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose(0, 2, 1, 3)
        )

        attn = q @ k.swapaxes(-2, -1)
        attn = nn.softmax(attn, axis=-1)

        x_cls = (attn @ v).swapaxes(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(axis=1)

        out = nn.Dense(self.out_features, name="linear")(x_cls)
        return out, x_cls
