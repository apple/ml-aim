# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "LayerNorm",
    "SinCosPosEmbed",
    "PatchEmbed",
    "ViTPreprocessor",
    "Attention",
    "MLP",
    "Block",
    "AverageLayers",
    "AttentionPoolingClassifier",
]


LayerNorm = functools.partial(nn.LayerNorm, eps=1e-6)


class SinCosPosEmbed(nn.Module):
    def __init__(self, cls_token: bool = False):
        super().__init__()
        self.cls_token = cls_token

    def __call__(self, h: int, w: int, embed_dim: int) -> mx.array:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = mx.arange(h, dtype=mx.float32)
        grid_h = mx.repeat(grid_h[None, :], repeats=w, axis=0)
        grid_w = mx.arange(w, dtype=mx.float32)
        grid_w = mx.repeat(grid_w[:, None], repeats=h, axis=1)
        grid = mx.stack([grid_h, grid_w], axis=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = mx.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = mx.concatenate([mx.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: mx.array) -> mx.array:
        omega = mx.arange(embed_dim // 2, dtype=mx.float32)
        omega /= embed_dim / 2.0
        omega = mx.array(1.0 / 10000) ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = pos[:, None] * omega[None, :]  # (M, D/2), outer product

        emb_sin = mx.sin(out)  # (M, D/2)
        emb_cos = mx.cos(out)  # (M, D/2)

        emb = mx.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )

        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)  # (N C H W) -> (N H W C)
        x = self.proj(x).flatten(1, 2)  # (N, PS * PS, EMB)
        x = self.norm(x)
        return x


class ViTPreprocessor(nn.Module):
    def __init__(
        self,
        patchifier: PatchEmbed,
        drop_patches: bool = False,
        cls_token: bool = False,
    ):
        super().__init__()

        self.patchifier = patchifier
        self.cls_token = (
            mx.zeros((1, 1, self.patchifier.embed_dim)) if cls_token else None
        )
        self.pos_embed = SinCosPosEmbed(cls_token)
        self.drop_patches = drop_patches

    def __call__(self, data: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        tokens = self.patchifier(data)
        if self.cls_token is not None:
            num_tokens = tokens.shape[0]
            _, num_queries, dim = self.cls_token.shape
            cls_token = mx.broadcast_to(self.cls_token, (num_tokens, num_queries, dim))
            tokens = mx.concatenate([cls_token, tokens], axis=1)

        B, N, D = tokens.shape

        h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        pos_embed = self.pos_embed(h, w, embed_dim)

        tokens = tokens + pos_embed[None, :N]

        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, :1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = mx.concatenate([cls_token, tokens], axis=1)

        return tokens


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compatibility with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: mx.array, **_: Any) -> mx.array:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.swapaxes(3, 2)) * self.scale
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        use_bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)
        self.drop = nn.Dropout(drop)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable[[bool], nn.Module],
        mlp_hidden_dim: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = LayerNorm,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = True,
    ):
        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"

        super().__init__()
        self.attn = attn_target(use_bias)

        self.norm_1 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
            use_bias=use_bias,
        )
        self.norm_2 = norm_layer(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # pre-norm
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AverageLayers(nn.Module):
    def __init__(self, layers: Sequence[int], reduce: bool = False):
        super().__init__()
        self.layers = layers
        self.reduce = reduce

    def __call__(self, _: mx.array, layer_features: List[mx.array]) -> mx.array:
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        feats = mx.stack(layer_features, axis=-1).mean(axis=-1)

        return feats.mean(axis=1) if self.reduce else feats

    @property
    def max_block_id(self) -> int:
        return max(self.layers)


class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = mx.random.normal((1, num_queries, dim)) * 0.02

        self.linear = nn.Linear(dim, out_features)
        self.bn = nn.BatchNorm(dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def __call__(self, x: mx.array, **_: Any) -> mx.array:
        B, N, C = x.shape

        x = self.bn(x)
        _, num_queries, dim = self.cls_token.shape
        cls_token = mx.broadcast_to(self.cls_token, (B, num_queries, dim))

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).transpose(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose(0, 2, 1, 3)
        )

        attn = q @ k.swapaxes(-2, -1)
        attn = mx.softmax(attn, axis=-1)

        x_cls = (attn @ v).swapaxes(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(axis=1)

        out = self.linear(x_cls)
        return out
