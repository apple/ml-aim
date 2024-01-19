# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# --------------------------------------------------
# References:
# https://github.com/facebookresearch/deit/blob/main/cait_models.py
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# https://github.com/facebookresearch/ImageBind
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

__all__ = [
    "LayerNorm",
    "SinCosPosEmbed",
    "PatchEmbed",
    "ViTPreprocessor",
    "Attention",
    "PrefixCausalAttention",
    "LoraAttention",
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

    def forward(self, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = torch.concatenate(
                [torch.zeros([1, embed_dim]), pos_embed], dim=0
            )
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int, pos: torch.Tensor
    ) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
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
            nn.Parameter(torch.zeros(1, 1, self.patchifier.embed_dim))
            if cls_token
            else None
        )
        self.pos_embed = SinCosPosEmbed(cls_token)
        self.drop_patches = drop_patches

        self.initialize_weights()

    def initialize_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens = self.patchifier(data)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        B, N, D = tokens.shape

        h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        pos_embed = self.pos_embed(h, w, embed_dim)
        pos_embed = pos_embed.to(tokens.device)

        tokens = tokens + pos_embed[None, :N]

        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, :1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = torch.cat([cls_token, tokens], dim=1)

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

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrefixCausalAttention(Attention):
    def __init__(self, *args: Any, num_patches: int = 256, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "attn_mask",
            torch.ones(1, num_patches, num_patches, dtype=torch.bool).tril(diagonal=0),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mask = kwargs.get("mask", None)
        assert mask is not None, "A mask is required for the PrefixLM Causal Attention."
        B, N, C = x.shape

        prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()
        attn_mask = self.attn_mask.clone().expand(B, -1, -1)
        attn_mask = torch.logical_or(attn_mask, prefix_mask)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LoraAttention(Attention):
    def __init__(self, dim: int, *args: Any, lora_rank: int = 8, **kwargs: Any):
        import loralib

        super().__init__(dim, *args, **kwargs)
        self.qkv = loralib.MergedLinear(
            dim,
            dim * 3,
            bias=kwargs.get("qkv_bias", False),
            r=lora_rank,
            enable_lora=[True, False, True],
        )
        self.proj = loralib.Linear(
            dim, dim, bias=kwargs.get("use_bias", True), r=lora_rank
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # pre-norm
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AverageLayers(nn.Module):
    def __init__(self, layers: Sequence[int], reduce: bool = False):
        super().__init__()
        self.layers = layers
        self.reduce = reduce

    def forward(
        self, _: torch.Tensor, layer_features: List[torch.Tensor]
    ) -> torch.Tensor:
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        feats = torch.stack(layer_features, dim=-1).mean(dim=-1)

        return feats.mean(dim=1) if self.reduce else feats

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

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features)
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out, x_cls
