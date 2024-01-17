# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax

from .. import mixins
from . import layers

__all__ = ["Transformer", "AIM", "aim_600M", "aim_1B", "aim_3B", "aim_7B"]


class Transformer(nn.Module):
    attn_target: Callable[[bool], nn.Module]
    embed_dim: int
    num_blocks: int
    post_transformer_layer: Optional[nn.Module] = None
    norm_layer: Callable[..., nn.Module] = layers.LayerNorm
    mlp_ratio: int = 4
    mlp_hidden_dim: Optional[int] = None
    ffn_dropout_rate: float = 0.0
    use_bias: bool = False
    post_trunk_norm: bool = True

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        mask: Optional[jax.Array] = None,
        max_block_id: Optional[int] = -1,
        return_features: bool = False,
    ) -> Union[Tuple[jax.Array, List[jax.Array]], List[jax.Array]]:
        if self.mlp_hidden_dim is None:
            mlp_hidden_dim = int(self.mlp_ratio * self.embed_dim)
        else:
            mlp_hidden_dim = self.mlp_hidden_dim

        # only evaluate up to the max block id
        if max_block_id is None:
            assert (
                self.post_transformer_layer is not None
            ), "Unable to determine the max block id."
            max_block_id = self.post_transformer_layer.max_block_id

        features = []
        for blk_id in range(self.num_blocks):
            tokens = layers.Block(
                dim=self.embed_dim,
                attn_target=self.attn_target,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_layer=self.norm_layer,
                ffn_dropout_rate=self.ffn_dropout_rate,
                use_bias=self.use_bias,
                name=f"layers_{blk_id}",
            )(tokens, mask=mask)
            features.append(tokens)

            if blk_id == max_block_id:
                break

        if return_features:
            return features

        if self.post_trunk_norm:
            tokens = self.norm_layer(name="post_trunk_norm")(tokens)

        if self.post_transformer_layer is not None:
            tokens = self.post_transformer_layer(tokens, layer_features=features)

        return tokens, features


class AIM(mixins.AIMMixin, nn.Module):
    preprocessor: nn.Module
    trunk: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        max_block_id: Optional[int] = -1,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return self.forward(x, mask=mask, max_block_id=max_block_id)


def _get_attention_target(dim: int, num_heads: int) -> Callable[[bool], nn.Module]:
    def callback(use_bias: bool, name: str) -> nn.Module:
        return layers.Attention(
            dim=dim, num_heads=num_heads, use_bias=use_bias, name=name
        )

    return callback


def _aim(
    img_size: Union[int, Tuple[int, int]],
    patch_size: Union[int, Tuple[int, int]],
    embed_dim: int,
    num_blocks: int,
    num_heads: int,
    num_channels: int = 3,
    probe_layers: Union[int, Tuple[int, ...]] = 6,
    num_classes: int = 1000,
    **kwargs: Any,
) -> AIM:
    # common
    norm_layer = layers.LayerNorm

    # preprocessor
    patchifier = layers.PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=num_channels,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )
    preprocessor = layers.ViTPreprocessor(
        patchifier, drop_patches=False, cls_token=False
    )

    # trunk
    if isinstance(probe_layers, int):
        probe_layers = tuple(range(num_blocks - probe_layers, num_blocks))
    assert all(layer >= 0 for layer in probe_layers), probe_layers

    attn_target = _get_attention_target(dim=embed_dim, num_heads=num_heads)
    post_transform_layer = layers.AverageLayers(probe_layers, reduce=False)
    trunk = Transformer(
        attn_target,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        norm_layer=norm_layer,
        post_transformer_layer=post_transform_layer,
        **kwargs,
    )

    # head
    head = layers.AttentionPoolingClassifier(
        dim=embed_dim,
        out_features=num_classes,
        num_heads=num_heads,
        qkv_bias=False,
        qk_scale=None,
        num_queries=1,
    )

    return AIM(preprocessor, trunk, head)


def aim_600M(img_size: Union[int, Tuple[int, int]] = 224, **kwargs: Any) -> AIM:
    return _aim(
        img_size=img_size,
        patch_size=14,
        embed_dim=1536,
        num_blocks=24,
        num_heads=12,
        **kwargs,
    )


def aim_1B(img_size: Union[int, Tuple[int, int]] = 224, **kwargs: Any) -> AIM:
    return _aim(
        img_size=img_size,
        patch_size=14,
        embed_dim=2048,
        num_blocks=24,
        num_heads=16,
        **kwargs,
    )


def aim_3B(
    img_size: Union[int, Tuple[int, int]] = 224, patch_size: int = 14, **kwargs: Any
) -> AIM:
    return _aim(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=3072,
        num_blocks=24,
        num_heads=24,
        **kwargs,
    )


def aim_7B(
    img_size: Union[int, Tuple[int, int]] = 224, patch_size: int = 14, **kwargs: Any
) -> AIM:
    return _aim(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=4096,
        num_blocks=32,
        num_heads=32,
        **kwargs,
    )
