# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import math
from typing import Any, Literal, Optional, Tuple, Union

from aim.v2 import mixins
from aim.v2.torch import layers

import torch
from torch import nn
from torch.nn import functional as F

from aim.v1.torch import models

__all__ = [
    "AIMv2VisionEncoder",
    "AIMv2TextEncoder",
    "AIMv2LiT",
    "aimv2_base",
    "aimv2_large",
    "aimv2_huge",
    "aimv2_1B",
    "aimv2_3B",
    "aimv2_large_native",
    "aimv2_large_lit",
]


class AIMv2VisionEncoder(mixins.AIMv2VisionMixin, nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        embed_dim: int = 1024,
        mlp_hidden_dim: int = 2816,
        num_blocks: int = 24,
        num_heads: int = 8,
        num_channels: int = 3,
        head_num_heads: int = 8,
        head_num_queries: int = 1,
        head_average_pool: bool = True,
        head_linear_bias: bool = False,
        pos_embed_type: Literal["sincos", "absolute"] = "absolute",
        head_type: Optional[Literal["attention-pool"]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        norm_layer = functools.partial(layers.RMSNorm, eps=1e-5)
        patchifier = layers.PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.preprocessor = layers.ViTPreprocessor(
            patchifier,
            drop_patches=False,
            cls_token=False,
            pos_embed_type=pos_embed_type,
        )
        self.trunk = models.Transformer(
            attn_target=lambda use_bias: layers.Attention(
                dim=embed_dim, num_heads=num_heads, use_bias=use_bias
            ),
            ffn_target=layers.SwiGLUFFN,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_blocks=num_blocks,
            norm_layer=norm_layer,
            **kwargs,
        )
        if head_type == "attention-pool":
            self.head = layers.AttentionPoolingClassifier(
                embed_dim,
                out_features=embed_dim,
                num_heads=head_num_heads,
                num_queries=head_num_queries,
                use_batch_norm=False,
                qkv_bias=False,
                linear_bias=head_linear_bias,
                average_pool=head_average_pool,
            )
        else:
            self.head = nn.Identity()


class AIMv2TextEncoder(mixins.AIMv2TextMixin, nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_dim: int = 2048,
        num_blocks: int = 12,
        num_heads: int = 6,
        vocab_size: int = 49408,
        eos_token_id: int = 49407,
        max_context_length: int = 77,
        **kwargs: Any,
    ):
        super().__init__()
        norm_layer = functools.partial(layers.RMSNorm, eps=1e-5)
        self.preprocessor = layers.TextPreprocessor(
            vocab_size,
            embed_dim,
            max_context_length=max_context_length,
            eos_token_id=eos_token_id,
        )
        self.trunk = models.Transformer(
            attn_target=lambda use_bias: layers.Attention(
                dim=embed_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                is_causal=True,
            ),
            ffn_target=layers.SwiGLUFFN,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_blocks=num_blocks,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.head = layers.ExtractEOS()


class AIMv2LiT(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        projection_dim: int = 768,
        vision_embed_dim: int = 1024,
        vision_mlp_hidden_dim: int = 2816,
        vision_num_blocks: int = 24,
        vision_num_heads: int = 8,
        text_embed_dim: int = 768,
        text_mlp_embed_dim: int = 2048,
        text_num_blocks: int = 12,
        text_num_heads: int = 6,
        vocab_size: int = 49408,
        max_context_length: int = 77,
        eos_token_id: int = 49407,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.image_encoder = AIMv2VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            mlp_hidden_dim=vision_mlp_hidden_dim,
            num_blocks=vision_num_blocks,
            num_heads=vision_num_heads,
            pos_embed_type="absolute",
            head_type="attention-pool",
            head_num_heads=vision_num_heads,
            head_num_queries=1,
            head_linear_bias=True,
            head_average_pool=True,
            **kwargs,
        )
        self.text_encoder = AIMv2TextEncoder(
            embed_dim=text_embed_dim,
            mlp_hidden_dim=text_mlp_embed_dim,
            num_blocks=text_num_blocks,
            num_heads=text_num_heads,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            max_context_length=max_context_length,
        )

        self.image_projector = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projector = nn.Linear(text_embed_dim, projection_dim, bias=False)

        self.log_logit_scale = nn.Parameter(
            torch.full([], fill_value=math.log(1.0 / init_temperature))
        )
        self.max_log_logit_scale = math.log(max_logit_scale)

    def forward(
        self,
        input_pixels: torch.Tensor,
        input_ids: torch.Tensor,
        output_features: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        img_embeds = self.encode_image(input_pixels, output_features=False)
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        text_embeds = self.encode_text(input_ids, output_features=False)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        logit_scale = self.log_logit_scale.clamp(0.0, self.max_log_logit_scale).exp()
        logits_per_text = (logit_scale * text_embeds) @ img_embeds.t()
        logits_per_image = logits_per_text.t()

        if output_features:
            return logits_per_image, logits_per_text, img_embeds, text_embeds
        return logits_per_image, logits_per_text

    def encode_image(
        self,
        input_pixels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        out = self.image_encoder(
            input_pixels, mask=mask, output_features=output_features
        )
        out = self.image_projector(out)
        return out

    def encode_text(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        out = self.text_encoder(input_ids, mask=mask, output_features=output_features)
        out = self.text_projector(out)
        return out


def aimv2_base(
    img_size: Union[int, Tuple[int, int]] = 224,
    **kwargs: Any,
) -> AIMv2VisionEncoder:
    return AIMv2VisionEncoder(
        img_size=img_size,
        patch_size=14,
        embed_dim=768,
        mlp_hidden_dim=2048,
        num_blocks=12,
        num_heads=6,
        pos_embed_type="absolute",
        head_type=None,
        **kwargs,
    )


def aimv2_large(
    img_size: Union[int, Tuple[int, int]] = 224,
    **kwargs: Any,
) -> AIMv2VisionEncoder:
    return AIMv2VisionEncoder(
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        mlp_hidden_dim=2816,
        num_blocks=24,
        num_heads=8,
        pos_embed_type="absolute",
        head_type=None,
        **kwargs,
    )


def aimv2_huge(
    img_size: Union[int, Tuple[int, int]] = 224,
    **kwargs: Any,
) -> AIMv2VisionEncoder:
    return AIMv2VisionEncoder(
        img_size=img_size,
        patch_size=14,
        embed_dim=1536,
        mlp_hidden_dim=4096,
        num_blocks=24,
        num_heads=12,
        pos_embed_type="absolute",
        head_type=None,
        **kwargs,
    )


def aimv2_1B(
    img_size: Union[int, Tuple[int, int]] = 224,
    **kwargs: Any,
) -> AIMv2VisionEncoder:
    return AIMv2VisionEncoder(
        img_size=img_size,
        patch_size=14,
        embed_dim=2048,
        mlp_hidden_dim=5632,
        num_blocks=24,
        num_heads=16,
        pos_embed_type="absolute",
        head_type=None,
        **kwargs,
    )


def aimv2_3B(
    img_size: Union[int, Tuple[int, int]] = 224,
    **kwargs: Any,
) -> AIMv2VisionEncoder:
    return AIMv2VisionEncoder(
        img_size=img_size,
        patch_size=14,
        embed_dim=3072,
        mlp_hidden_dim=8192,
        num_blocks=24,
        num_heads=24,
        pos_embed_type="absolute",
        head_type=None,
        **kwargs,
    )


def aimv2_large_native(**kwargs: Any) -> AIMv2VisionEncoder:
    _ = kwargs.pop("img_size", None)
    return AIMv2VisionEncoder(
        patch_size=14,
        embed_dim=1024,
        mlp_hidden_dim=2816,
        num_blocks=24,
        num_heads=8,
        pos_embed_type="sincos",
        head_type=None,
        **kwargs,
    )


def aimv2_large_lit(
    img_size: Union[int, Tuple[int, int]] = 224,
    patch_size: Union[int, Tuple[int, int]] = 14,
    vocab_size: int = 49408,
    eos_token_id: int = 49407,
    max_context_length: int = 77,
    **kwargs: Any,
) -> AIMv2LiT:
    return AIMv2LiT(
        img_size,
        patch_size=patch_size,
        projection_dim=768,
        vision_embed_dim=1024,
        vision_mlp_hidden_dim=2816,
        vision_num_blocks=24,
        vision_num_heads=8,
        text_embed_dim=768,
        text_mlp_embed_dim=2048,
        text_num_blocks=12,
        text_num_heads=6,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        max_context_length=max_context_length,
        **kwargs,
    )
