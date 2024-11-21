# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from aim.v1.torch.layers import (
    Attention,
    AttentionPoolingClassifier,
    PatchEmbed,
    ViTPreprocessor,
)

__all__ = [
    "TextPreprocessor",
    "ExtractEOS",
    "RMSNorm",
    "SwiGLUFFN",
    "AttentionPoolingClassifier",
    "ViTPreprocessor",
    "PatchEmbed",
    "Attention",
]

RMSNorm = nn.RMSNorm


class TextPreprocessor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_context_length: int = 77,
        eos_token_id: int = 49407,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_context_length, embed_dim)
        )
        self.max_context_length = max_context_length
        self.eos_token_id = eos_token_id

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, N = input_ids.shape
        max_len = min(N, self.max_context_length)
        eos_token_mask = input_ids == self.eos_token_id
        tokens = self.text_embedding(input_ids)
        tokens = tokens[:, :max_len] + self.positional_embedding[:max_len].unsqueeze(0)
        return tokens, eos_token_mask


class ExtractEOS(nn.Module):
    def forward(
        self, tokens: torch.Tensor, eos_token_mask: torch.Tensor
    ) -> torch.Tensor:
        B, _, D = tokens.shape
        eos_token_mask = torch.argmax(eos_token_mask.float(), dim=-1)
        assert eos_token_mask.shape == (B,)
        eos_token_mask = eos_token_mask.reshape(B, 1, 1).expand(B, 1, D)
        eos_token = torch.gather(tokens, 1, eos_token_mask)
        eos_token = eos_token.squeeze(1)
        return eos_token


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        use_bias: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        **_: Any,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=use_bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.norm_layer = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.norm_layer(x)
        x = self.fc2(x)
        return x
