# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Optional, Union, Tuple

__all__ = ["AIMv2VisionMixin", "AIMv2TextMixin"]

ArrayLike = Any
Module = Callable[..., Any]


class AIMv2VisionMixin:
    preprocessor: Module
    trunk: Module
    head: Module

    def forward(
        self,
        input_pixels: ArrayLike,
        mask: Optional[ArrayLike] = None,
        output_features: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, Tuple[ArrayLike, ...]]]:
        x = self.preprocessor(input_pixels)
        x, features = self.trunk(x, mask=mask)
        x = self.head(x)
        return (x, tuple(features)) if output_features else x


class AIMv2TextMixin:
    preprocessor: Module
    trunk: Module
    head: Module

    def forward(
        self,
        input_ids: ArrayLike,
        mask: Optional[ArrayLike] = None,
        output_features: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, Tuple[ArrayLike, ...]]]:
        x, eos_token_mask = self.preprocessor(input_ids)
        x, features = self.trunk(x, mask=mask)
        x = self.head(x, eos_token_mask=eos_token_mask)
        return (x, tuple(features)) if output_features else x
