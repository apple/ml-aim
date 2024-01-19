# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["AIMMixin"]

ArrayLike = Any
Module = Callable[..., Any]


class AIMMixin:
    preprocessor: Module
    trunk: Module
    head: Module

    def forward(
        self,
        x: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
    ) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:

        x = self.preprocessor(x, mask=mask)
        x, feats = self.trunk(x, mask=mask, max_block_id=max_block_id)
        logits, descriptor = self.head(x, mask=mask)

        return logits, descriptor

    def extract_features(
        self,
        x: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
    ) -> List[ArrayLike]:
        x = self.preprocessor(x, mask=mask)
        feats = self.trunk(
            x, mask=mask, max_block_id=max_block_id, return_features=True
        )
        return feats
