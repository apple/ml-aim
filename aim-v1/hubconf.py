# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any

from torch import nn

from aim.v1 import utils

dependencies = ["torch", "huggingface_hub"]


def aim_600M(*args: Any, **kwargs: Any) -> nn.Module:
    return utils.load_pretrained("aim-600M-2B-imgs", backend="torch", *args, **kwargs)


def aim_1B(*args: Any, **kwargs: Any) -> nn.Module:
    return utils.load_pretrained("aim-1B-5B-imgs", backend="torch", *args, **kwargs)


def aim_3B(*args: Any, **kwargs: Any) -> nn.Module:
    return utils.load_pretrained("aim-3B-5B-imgs", backend="torch", *args, **kwargs)


def aim_7B(*args: Any, **kwargs: Any) -> nn.Module:
    return utils.load_pretrained("aim-7B-5B-imgs", backend="torch", *args, **kwargs)
