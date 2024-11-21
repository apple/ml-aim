# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import numpy as np
import pytest
from transformers import CLIPTokenizer


@pytest.fixture
def img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(size=(1, 3, 224, 224)).astype(np.float32))


@pytest.fixture()
def input_ids() -> np.ndarray:
    tokenizer = CLIPTokenizer.from_pretrained("apple/aimv2-large-patch14-224-lit")
    text = ["This is a testing string: Hello world!"]
    input_ids = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="np",
        truncation=True,
        padding=True,
    )
    return input_ids["input_ids"]
