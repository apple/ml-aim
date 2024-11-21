# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest

import numpy as np


@pytest.fixture
def img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(size=(1, 3, 224, 224)).astype(np.float32))
