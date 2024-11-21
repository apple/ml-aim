# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pathlib

import pytest

import flax.linen as jax_nn
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import torch
from mlx import nn as mlx_nn
from torch import nn as torch_nn

from aim.v1 import constants, utils
from aim.v1.jax import models as jax_models


@pytest.mark.parametrize(
    "model", ["aim-600M-2B-imgs", "aim-1B-5B-imgs", "aim-3B-5B-imgs", "aim-7B-5B-imgs"]
)
def test_mlx_backend(model: str, img: np.ndarray, tmp_path: pathlib.Path):
    torch_model = utils.load_pretrained(model, backend="torch", pretrained=False)

    mlx_model = utils.load_pretrained(model, backend="mlx", pretrained=False)
    weights = utils.torch_weights_to_mlx(torch_model.state_dict())
    mlx_model.load_weights(weights, strict=True)

    assert isinstance(torch_model, torch_nn.Module)
    assert isinstance(mlx_model, mlx_nn.Module)

    torch_out = torch_model(torch.from_numpy(img))
    torch_out = torch_out.detach().cpu().numpy()

    mlx_out = mlx_model(mx.array(img))
    mlx_out = np.array(mlx_out)

    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "model", ["aim-600M-2B-imgs", "aim-1B-5B-imgs", "aim-3B-5B-imgs", "aim-7B-5B-imgs"]
)
def test_jax_backend(model: str, img: np.ndarray):
    torch_model = utils.load_pretrained(model, backend="torch", pretrained=False)

    jax_model, _ = utils.load_pretrained(model, backend="jax", pretrained=False)
    params = utils.torch_weights_to_jax(torch_model.state_dict())

    assert isinstance(torch_model, torch_nn.Module)
    assert isinstance(jax_model, jax_nn.Module)

    jax_out, _batch_stats = jax_model.apply(
        params, jnp.array(img), mutable=["batch_stats"]
    )
    jax_out = np.array(jax_out)

    torch_out = torch_model(torch.from_numpy(img))
    torch_out = torch_out.detach().cpu().numpy()

    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "max_block_id,probe_layers", [(None, "last"), (None, "best"), (2, "last")]
)
def test_max_block_id(img: np.ndarray, max_block_id: int, probe_layers: str):
    arch = "aim-600M-2B-imgs"
    model = utils.load_pretrained(
        arch, backend="torch", probe_layers=probe_layers, pretrained=False
    )
    inp = torch.from_numpy(img)

    feats = model.extract_features(inp, max_block_id=max_block_id)

    if max_block_id is None:
        max_block_id = model.trunk.post_transformer_layer.max_block_id
        if probe_layers == "best":
            assert max_block_id == max(constants.LAYERS_BEST[arch])
        else:
            assert max_block_id == len(model.trunk.blocks) - 1

    assert isinstance(feats, list)
    assert len(feats) == (max_block_id + 1)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_jax_dtype(dtype: jnp.dtype):
    model = jax_models.aim_600M()
    inp = jnp.ones((1, 224, 224, 3), dtype=dtype)

    params = model.init(jax.random.PRNGKey(0), inp)
    params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)

    logits, _batch_stats = model.apply(params, inp, mutable=["batch_stats"])

    assert logits.dtype == dtype, (logits.dtype, dtype)
