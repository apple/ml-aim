# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import flax.linen as jax_nn
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn as mlx_nn
from torch import nn as torch_nn

from aim.v1 import utils
from aim.v2.jax import models as jax_models
from aim.v2.mlx import models as mlx_models
from aim.v2.torch import models as torch_models

MODELS = (
    "aimv2_base",
    "aimv2_large",
    "aimv2_huge",
    "aimv2_1B",
    "aimv2_3B",
    "aimv2_large_native",
)


class TestMLXBackend:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_image_encoders(self, model_name: str, img: np.ndarray):
        torch_model = getattr(torch_models, model_name)()
        mlx_model = getattr(mlx_models, model_name)()

        weights = utils.torch_weights_to_mlx(torch_model.state_dict())
        mlx_model.load_weights(weights, strict=True)

        assert isinstance(torch_model, torch_nn.Module)
        assert isinstance(mlx_model, mlx_nn.Module)

        torch_out = torch_model(torch.from_numpy(img))
        torch_out = torch_out.detach().cpu().numpy()

        mlx_out = mlx_model(mx.array(img))
        mlx_out = np.array(mlx_out)

        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-3, atol=1e-3)

    def test_lit(self, img: np.ndarray, input_ids: np.ndarray):
        torch_model = torch_models.aimv2_large_lit()
        mlx_model = mlx_models.aimv2_large_lit()

        weights = utils.torch_weights_to_mlx(torch_model.state_dict())
        mlx_model.load_weights(weights, strict=True)

        torch_img_out, torch_text_out = torch_model(
            torch.from_numpy(img), torch.from_numpy(input_ids)
        )
        torch_img_out = torch_img_out.detach().cpu().numpy()
        torch_text_out = torch_text_out.detach().cpu().numpy()

        mlx_img_out, mlx_text_out = mlx_model(mx.array(img), mx.array(input_ids))
        mlx_img_out = np.array(mlx_img_out)
        mlx_text_out = np.array(mlx_text_out)

        np.testing.assert_allclose(mlx_img_out, torch_img_out, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(mlx_text_out, torch_text_out, rtol=1e-3, atol=1e-3)


class TestJAXBackend:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_image_encoder(self, model_name: str, img: np.ndarray):
        torch_model = getattr(torch_models, model_name)()
        jax_model = getattr(jax_models, model_name)()

        params = utils.torch_weights_to_jax(torch_model.state_dict())

        assert isinstance(torch_model, torch_nn.Module)
        assert isinstance(jax_model, jax_nn.Module)

        jax_out, _ = jax.jit(
            lambda x: jax_model.apply(params, img, mutable=["batch_stats"])
        )(jnp.array(img))
        jax_out = np.array(jax_out)

        torch_out = torch_model(torch.from_numpy(img))
        torch_out = torch_out.detach().cpu().numpy()

        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-3, atol=1e-3)

    def test_lit(self, img: np.ndarray, input_ids: np.ndarray):
        torch_model = torch_models.aimv2_large_lit()
        jax_model = jax_models.aimv2_large_lit()

        params = utils.torch_weights_to_jax(torch_model.state_dict())

        torch_img_out, torch_text_out = torch_model(
            torch.from_numpy(img), torch.from_numpy(input_ids)
        )
        torch_img_out = torch_img_out.detach().cpu().numpy()
        torch_text_out = torch_text_out.detach().cpu().numpy()

        (jax_img_out, jax_text_out), _ = jax.jit(
            lambda img, input_ids: jax_model.apply(
                params, img, input_ids, mutable=["batch_stats"]
            )
        )(jnp.array(img), jnp.array(input_ids))
        jax_img_out = np.array(jax_img_out)
        jax_text_out = np.array(jax_text_out)

        np.testing.assert_allclose(jax_img_out, torch_img_out, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jax_text_out, torch_text_out, rtol=1e-3, atol=1e-3)
