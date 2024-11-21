# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, Any, Tuple, Optional, Union
from huggingface_hub import hf_hub_download

__all__ = ["load_pretrained"]


MODELS = (
    # 224
    "aimv2-large-patch14-224",
    "aimv2-huge-patch14-224",
    "aimv2-1B-patch14-224",
    "aimv2-3B-patch14-224",
    # 336
    "aimv2-large-patch14-336",
    "aimv2-huge-patch14-336",
    "aimv2-1B-patch14-336",
    "aimv2-3B-patch14-336",
    # 448
    "aimv2-large-patch14-448",
    "aimv2-huge-patch14-448",
    "aimv2-1B-patch14-448",
    "aimv2-3B-patch14-448",
    # distilled
    "aimv2-large-patch14-224-distilled",
    "aimv2-large-patch14-336-distilled",
    # native
    "aimv2-large-patch14-native",
    # lit
    "aimv2-large-patch14-224-lit",
)
ModelName = Literal[MODELS]


def _get_weights_fname(backend: Literal["torch", "jax", "mlx"]) -> str:
    if backend == "torch":
        return "model.safetensors"
    if backend == "jax":
        return "flax_model.msgpack"
    if backend == "mlx":
        return "mlx_model.safetensors"
    raise ValueError(
        f"Invalid backend {backend}. Valid backends are: ('torch', 'jax', 'mlx')."
    )


def _get_model_func_and_img_size(model_name: ModelName) -> Tuple[str, Optional[int]]:
    if model_name.endswith("-lit"):
        # e.g., aimv2-{model_size}-patch14-{img_size}-lit
        prefix, model_size, _, img_size, _ = model_name.split("-")
        return f"{prefix}_{model_size}_lit", int(img_size)

    model_name = model_name.removesuffix("-distilled")
    # e.g., aimv2-{model_size}-patch14-{img_size}
    prefix, model_size, _, img_size = model_name.split("-")

    if img_size == "native":
        return f"{prefix}_{model_size}_native", None
    return f"{prefix}_{model_size}", int(img_size)


def _load_torch(model_name: ModelName, weights_path: str) -> Any:
    from aim.v2.torch import models
    from safetensors.torch import load_file

    model_func, img_size = _get_model_func_and_img_size(model_name)
    model_func = getattr(models, model_func)
    model = model_func(img_size=img_size)

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_jax(model_name: ModelName, weights_path: str) -> Tuple[Any, Any]:
    from aim.v2.jax import models
    from flax import serialization

    model_func, img_size = _get_model_func_and_img_size(model_name)
    model_func = getattr(models, model_func)
    model = model_func(img_size=img_size)

    with open(weights_path, "rb") as fin:
        params = serialization.msgpack_restore(fin.read())
    return model, params


def _load_mlx(model_name: ModelName, weights_path: str) -> Tuple[Any, Any]:
    from aim.v2.mlx import models

    model_func, img_size = _get_model_func_and_img_size(model_name)
    model_func = getattr(models, model_func)
    model = model_func(img_size=img_size)

    model = model.load_weights(weights_path, strict=True)
    return model


def load_pretrained(
    model_name: ModelName,
    backend: Literal["torch", "jax", "mlx"] = "torch",
    **kwargs: Any,
) -> Union[Any, Tuple[Any, Any]]:
    """Load pre-trained AIMv2 model.

    Args:
        model_name: Name of the model.
        backend: Compute backend.
        kwargs: Keyword arguments for :func:`~huggingface_hub.hf_hub_download`.

    Returns:
        If ``backend = 'jax'``, returns the model definition and the weights.
        Otherwise, returns the pre-trained model containing the weights.
    """
    if model_name not in MODELS:
        raise ValueError(f"Invalid model: {model_name}. Valid models are: {MODELS}.")

    fname = _get_weights_fname(backend)
    weights_path = hf_hub_download(f"apple/{model_name}", filename=fname, **kwargs)

    if backend == "torch":
        model = _load_torch(model_name, weights_path)
        return model
    if backend == "jax":
        model, params = _load_jax(model_name, weights_path)
        return model, params
    if backend == "mlx":
        model = _load_mlx(model_name, weights_path)
        return model
    raise ValueError(
        f"Invalid backend {backend}. Valid backends are: ('torch', 'jax', 'mlx')."
    )
