# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, Any, Tuple, Optional, Union
from huggingface_hub import hf_hub_download

__all__ = ["load_pretrained"]


MODELS = {
    # 224
    "aimv2-large-patch14-224": "ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
    "aimv2-huge-patch14-224": "9e0bab0e0abb7b20796d2d0b37f39b7c3b1c1af1",
    "aimv2-1B-patch14-224": "4d24a422f71e87d327d476bc405030fe6593d8b1",
    "aimv2-3B-patch14-224": "df1cbc288096c9742ec8bd6709ff82757972338c",
    # 336
    "aimv2-large-patch14-336": "639423ae9f07319d24a7fc431b61908110d08d3a",
    "aimv2-huge-patch14-336": "21528bd53a2438a84162e79f1f9a46453b88004c",
    "aimv2-1B-patch14-336": "5a15f23f6757776fdd487283353a018677621939",
    "aimv2-3B-patch14-336": "d1adb39ee92dfd7ecf3114b1ee3aa7e9027ce98f",
    # 448
    "aimv2-large-patch14-448": "cefb13f21003bdadba65bfbee956c82b976cd23d",
    "aimv2-huge-patch14-448": "44e6440a24a8ea966ad0deadf5399a22d852bcc0",
    "aimv2-1B-patch14-448": "7f292735d3a07a911559c0fabb3ad3e9d141713f",
    "aimv2-3B-patch14-448": "70810b618e4456b724bacc8ef4d2d038060ceda6",
    # distilled
    "aimv2-large-patch14-224-distilled": "d6d09b071e8ba31735c5e05ccd4b2a393020ce90",
    "aimv2-large-patch14-336-distilled": "8fe1704e5cc1c80cefc99c194afcefe21dff1faa",
    # native
    "aimv2-large-patch14-native": "cfe7bdbbc924a5f65272d5e2c27fe5f61771a629",
    # lit
    "aimv2-large-patch14-224-lit": "c2cd59a786c4c06f39d199c50d08cc2eab9f8605",
}
ModelName = Literal[tuple(MODELS)]


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
        raise ValueError(
            f"Invalid model: {model_name}. Valid models are: {tuple(MODELS.keys())}."
        )

    fname = _get_weights_fname(backend)
    revision = MODELS[model_name]
    print(model_name, revision)
    weights_path = hf_hub_download(
        f"apple/{model_name}",
        filename=fname,
        revision=revision,
        **kwargs,
    )

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
