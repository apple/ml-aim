# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# --------------------------------------------------
# References:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist

from aim import constants

__all__ = [
    "accuracy",
    "is_dist_avail_and_initialized",
    "merge_state_dicts",
    "is_main_process",
    "init_distributed_mode",
    "torch_weights_to_mlx",
    "torch_weights_to_jax",
    "init_jax_params",
    "load_pretrained",
]


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[float]:
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


def merge_state_dicts(
    backbone_state_dict: Dict[str, torch.Tensor],
    head_state_dict: Dict[str, torch.Tensor],
    *,
    allow_override: bool = False,
) -> Dict[str, torch.Tensor]:
    overlapping_keys = set(backbone_state_dict.keys()) & set(head_state_dict.keys())
    if overlapping_keys and not allow_override:
        raise ValueError(
            f"Backbone and head state dicts have "
            f"following overlapping keys: {sorted(overlapping_keys)}."
        )

    return {**backbone_state_dict, **head_state_dict}


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    if is_dist_avail_and_initialized():
        return dist.get_rank() == 0
    return True


def init_distributed_mode(dist_url: str) -> List[int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        rank, gpu, world_size = 0, 0, 1
    else:
        raise RuntimeError(
            "Unable to initialize distributed mode. "
            "Please ensure that at least 1 GPU is available."
        )

    torch.cuda.set_device(gpu)
    print(f"| distributed init ({rank=}): {dist_url}, ({gpu=})")
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    setup_for_distributed(is_master=rank == 0)

    return [gpu]


def setup_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def torch_weights_to_mlx(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, Any]]:
    import mlx.core as mx

    weights = []
    for key, val in state_dict.items():
        # not present in `mlx`
        if key == "head.bn.num_batches_tracked":
            continue
        key = key.replace("blocks.", "blocks.layers.")
        val = mx.array(val.numpy())
        if key == "preprocessor.patchifier.proj.weight":
            # (outC, inC, kH, kW) -> (outC, kH, kW, inC)
            val = val.transpose(0, 2, 3, 1)

        weights.append((key, val))

    return weights


def torch_weights_to_jax(state_dict: Dict[str, Any]):
    import jax
    import jax.numpy as jnp
    from flax import traverse_util

    params = {}
    with jax.default_device(jax.devices("cpu")[0]):
        for k, v in state_dict.items():
            k = k.replace(".weight", ".kernel")
            k = k.replace("norm.kernel", "norm.scale")
            k = k.replace("norm_1.kernel", "norm_1.scale")
            k = k.replace("norm_2.kernel", "norm_2.scale")
            k = k.replace("blocks.", "layers_")
            v = jnp.array(v.numpy())

            if k == "preprocessor.patchifier.proj.kernel":
                # (outC, inC, kH, kW) -> (kH, kW, inC, outC)
                v = v.transpose(2, 3, 1, 0)
            elif ".kernel" in k and v.ndim == 2:
                # (outC, inC) -> (inC outC)
                v = v.swapaxes(0, 1)
            params[k] = v

        batch_stats = {
            "head.bn.mean": params.pop("head.bn.running_mean"),
            "head.bn.var": params.pop("head.bn.running_var"),
        }
        params = traverse_util.unflatten_dict(params, sep=".")
        batch_stats = traverse_util.unflatten_dict(batch_stats, sep=".")
        return {"params": params, "batch_stats": batch_stats}


def init_jax_params(model: Any, seed: int = 0) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp

    rng = jax.random.PRNGKey(seed)
    img_size = model.preprocessor.patchifier.img_size
    shape = (
        (1, 3, img_size, img_size) if isinstance(img_size, int) else (1, 3, *img_size)
    )

    return model.init(rng, jnp.ones(shape))


def load_pretrained(
    arch: Literal[
        "aim-600M-2B-imgs", "aim-1B-5B-imgs", "aim-3B-5B-imgs", "aim-7B-5B-imgs"
    ],
    backend: Literal["torch", "jax", "mlx"] = "torch",
    pretrained: bool = True,
    load_head: bool = True,
    probe_layers: Literal["last", "best"] = "best",
    backbone_ckpt_path: Optional[str] = None,
    head_ckpt_path: Optional[str] = None,
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """Load an autoregressive image model (AIM).

    Args:
        arch: Model architecture.
        pretrained: Whether to also load the pretrained weights.
        backend: Model's backend.
        load_head: Whether to load the head weights.
        probe_layers: Whether to use the last or the best layers for classification.
        backbone_ckpt_path: Path where backbone weights are stored.
            If not specified, the weights will be downloaded.
        head_ckpt_path: Path where head weights are stored.
            If not specified, the weights will be downloaded.
        strict: Whether to pass `strict` flag when loading the checkpoint.
            If `None`, it will be set `True` when `load_head=True`.
        kwargs: Keyword arguments when constructing the model.

    Returns:
        The model. If `pretrained=True` and `backend='jax'`,
        return both the model and its state.
    """

    def get_load_model_fn(arch: str) -> Callable[..., Any]:
        if backend == "torch":
            from aim.torch import models
        elif backend == "jax":
            from aim.jax import models
        elif backend == "mlx":
            from aim.mlx import models
        else:
            raise ValueError(f"Invalid backend: {backend}.")

        return {
            "aim-600M-2B-imgs": models.aim_600M,
            "aim-1B-5B-imgs": models.aim_1B,
            "aim-3B-5B-imgs": models.aim_3B,
            "aim-7B-5B-imgs": models.aim_7B,
        }[arch]

    def load_state_dict(
        loader: Callable[..., Dict[str, torch.Tensor]],
        backbone_loc: str,
        head_loc: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        backbone_state_dict = loader(backbone_loc, map_location="cpu")
        if not load_head:
            return backbone_state_dict

        if head_loc is None:
            raise RuntimeError("Unable to load the head, no location specified.")
        head_state_dict = loader(head_loc, map_location="cpu")
        return merge_state_dicts(
            backbone_state_dict, head_state_dict, allow_override=False
        )

    if strict is None:
        strict = load_head  # be strict when loading the backbone and the head

    if probe_layers == "last":
        head_loc = constants.HEADS_LAST[arch]
        probe_layers = 6  # the number of last layers
    elif probe_layers == "best":
        head_loc = constants.HEADS_BEST[arch]
        probe_layers = constants.LAYERS_BEST[arch]
    else:
        raise ValueError(f"Invalid probing layers: {probe_layers}.")

    model_fn = get_load_model_fn(arch)
    model = model_fn(probe_layers=probe_layers, **kwargs)
    if not pretrained:
        if backend == "jax":
            params = init_jax_params(model, seed=0)
            return model, params
        return model

    if backbone_ckpt_path is None:
        loader = torch.hub.load_state_dict_from_url
        backbone_loc = constants.BACKBONES[arch]
    else:
        loader = torch.load
        backbone_loc, head_loc = backbone_ckpt_path, head_ckpt_path

    state_dict = load_state_dict(loader, backbone_loc=backbone_loc, head_loc=head_loc)
    if backend == "torch":
        _ = model.load_state_dict(state_dict, strict=strict)
        return model

    if backend == "jax":
        params = torch_weights_to_jax(state_dict)
        return model, params

    if backend == "mlx":
        weights = torch_weights_to_mlx(state_dict)
        model.load_weights(weights, strict=strict)
        return model

    raise ValueError(f"Invalid backend: {backend}.")
