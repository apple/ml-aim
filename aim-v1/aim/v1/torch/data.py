# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
import math
from typing import TYPE_CHECKING, Literal, NoReturn

from torch.utils.data import DataLoader, distributed

if TYPE_CHECKING:
    from torchvision import transforms

__all__ = ["create_dataloader", "val_transforms"]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_dataloader(
    root: str, *, split: Literal["train", "val"], batch_size: int, num_workers: int
) -> DataLoader:
    """Create a dataloader from a directory.

    Args:
        root: Path to the root of the dataset.
        split: Whether to create train or validation dataloader.
            It assumes there exists `path/train` or `path/val` directory.
        batch_size: Batch size per GPU.
        num_workers: Number of workers per GPU.

    Returns:
        The train or validation dataloader, depending on the `split`.
    """
    from torchvision import datasets

    if split == "train":
        transform = _train_transforms()
    elif split == "val":
        transform = val_transforms()
    else:
        raise ValueError(f"Invalid split: {split}.")

    root = os.path.join(root, split)
    dataset = datasets.ImageFolder(root, transform=transform)
    sampler = distributed.DistributedSampler(dataset, shuffle=split == "train")

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def _train_transforms() -> NoReturn:
    raise NotImplementedError("Train transforms are not yet implemented.")


def val_transforms(img_size: int = 224) -> "transforms.Compose":
    """Validation transformations.

    Args:
        img_size: Image size.

    Returns:
        The transformation.
    """
    from torchvision import transforms

    size = math.ceil(img_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
