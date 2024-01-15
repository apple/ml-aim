# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import argparse
import logging
from typing import Dict

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from aim import logger, utils
from aim.torch import data

LOGGER = logging.getLogger("AIM")


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    model.eval()
    metric_logger = logger.MetricLogger(delimiter="  ")
    criterion = nn.CrossEntropyLoss()

    for inp, target in metric_logger.log_every(
        data_loader, print_freq=10, header="Test:"
    ):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            # determine the max block id from, e.g., `AverageLayers.max_block_id`
            # useful for faster eval when `probe_layers='best'`
            output, _ = model(inp, max_block_id=None)
            loss = criterion(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = target.shape[0]
        metric_logger.meters["test_loss"].update(loss, n=batch_size)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    LOGGER.info(f"Averaged stats: {metric_logger!s}")

    return metrics


def main(args: argparse.Namespace) -> None:
    cudnn.benchmark = True
    device_ids = utils.init_distributed_mode(args.dist_url)
    logger.setup_logger()

    postfix = "2B-imgs" if args.model == "aim-600M" else "5B-imgs"
    model = utils.load_pretrained(
        f"{args.model}-{postfix}",
        backend="torch",
        pretrained=True,
        probe_layers=args.probe_layers,
        backbone_ckpt_path=args.backbone_ckpt_path,
        head_ckpt_path=args.head_ckpt_path,
        load_head=True,
    )
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=True
    )

    val_dataloader = data.create_dataloader(
        args.data_path,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    _metrics = evaluate(model, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate AIM models.")

    # model
    parser.add_argument(
        "--model",
        choices=("aim-600M", "aim-1B", "aim-3B", "aim-7B"),
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--backbone-ckpt-path",
        default="/path/to/backbone_ckpt.pth",
        help="Path where the backbone checkpoint is stored.",
    )
    parser.add_argument(
        "--head-ckpt-path",
        default="/path/to/head_ckpt.pth,",
        help="Path where the attention probe head is stored.",
    )
    parser.add_argument(
        "--data-path",
        default="/path/to/imagenet",
        help="Root of the dataset. It must contain a `val` directory.",
    )
    parser.add_argument(
        "--probe-layers",
        choices=("last", "best"),
        default="last",
        help="Layers to evaluate.",
    )
    # data
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Number of batches per GPU."
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of workers per GPU."
    )
    # distributed eval
    parser.add_argument(
        "--dist-url", default="env://", help="URL used to set up distributed training."
    )

    main(parser.parse_args())
