# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
__all__ = ["BACKBONES", "HEADS_BEST", "HEADS_LAST", "LAYERS_BEST"]

BACKBONES = {
    "aim-600M-2B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_backbone.pth",
    "aim-1B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_backbone.pth",
    "aim-3B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_backbone.pth",
    "aim-7B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_backbone.pth",
}
HEADS_LAST = {
    "aim-600M-2B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_head_last_layers.pth",
    "aim-1B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_head_last_layers.pth",
    "aim-3B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_head_last_layers.pth",
    "aim-7B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_head_last_layers.pth",
}
HEADS_BEST = {
    "aim-600M-2B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_head_best_layers.pth",
    "aim-1B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_head_best_layers.pth",
    "aim-3B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_head_best_layers.pth",
    "aim-7B-5B-imgs": "https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_head_best_layers.pth",
}
LAYERS_BEST = {
    "aim-600M-2B-imgs": (14, 15, 16, 17, 18, 19),
    "aim-1B-5B-imgs": (12, 13, 14, 15, 16, 17),
    "aim-3B-5B-imgs": (12, 13, 14, 15, 16, 17),
    "aim-7B-5B-imgs": (14, 15, 16, 17, 18, 19),
}
