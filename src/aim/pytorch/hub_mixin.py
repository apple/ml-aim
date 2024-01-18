"""Illustrates the mixin for one class."""

import torch
from models import AIMForImageClassification
from data import val_transforms
from PIL import Image
import requests

# define config
config = dict(
        image_size=224,
        num_channels=3,
        patch_size=14,
        embed_dim=1536,
        num_blocks=24,
        num_heads=12,
        probe_layers=6,
        num_classes=1000,
)

# define model
model = AIMForImageClassification(config)

# load weights
aim_600m = torch.hub.load("apple/ml-aim", "aim_600M")
model.load_state_dict(aim_600m.state_dict())

# save locally
# model.save_pretrained(".", config=config)

# push to hub
# model.push_to_hub("nielsr/aim-600M", config=config)

# reload from hub
reload_model = AIMForImageClassification.from_pretrained("nielsr/aim-600m")

transform = val_transforms()

# inference
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inp = transform(image).unsqueeze(0)
logits, _ = model(inp)

print(logits.argmax(-1))