# AIM: Autoregressive Image Models

*Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar,
Joshua M Susskind, and Armand Joulin*

[[`Paper`](https://arxiv.org/abs/2401.08541)]  [[`BibTex`](#citation)]

This software project accompanies the research paper, [Scalable Pre-training of Large Autoregressive Image Models](https://arxiv.org/abs/2401.08541).

We introduce **AIM** a collection of vision models pre-trained with an autoregressive generative objective.
We show that autoregressive pre-training of image features exhibits similar scaling properties to their
textual counterpart (i.e. Large Language Models). Specifically, we highlight two findings:
1. the model capacity can be trivially scaled to billions of parameters, and
2. AIM effectively leverages large collections of uncurated image data.

## Installation
Please install PyTorch using the official [installation instructions](https://pytorch.org/get-started/locally/).
Afterward, install the package as:
```commandline
pip install git+https://git@github.com/apple/ml-aim.git
```
We also offer [MLX](https://github.com/ml-explore/mlx) backend support for research and experimentation on Apple silicon.
To enable MLX support, simply run:
```commandline
pip install mlx
```

## Usage
Below we provide an example of usage in [PyTorch](https://pytorch.org/):
```python
from PIL import Image

from aim.utils import load_pretrained
from aim.torch.data import val_transforms

img = Image.open(...)
model = load_pretrained("aim-600M-2B-imgs", backend="torch")
transform = val_transforms()

inp = transform(img).unsqueeze(0)
logits, _ = model(inp)
```

<details>
<summary>and in both <a href="https://ml-explore.github.io/mlx/">MLX</a></summary>

```python
from PIL import Image
import mlx.core as mx

from aim.utils import load_pretrained
from aim.torch.data import val_transforms

img = Image.open(...)
model = load_pretrained("aim-600M-2B-imgs", backend="mlx")
transform = val_transforms()

inp = transform(img).unsqueeze(0)
inp = mx.array(inp.numpy())
logits, _ = model(inp)
```
</details>

<details>
<summary>and <a href="https://jax.readthedocs.io/">JAX</a></summary>

```python
from PIL import Image
import jax.numpy as jnp

from aim.utils import load_pretrained
from aim.torch.data import val_transforms

img = Image.open(...)
model, params = load_pretrained("aim-600M-2B-imgs", backend="jax")
transform = val_transforms()

inp = transform(img).unsqueeze(0)
inp = jnp.array(inp)
(logits, _), _ = model.apply(params, inp, mutable=['batch_stats'])
```
</details>

## Pre-trained checkpoints

The pre-trained models can be accessed via [PyTorch Hub](https://pytorch.org/hub/) as:
```python
import torch

aim_600m = torch.hub.load("apple/ml-aim", "aim_600M")
aim_1b   = torch.hub.load("apple/ml-aim", "aim_1B")
aim_3b   = torch.hub.load("apple/ml-aim", "aim_3B")
aim_7b   = torch.hub.load("apple/ml-aim", "aim_7B")
```
or via [HuggingFace Hub](https://huggingface.co/docs/hub/) as:
```python
from aim.torch.models import AIMForImageClassification

aim_600m = AIMForImageClassification.from_pretrained("apple/aim-600M")
aim_1b   = AIMForImageClassification.from_pretrained("apple/aim-1B")
aim_3b   = AIMForImageClassification.from_pretrained("apple/aim-3B")
aim_7b   = AIMForImageClassification.from_pretrained("apple/aim-7B")
```

### Pre-trained backbones

The following table contains pre-trained backbones used in our paper.

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>#params</th>
      <th>attn (best layer)</th>
      <th>backbone, SHA256</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AIM-0.6B</td>
      <td>0.6B</td>
      <td>79.4%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_backbone.pth">link</a>, 0d6f6b8f</td>
    </tr>
    <tr>
      <td>AIM-1B</td>
      <td>1B</td>
      <td>82.3%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_backbone.pth">link</a>, d254ecd3</td>
    </tr>
    <tr>
      <td>AIM-3B</td>
      <td>3B</td>
      <td>83.3%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_backbone.pth">link</a>, 8475ce4e</td>
    </tr>
    <tr>
      <td>AIM-7B</td>
      <td>7B</td>
      <td>84.0%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_backbone.pth">link</a>, 184ed94c</td>
    </tr>
  </tbody>
</table>

### Pre-trained attention heads

The table below contains the classification results on ImageNet-1k validation set.

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">model</th>
      <th colspan="2">top-1 IN-1k</th>
      <th colspan="2">attention head, SHA256</th>
    </tr>
    <tr>
      <th>last layer</th>
      <th>best layer</th>
      <th>last layer</th>
      <th>best layer</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>AIM-0.6B</td>
      <td>78.5%</td>
      <td>79.4%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_head_last_layers.pth">link</a>, 5ce5a341</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_head_best_layers.pth">link</a>, ebd45c05</td>
    </tr>
    <tr>
      <td>AIM-1B</td>
      <td>80.6%</td>
      <td>82.3%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_head_last_layers.pth">link</a>, db3be2ad</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_head_best_layers.pth">link</a>, f1ed7852</td>
    </tr>
    <tr>
      <td>AIM-3B</td>
      <td>82.2%</td>
      <td>83.3%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_head_last_layers.pth">link</a>, 5c057b30</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_head_best_layers.pth">link</a>, ad380e16</td>
    </tr>
    <tr>
      <td>AIM-7B</td>
      <td>82.4%</td>
      <td>84.0%</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_head_last_layers.pth">link</a>, 1e5c99ba</td>
      <td><a href="https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_head_best_layers.pth">link</a>, 73ecd732</td>
    </tr>
  </tbody>
</table>

## Reproducing the IN-1k classification results
The commands below reproduce the [attention probe results](#pre-trained-attention-heads) on ImageNet-1k
validation set. We run the evaluation using 1 node with 8 GPUs:
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=8 main_attnprobe.py \
  --model=aim-7B \
  --batch-size=64 \
  --data-path=/path/to/imagenet \
  --probe-layers=last \
  --backbone-ckpt-path=/path/to/backbone_ckpt.pth \
  --head-ckpt-path=/path/to/head_ckpt.pth
```
By default, we probe the last 6 layers. To change this, simply pass `--probe-layers=best`.

## Citation
If you find our work useful, please consider citing us as:
```
@misc{elnouby2024scalable,
      title={Scalable Pre-training of Large Autoregressive Image Models},
      author={Alaaeldin El-Nouby and Michal Klein and Shuangfei Zhai and Miguel Angel Bautista and Alexander Toshev and Vaishaal Shankar and Joshua M Susskind and Armand Joulin},
      year={2024},
      eprint={2401.08541},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
