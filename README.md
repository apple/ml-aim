# Autoregressive Pre-training of Large Vision Encoders
<div>
<a href="https://arxiv.org/abs/2411.14402" target="_blank"><img alt="AIMv2 arXiv" src="https://img.shields.io/badge/arXiv-AIMv2-red?logo=arxiv"/></a>
<a href="#aimv2-model-gallery"><img alt="AIMv2 model gallery" src="https://img.shields.io/badge/model_gallery-AIMv2-blue"/></a>
<a href="https://arxiv.org/abs/2401.08541" target="_blank"><img alt="AIMv1 arXiv" src="https://img.shields.io/badge/arXiv-AIMv1-red?logo=arxiv"/></a>
<a href="aim-v1/README.md#pre-trained-backbones"> <img alt="AIMv1 model gallery" src="https://img.shields.io/badge/model_gallery-AIMv1-blue"/></a>
</div>

This repository is the entry point for all things AIM, a family of autoregressive models that push the boundaries of
visual and multimodal learning:

- **AIMv2**: [`Multimodal Autoregressive Pre-training of Large Vision Encoders`](https://arxiv.org/abs/2411.14402)  [[`BibTeX`](#citation)]
  <br>
  Enrico Fini*, Mustafa Shukor*, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju,
  Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang,
  Joshua M. Susskind, and Alaaeldin El-Nouby*
- **AIMv1**: [`Scalable Pre-training of Large Autoregressive Image Models`](https://arxiv.org/abs/2401.08541) [[`BibTeX`](#citation)]<br>
  Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar,
  Joshua M Susskind, Armand Joulin.

*: Equal technical contribution

If you're looking for the original AIM model (AIMv1), please refer to the README [here](aim-v1/README.md).

---

## Overview of AIMv2
We introduce the AIMv2 family of vision models pre-trained with a multimodal autoregressive objective.
AIMv2 pre-training is simple and straightforward to train and to scale effectively. Some AIMv2 highlights include:

1. Outperforms OAI CLIP and SigLIP on the majority of multimodal understanding benchmarks.
2. Outperforms DINOv2 on open-vocabulary object detection and referring expression comprehension.
3. Exhibits strong recognition performance with AIMv2-3B achieving *89.5% on ImageNet using a frozen trunk*.

![gh_aimv2_dark](aim-v2/assets/aimv2_overview_dark.png#gh-dark-mode-only)
![gh_aimv2_light](aim-v2/assets/aimv2_overview_light.png#gh-light-mode-only)

## AIMv2 Model Gallery
<div>
<a href="#using-pytorch"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" height="25"/></a>
<a href="#using-jax"><img alt="JAX" src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" height="25"/></a>
<a href="#using-mlx"><img alt="MLX" src="aim-v2/assets/mlx_logo_light.png" height="25"/></a>
<a href="https://huggingface.co/collections/apple/aimv2-6720fe1558d94c7805f7688c"><img alt="HuggingFace" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" height="25"></a>
</div>

We share with the community AIMv2 pre-trained checkpoints of varying capacities, pre-training resolutions:

+ [[`AIMv2 with 224px`]](#aimv2-with-224px)
+ [[`AIMv2 with 336px`]](#aimv2-with-336px)
+ [[`AIMv2 with 448px`]](#aimv2-with-448px)
+ [[`AIMv2 with Native Resolution`]](#aimv2-with-native-resolution)
+ [[`AIMv2 distilled ViT-Large`]](#aimv2-distilled-vit-large) (*recommended for multimodal applications*)
+ [[`Zero-shot Adapted AIMv2`]](#zero-shot-adapted-aimv2)

## Installation
Please install PyTorch using the official [installation instructions](https://pytorch.org/get-started/locally/).
Afterward, install the package as:
```commandline
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v1'
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'
```
We also offer [MLX](https://ml-explore.github.io/mlx/) backend support for research and experimentation on Apple silicon.
To enable MLX support, simply run:
```commandline
pip install mlx
```

## Examples

### Using PyTorch

```python
from PIL import Image

from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

img = Image.open(...)
model = load_pretrained("aimv2-large-patch14-336", backend="torch")
transform = val_transforms(img_size=336)

inp = transform(img).unsqueeze(0)
features = model(inp)
```

### Using MLX
<details>

```python
from PIL import Image
import mlx.core as mx

from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

img = Image.open(...)
model = load_pretrained("aimv2-large-patch14-336", backend="mlx")
transform = val_transforms(img_size=336)

inp = transform(img).unsqueeze(0)
inp = mx.array(inp.numpy())
features = model(inp)
```
</details>

### Using JAX

<details>

```python
from PIL import Image
import jax.numpy as jnp

from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

img = Image.open(...)
model, params = load_pretrained("aimv2-large-patch14-336", backend="jax")
transform = val_transforms(img_size=336)

inp = transform(img).unsqueeze(0)
inp = jnp.array(inp)
features = model.apply({"params": params}, inp)
```
</details>

## Pre-trained Checkpoints
The pre-trained models can be accessed via [HuggingFace Hub](https://huggingface.co/collections/apple/aimv2-6720fe1558d94c7805f7688c) as:
```python
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

image = Image.open(...)
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-336")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-336", trust_remote_code=True)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

### AIMv2 with 224px
<table style="margin: auto">
  <thead>
    <tr>
      <th>model_id</th>
      <th>#params</th>
      <th>IN-1k</th>
      <th>HF Link</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>aimv2-large-patch14-224</td>
      <td>0.3B</td>
      <td>86.6</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-224" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-224/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-huge-patch14-224</td>
      <td>0.6B</td>
      <td>87.5</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-huge-patch14-224" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-huge-patch14-224/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-1B-patch14-224</td>
      <td>1.2B</td>
      <td>88.1</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-1B-patch14-224" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-1B-patch14-224/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-3B-patch14-224</td>
      <td>2.7B</td>
      <td>88.5</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-3B-patch14-224" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-3B-patch14-224/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

### AIMv2 with 336px
<table style="margin: auto">
  <thead>
    <tr>
      <th>model_id</th>
      <th>#params</th>
      <th>IN-1k</th>
      <th>HF Link</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>aimv2-large-patch14-336</td>
      <td>0.3B</td>
      <td>87.6</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-336" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-336/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-huge-patch14-336</td>
      <td>0.6B</td>
      <td>88.2</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-huge-patch14-336" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-huge-patch14-336/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-1B-patch14-336</td>
      <td>1.2B</td>
      <td>88.7</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-1B-patch14-336" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-1B-patch14-336/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-3B-patch14-336</td>
      <td>2.7B</td>
      <td>89.2</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-3B-patch14-336" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-3B-patch14-336/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

### AIMv2 with 448px
<table style="margin: auto">
  <thead>
    <tr>
      <th>model_id</th>
      <th>#params</th>
      <th>IN-1k</th>
      <th>HF Link</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>aimv2-large-patch14-448</td>
      <td>0.3B</td>
      <td>87.9</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-448">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-448/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-huge-patch14-448</td>
      <td>0.6B</td>
      <td>88.6</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-huge-patch14-448">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-huge-patch14-448/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-1B-patch14-448</td>
      <td>1.2B</td>
      <td>89.0</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-1B-patch14-448">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-1B-patch14-448/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-3B-patch14-448</td>
      <td>2.7B</td>
      <td>89.5</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-3B-patch14-448">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-3B-patch14-448/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

### AIMv2 with Native Resolution
We additionally provide an AIMv2-L checkpoint that is finetuned to process a wide range of image resolutions and
aspect ratios. Regardless of the aspect ratio, the image is patchified (patch_size=14) and
*a 2D sinusoidal positional embedding* is added to the linearly projected input patches.
*This checkpoint supports number of patches in the range of [112, 4096]*.

<table style="margin: auto">
  <thead>
    <tr>
      <th>model_id</th>
      <th>#params</th>
      <th>IN-1k</th>
      <th>HF Link</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>aimv2-large-patch14-native</td>
      <td>0.3B</td>
      <td>87.3</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-native" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-native/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

### AIMv2 distilled ViT-Large
We provide an AIMv2-L checkpoint distilled from AIMv2-3B that provides a remarkable performance for multimodal
understanding benchmarks.

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>VQAv2</th>
      <th>GQA</th>
      <th>OKVQA</th>
      <th>TextVQA</th>
      <th>DocVQA</th>
      <th>InfoVQA</th>
      <th>ChartQA</th>
      <th>SciQA</th>
      <th>MMEp</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>AIMv2-L</td>
      <td>80.2</td>
      <td>72.6</td>
      <td>60.9</td>
      <td>53.9</td>
      <td>26.8</td>
      <td>22.4</td>
      <td>20.3</td>
      <td>74.5</td>
      <td>1457</td>
     </tr>
    <tr>
      <td>AIMv2-L-distilled</td>
      <td>81.1</td>
      <td>73.0</td>
      <td>61.4</td>
      <td>53.5</td>
      <td>29.2</td>
      <td>23.3</td>
      <td>24.0</td>
      <td>76.3</td>
      <td>1627</td>
    </tr>
  </tbody>
</table>

<table style="margin: auto">
  <thead>
    <tr>
      <th>model_id</th>
      <th>#params</th>
      <th>Res.</th>
      <th>HF Link</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>aimv2-large-patch14-224-distilled</td>
      <td>0.3B</td>
      <td>224px</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-224-distilled" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-224-distilled/resolve/main/model.safetensors">link</a></td>
    </tr>
    <tr>
      <td>aimv2-large-patch14-336-distilled</td>
      <td>0.3B</td>
      <td>336px</td>
      <td>🤗<a href="https://huggingface.co/apple/aimv2-large-patch14-336-distilled" target="_blank">link</a></td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-336-distilled/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

### Zero-shot Adapted AIMv2
We provide the AIMv2-L vision and text encoders after LiT tuning to enable zero-shot recognition.

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>#params</th>
      <th>zero-shot IN1-k</th>
      <th>Backbone</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>AIMv2-L</td>
      <td>0.3B</td>
      <td>77.0</td>
      <td><a href="https://huggingface.co/apple/aimv2-large-patch14-224-lit/resolve/main/model.safetensors">link</a></td>
    </tr>
  </tbody>
</table>

## Citation
If you find our work useful, please consider citing us as:

### AIMv2 bibtex

```bibtex
@misc{fini2024multimodal,
    title={Multimodal Autoregressive Pre-training of Large Vision Encoders},
    author={Enrico Fini and Mustafa Shukor and Xiujun Li and Philipp Dufter and Michal Klein and David Haldimann and Sai Aitharaju and Victor Guilherme Turrisi da Costa and Louis Béthune and Zhe Gan and Alexander T Toshev and Marcin Eichner and Moin Nabi and Yinfei Yang and Joshua M. Susskind and Alaaeldin El-Nouby},
    year={2024},
    eprint={2411.14402},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### AIMv1 bibtex

```bibtex
@InProceedings{pmlr-v235-el-nouby24a,
  title     = {Scalable Pre-training of Large Autoregressive Image Models},
  author    = {El-Nouby, Alaaeldin and Klein, Michal and Zhai, Shuangfei and Bautista, Miguel \'{A}ngel and Shankar, Vaishaal and Toshev, Alexander T and Susskind, Joshua M. and Joulin, Armand},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {12371--12384},
  year      = {2024},
}
```

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and models.
