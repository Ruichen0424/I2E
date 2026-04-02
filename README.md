<div align="center">

<h1>I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks</h1>

<!-- Badges -->
[![Paper](https://img.shields.io/badge/Arxiv-2511.08065-B31B1B.svg?style=flat-square)](https://arxiv.org/abs/2511.08065)
[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Oral-4b44ce.svg?style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/37179)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Paper-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/scholar?cluster=1814482600796011970)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2511.08065)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Ruichen0424/I2E)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/UESTC-BICS/I2E)

[![YouTube](https://img.shields.io/badge/YouTube-Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/v9z0pn8kTsI?si=dkbjR6AV_RgCU3wA)
[![Bilibili](https://img.shields.io/badge/Bilibili-Video-FE7398?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV14XrfBjENb/)

</div>

<br>

## 🚀 Introduction

This is the official PyTorch implementation of the paper **I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks**, accepted for **Oral Presentation** at **AAAI 2026**.

**I2E** is a pioneering framework that bridges the data scarcity gap in neuromorphic computing.
By simulating microsaccadic eye movements via highly parallelized convolution, I2E converts static images into high-fidelity event streams in real-time (>300x faster than prior methods).

### ✨ Key Highlights
* **SOTA Performance**: Achieves **60.50%** top-1 accuracy on Event-based ImageNet.
* **Sim-to-Real Transfer**: Pre-training on I2E data enables **92.5%** accuracy on real-world CIFAR10-DVS, establishing a new SOTA benchmark.
* **Real-Time Conversion**: Uniquely enables on-the-fly data augmentation for deep SNN training.

## 📄 Abstract
Spiking neural networks (SNNs) promise highly energy-efficient computing, but their adoption is hindered by a critical scarcity of event-stream data.
This work introduces I2E, an algorithmic framework that resolves this bottleneck by converting static images into high-fidelity event streams.
By simulating microsaccadic eye movements with a highly parallelized convolution, I2E achieves a conversion speed over 300x faster than prior methods, uniquely enabling on-the-fly data augmentation for SNN training.
The framework's effectiveness is demonstrated on large-scale benchmarks.
An SNN trained on the generated I2E-ImageNet dataset achieves a state-of-the-art accuracy of 60.50%.
Critically, this work establishes a powerful sim-to-real paradigm where pre-training on synthetic I2E data and fine-tuning on the real-world CIFAR10-DVS dataset yields an unprecedented accuracy of 92.5%.
This result validates that synthetic event data can serve as a high-fidelity proxy for real sensor data, bridging a long-standing gap in neuromorphic engineering.
By providing a scalable solution to the data problem, I2E offers a foundational toolkit for developing high-performance neuromorphic systems.
The open-source algorithm and all generated datasets are provided to accelerate research in the field.

## 👁️ Visualization

Below is the visualization of the conversion process from static RGB images to dynamic event streams. We illustrate the high-fidelity conversion with four examples.

More than 200 additional visualization comparisons can be found in [Visualization.md](./Visualization.md).

<table border="0" style="width: 100%">
  <tr>
    <td width="25%" align="center"><img src="./assets/original_1.jpg" alt="Original 1" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/converted_1.gif" alt="Converted 1" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/original_2.jpg" alt="Original 2" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/converted_2.gif" alt="Converted 2" style="width:100%"></td>
  </tr>
  <tr>
    <td width="25%" align="center"><img src="./assets/original_3.jpg" alt="Original 3" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/converted_3.gif" alt="Converted 3" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/original_4.jpg" alt="Original 4" style="width:100%"></td>
    <td width="25%" align="center"><img src="./assets/converted_4.gif" alt="Converted 4" style="width:100%"></td>
  </tr>
</table>


## 📦 Dataset Catalog

We provide a comprehensive collection of standard benchmarks converted into event streams via the I2E algorithm.

### 1. Standard Benchmarks (Classification)
| Config Name | Original Source | Resolution $(H, W)$ | I2E Ratio | Event Rate | Samples (Train/Val) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`I2E-CIFAR10`** | CIFAR-10 | 128 x 128 | 0.07 | 5.86% | 50k / 10k |
| **`I2E-CIFAR100`** | CIFAR-100 | 128 x 128 | 0.07 | 5.76% | 50k / 10k |
| **`I2E-ImageNet`** | ILSVRC2012 | 224 x 224 | 0.12 | 6.66% | 1.28M / 50k |

### 2. Transfer Learning & Fine-grained
| Config Name | Original Source | Resolution $(H, W)$ | I2E Ratio | Event Rate | Samples |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`I2E-Caltech101`** | Caltech-101 | 224 x 224 | 0.12 | 6.25% | 8.677k |
| **`I2E-Caltech256`** | Caltech-256 | 224 x 224 | 0.12 | 6.04% | 30.607k |
| **`I2E-Mini-ImageNet`**| Mini-ImageNet | 224 x 224 | 0.12 | 6.65% | 60k |

### 3. Small Scale / Toy
| Config Name | Original Source | Resolution $(H, W)$ | I2E Ratio | Event Rate | Samples |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`I2E-MNIST`** | MNIST | 32 x 32 | 0.10 | 9.56% | 60k / 10k |
| **`I2E-FashionMNIST`** | Fashion-MNIST | 32 x 32 | 0.15 | 10.76% | 60k / 10k |

> 🔜 **Coming Soon:** Object Detection and Semantic Segmentation datasets.

**Download Links:**

 - [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/UESTC-BICS/I2E)

 - [![Baidu Netdisk](https://img.shields.io/badge/Baidu%20Netdisk-Datasets-06A7FF?style=flat-square&logo=baidu&logoColor=white)](https://pan.baidu.com/s/1G1J6MG0d_NFQuoTxR7YLWQ?pwd=ItoE)


### 🚀 Quick Start

You **do not** need to download any extra scripts. Just copy the code below. It handles the binary unpacking (converting Parquet bytes to PyTorch Tensors) automatically.

```python
import io
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# ==================================================================
# 1. Core Decoding Function (Handles the binary packing)
# ==================================================================
def unpack_event_data(item, use_io=True):
    """
    Decodes the custom binary format:
    Header (8 bytes) -> Shape (T, C, H, W) -> Body (Packed Bits)
    """
    if use_io:
        with io.BytesIO(item['data']) as f:
            raw_data = np.load(f)
    else:
        raw_data = np.load(item)
        
    header_size = 4 * 2      # Parse Header (First 8 bytes for 4 uint16 shape values)
    shape_header = raw_data[:header_size].view(np.uint16)
    original_shape = tuple(shape_header) # Returns (T, C, H, W)
    
    packed_body = raw_data[header_size:]    # Parse Body & Bit-unpacking
    unpacked = np.unpackbits(packed_body)
    
    num_elements = np.prod(original_shape)  # Extract valid bits (Handle padding)
    event_flat = unpacked[:num_elements]
    event_data = event_flat.reshape(original_shape).astype(np.float32).copy()
    
    return torch.from_numpy(event_data)

# ==================================================================
# 2. Dataset Wrapper
# ==================================================================
class I2E_Dataset(Dataset):
    def __init__(self, cache_dir, config_name, split='train', transform=None, target_transform=None):
        print(f"🚀 Loading {config_name} [{split}] from Hugging Face...")
        self.ds = load_dataset('UESTC-BICS/I2E', config_name, split=split, cache_dir=cache_dir, keep_in_memory=False)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        event = unpack_event_data(item)
        label = item['label']
        if self.transform:
            event = self.transform(event)
        if self.target_transform:
            label = self.target_transform(label)
        return event, label

# ==================================================================
# 3. Run Example
# ==================================================================
if __name__ == "__main__":
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'     # Use HF mirror server in some regions

    DATASET_NAME = 'I2E-CIFAR10'                            # Choose your config: 'I2E-CIFAR10', 'I2E-ImageNet', etc.
    MODEL_PATH = 'Your cache path here'                     # e.g., './hf_datasets_cache/'
    
    train_dataset = I2E_Dataset(MODEL_PATH, DATASET_NAME, split='train')
    val_dataset = I2E_Dataset(MODEL_PATH, DATASET_NAME, split='validation')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=32, persistent_workers=True)

    events, labels = next(iter(train_loader))
    print(f"✅ Loaded Batch Shape: {events.shape}") # Expect: [32, T, 2, H, W]
    print(f"✅ Labels: {labels}")
```


### 🛠️ Preprocessing Protocol

To ensure reproducibility, we specify the exact data augmentation pipeline applied to the static images **before** I2E conversion. 

The `(H, W)` in the code below corresponds to the "Resolution" column in the Dataset Catalog above.

```python
from torchvision.transforms import v2

# Standard Pre-processing Pipeline used for I2E generation
transform_train = v2.Compose([
    # Ensure 3-channel RGB (crucial for grayscale datasets like MNIST)
    v2.Lambda(lambda x: x.convert('RGB')),
    v2.PILToTensor(),
    v2.Resize((H, W), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
])
````

## 🛠️ Requirements

- python==3.10
- pytorch==2.2.0
- torchvision==0.17.0
- spikingjelly (dev version between 0.0.0.0.14 and 0.0.0.1.0)
- timm==1.0.19

### Environment Setup

We recommend using Anaconda to create a virtual environment:

```bash
conda create -n i2e python=3.10
conda activate i2e
```

Install PyTorch and dependencies:

```bash
# Install PyTorch (Choose based on your CUDA version)
# CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install SpikingJelly and timm
pip install timm==1.0.19
pip install spikingjelly
```


## 💻 Usage

All training scripts are located in the [`./Train Script`](./Train%20Script/) folder. We provide training code for [**Baseline-I**](./Train%20Script/Baseline-I/), [**Baseline-II**](./Train%20Script/Baseline-II/), and [**DVS-CIFAR10**](./Train%20Script/DVS_CIFAR10/), as well as [inference code](./Train%20Script/Inference/) for all provided weights.

### Training (Baseline-II)
To train the models using the Baseline-II setting (with full augmentation), use the following commands. Please ensure you update the `--dataset_path` (or `-dp`) argument to point to your local dataset location.

**CIFAR-10**
```bash
python train.py -bz 128 -dp '/path/to/CIFAR10/' --dataset 'cifar10' -n 'CIFAR10' -cn 10 -e 256 --lr 0.1 --lr_min 5e-5 -wd 2e-4 --label_smooth 0.1 --model 'resnet18' --ratio 0.07 --shuffle 4 -p 30
```

**CIFAR-100**
```bash
python train.py -bz 128 -dp '/path/to/CIFAR100/' --dataset 'cifar100' -n 'CIFAR100' -cn 100 -e 256 --lr 0.1 --lr_min 5e-5 -wd 2e-4 --label_smooth 0.1 --model 'resnet18' --ratio 0.07 --shuffle 4 -p 30
```

**ImageNet**
```bash
python train.py -bz 128 -dp '/path/to/ImageNet/' --dataset 'imagenet' -n 'ImageNet' -cn 1000 -e 128 --lr 0.1 --lr_min 5e-5 -wd 1e-5 --label_smooth 0.1 --model 'resnet18' --ratio 0.12 --shuffle 4 -p 200 --multiprocessing_distributed
```



## 🤖 Pre-trained Models
We provide pre-trained models for I2E-CIFAR and I2E-ImageNet.

- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Ruichen0424/I2E)
- [![Baidu Netdisk](https://img.shields.io/badge/Baidu%20Netdisk-Models-06A7FF?style=flat-square&logo=baidu&logoColor=white)](https://pan.baidu.com/s/1IFyfL8EwtPCEcu73xmx13Q?pwd=ItoE)


## 📊 Main Results & Model Zoo
The experimental settings for the methods listed below are as follows:
- **Baseline-I**: Training from scratch with minimal augmentation.
- **Baseline-II**: Training from scratch with full augmentation (random crop, etc.), enabled by I2E.
- **Transfer-I**: Fine-tuning on target dataset after pre-training on a source dataset.
- **Transfer-II**: Fine-tuning on target dataset after pre-training on **I2E-CIFAR10**.

<table>

<tr>
<th>Dataset</th>
<th align="center">Structure</th>
<th align="center">Method</th>
<th align="center">Top-1 Acc</th>
<th align="center">Downloadable</th>
</tr>

<tr>
<td rowspan=3 align="center"><strong>CIFAR10-DVS</strong></td>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline</td>
<td align="center">65.6%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Transfer-I</td>
<td align="center">83.1%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Transfer-II</td>
<td align="center"><strong>92.5%</strong></td>
<td align="center">&#x2714;</td>
</tr>

<tr>
<td rowspan=3 align="center"><strong>I2E-CIFAR10</strong></td>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-I</td>
<td align="center">85.07%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-II</td>
<td align="center">89.23%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Transfer-I</td>
<td align="center"><strong>90.86%</strong></td>
<td align="center">&#x2714;</td>
</tr>

<tr>
<td rowspan=3 align="center"><strong>I2E-CIFAR100</strong></td>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-I</td>
<td align="center">51.32%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-II</td>
<td align="center">60.68%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Transfer-I</td>
<td align="center"><strong>64.53%</strong></td>
<td align="center">&#x2714;</td>
</tr>

<tr>
<td rowspan=4 align="center"><strong>I2E-ImageNet</strong></td>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-I</td>
<td align="center">48.30%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Baseline-II</td>
<td align="center">57.97%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet18</td>
<td align="center">Transfer-I</td>
<td align="center">59.28%</td>
<td align="center">&#x2714;</td>
</tr>
<tr>
<td align="center">MS-ResNet34</td>
<td align="center">Baseline-II</td>
<td align="center"><strong>60.50%</strong></td>
<td align="center">&#x2714;</td>
</tr>

</table>

## 📜 Citation
If you find our code useful for your research, or use the I2E algorithm, or use the provided I2E-Datasets, please consider citing:

```bibtex
@article{ma2025i2e,
  title={I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks},
  author={Ma, Ruichen and Meng, Liwei and Qiao, Guanchao and Ning, Ning and Liu, Yang and Hu, Shaogang},
  journal={arXiv preprint arXiv:2511.08065},
  year={2025}
}
```

## 🖼️ Poster
![poster](./assets/poster.png)