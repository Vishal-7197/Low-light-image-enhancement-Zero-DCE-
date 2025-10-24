# Low-Light Image Enhancement — Zero-DCE

An AI project using Python and TensorFlow to implement Zero-Reference Deep Curve Estimation (Zero‑DCE) for low-light image enhancement. The goal is to improve image quality under low illumination without requiring paired reference images by learning light-correction curves via a custom loss formulation.

Badges
- Language: Python (100%)
- Framework: TensorFlow
- Status: Draft / Research

## Table of contents
- [Overview](#overview)
- [Key features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start (inference)](#quick-start-inference)
- [Training](#training)
- [Dataset](#dataset)
- [Model & outputs](#model--outputs)

## Overview
Zero‑DCE is a deep learning approach that directly estimates pixel-wise light enhancement curves without paired ground-truth images. This repository contains:
- Implementation of Zero‑DCE in TensorFlow/Python
- Training scripts, data loaders, and loss functions
- Inference utilities and sample notebooks / scripts

This implementation is intended for research and experimentation. It can be used for image enhancement on low-light photography, pre-processing for downstream vision tasks, or as a baseline for further model improvements.

## Key features
- Zero-reference learning (no paired GT images required)
- Custom losses for naturalness and contrast preservation
- Inference script to enhance single images or batches
- Training pipeline with configurable hyperparameters
- Example notebooks / scripts for quick experiments

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, OpenCV (cv2), Pillow
- Optional: Matplotlib (visualization), tqdm

Install core dependencies:
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, install minimal:
```bash
pip install tensorflow numpy opencv-python pillow tqdm matplotlib
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Vishal-7197/Low-light-image-enhancement-Zero-DCE-.git
cd Low-light-image-enhancement-Zero-DCE-
```
2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies (see Requirements).

## Quick start (inference)
Enhance a single image using the provided inference script.

Example CLI (assuming a script `inference.py` exists):
```bash
python inference.py --input path/to/low_light.jpg --output path/to/enhanced.jpg --weights path/to/pretrained_weights.h5
```

Programmatic usage (Python):
```python
from model import ZeroDCEModel
from utils import load_image, save_image

img = load_image('low_light.jpg')           # HWC, float32 [0,1]
model = ZeroDCEModel()
model.load_weights('pretrained_weights.h5')
enhanced = model.enhance(img)               # returns enhanced image [0,1]
save_image('enhanced.jpg', enhanced)
```

See USAGE.md for detailed examples and options.

## Training
High-level steps:
1. Prepare a dataset of low-light images (and optional higher-quality images for evaluation).
2. Configure training parameters in `configs/train_config.yaml` or the training script arguments.
3. Run the training script:
```bash
python train.py --data_dir /path/to/dataset --epochs 100 --batch_size 8 --save_dir experiments/run1
```
Training uses zero-reference losses (e.g., spatial consistency, exposure control, color constancy). Monitor logs and checkpoints during training.

Tips:
- Normalize images to [0,1] float32.
- Use data augmentation (random crops, flips) to improve robustness.
- Start from small batch sizes if GPU memory is limited.

## Dataset
This repository does not include copyrighted datasets. Typical datasets used in low-light enhancement research:
- LOL (Low-Light) dataset
- SID (See-in-the-Dark) for extreme low-light (note: SID is raw data)
- Custom low-light photos collected from cameras or smartphones

Organize your dataset as:
```
/dataset
  /images
    img001.jpg
    img002.jpg
    ...
```

## Model & outputs
- The model outputs an enhanced RGB image in the same resolution as input.
- Checkpoints are saved as TensorFlow `.h5` or SavedModel format depending on config.
- For visual inspection, a `results/` directory will store before/after comparisons and metrics.

## Evaluation
Common evaluation metrics:
- PSNR / SSIM (when paired GT available)
- NIQE / BRISQUE / perceptual metrics for no-reference quality
- Visual inspection on diverse scenes


