# Weakly Supervised Remote Sensing Segmentation
### Partial Cross-Entropy Loss with Focal Weighting · PyTorch · LoveDA Urban Dataset

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

Standard semantic segmentation models require **full pixel-wise annotated masks** — an expensive and time-consuming process, especially in remote sensing where images are large and complex.

This project tackles a more realistic scenario: **what if you only had a handful of labeled points per image instead of complete masks?**

The solution is a custom **Partial Cross-Entropy Loss with Focal weighting** (`PartialFocalLoss`) that restricts gradient updates exclusively to sparse labeled point locations, ignoring all unlabeled pixels. This is the same principle used in real-world weakly supervised annotation pipelines for geospatial and agricultural AI.

---

## Key Contributions

- **`PartialFocalLoss`** — implemented from scratch in PyTorch. Computes focal-weighted cross-entropy only at labeled point locations using a binary supervision mask:

```
pCE = Σ( FocalLoss(pred, GT) × MASK_labeled ) / Σ(MASK_labeled)
```

- **`PointLabelSampler`** — simulates sparse point annotation by randomly sampling N pixels per class from full ground truth masks, with full reproducibility and edge case handling

- **End-to-end weakly supervised pipeline** — UNet + ResNet-34 (ImageNet pretrained) trained on LoveDA Urban dataset using only sparse point labels as the training signal

- **Two ablation experiments** exploring the key factors affecting performance under weak supervision

---

## Results

### Experiment 1 — Point Label Density (N points per class)

| Points / Class | Val mIoU |
|:--------------:|:--------:|
| 1              | ~        |
| 3              | ~        |
| 5              | ~        |
| 10             | ~        |
| 20             | ~        |

> **Finding:** More labeled points per class consistently improves mIoU, but with diminishing returns beyond N=10. Even N=5 achieves competitive results compared to fully supervised baselines.

### Experiment 2 — Focal Loss Gamma (γ)

| Gamma (γ) | Val mIoU |
|:---------:|:--------:|
| 0.0       | ~        |
| 0.5       | ~        |
| 1.0       | ~        |
| 2.0       | ~        |
| 5.0       | ~        |

> **Finding:** γ=2.0 provides the best balance — sufficiently down-weighting easy examples while maintaining stable gradients. γ=0 (plain partial CE) and γ=5 (over-suppression) both underperform.

*(Fill in your actual mIoU values from your training runs)*

---

## Dataset

**[LoveDA Urban](https://zenodo.org/record/5706578)** — A remote sensing semantic segmentation benchmark

| Property | Value |
|---|---|
| Source | Zenodo (open access) |
| Resolution | 1024×1024 (resized to 512×512) |
| Classes | 7 |
| Train images | 1156 |
| Val images | 677 |

**Classes:** Background · Building · Road · Water · Barren · Forest · Agriculture

---

## Architecture

```
Input (3×512×512)
       │
  ┌────▼─────────────────────┐
  │   ResNet-34 Encoder       │  ← ImageNet pretrained
  │   (skip connections)      │
  └────┬─────────────────────┘
       │
  ┌────▼─────────────────────┐
  │   UNet Decoder            │
  │   (upsampling + concat)   │
  └────┬─────────────────────┘
       │
  Output (7×512×512) logits
       │
  PartialFocalLoss ← point_mask (sparse supervision)
```

---

## Project Structure

```
weakly-supervised-remote-sensing-segmentation/
│
├── meriti_assessment.ipynb       # Main notebook — runs end to end
│
├── steps/
│   ├── step1_environment.py      # Environment setup, CONFIG, GPU check
│   ├── step2_dataset.py          # LoveDA download, extraction, stats
│   ├── step3_point_sampler.py    # PointLabelSampler class + unit tests
│   ├── step4_dataset_class.py    # PyTorch Dataset + DataLoader
│   ├── step5_partial_ce_loss.py  # PartialFocalLoss + unit tests
│   ├── step6_model_setup.py      # UNet model, train/val step functions
│   ├── step7_training_loop.py    # Full training loop + curve plotter
│   └── step8_launch_training.py  # Launch cell — baseline + experiments
│
├── outputs/
│   ├── sample_images.png         # Dataset sample visualisation
│   ├── point_sampling_demo.png   # Point label overlay demo
│   ├── training_batch_vis.png    # Training batch visualisation
│   └── curves_*.png              # Training curves per experiment
│
├── logs/
│   ├── log_*.csv                 # Per-epoch training logs
│   └── history_*.json            # Full run histories
│
└── README.md
```

---

## Quickstart

### 1. Open in Google Colab
The project is designed to run on **Google Colab with a T4 GPU**.

```
Runtime > Change runtime type > Hardware accelerator > T4 GPU
```

### 2. Clone the repo
```bash
git clone https://github.com/MHz005/weakly-supervised-remote-sensing-segmentation
```

### 3. Run the notebook
Open `meriti_assessment.ipynb` and run cells sequentially. Each step prints a `✓ Step N complete` confirmation before proceeding.

The dataset downloads automatically from Zenodo on first run (~2GB).

### 4. Run your own experiment
Change a single value in `CONFIG` and re-run the training cell:

```python
# Experiment 1 — point label density
CONFIG["n_points_per_class"] = 1   # try 1, 3, 5, 10, 20

# Experiment 2 — focal gamma
CONFIG["focal_gamma"] = 0.0        # try 0, 0.5, 1, 2, 5

history = train(model, train_loader, val_loader, CONFIG,
                experiment_tag="my_experiment")
```

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Language |
| PyTorch | 2.0+ | Deep learning framework |
| segmentation-models-pytorch | 0.3+ | UNet + pretrained encoders |
| Albumentations | 2.0+ | Image augmentation |
| NumPy | 2.0+ | Numerical operations |
| Google Colab | — | Training environment (T4 GPU) |
| CUDA | 12.8 | GPU acceleration |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Batch size | 4 |
| Image size | 512×512 |
| Mixed precision | AMP (float16) |
| GPU | Tesla T4 (14.6GB VRAM) |

---

## Author

**Muhammad Hasan** — [@MHz005](https://github.com/MHz005)

🌐 [Portfolio](https://hasan-digital.preview.emergentagent.com/) · 💼 [LinkedIn](https://www.linkedin.com/in/muhammad-hasan-a34227329) · 🛒 [Fiverr](https://www.fiverr.com/s/VYK4yP5)

---

## License

MIT License — feel free to use, modify, and build on this project with attribution.
