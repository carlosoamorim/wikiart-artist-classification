# WikiArt Artist Classification with Keras

**Artist attribution from pixels alone - a deep learning approach to a genuinely hard vision problem.**

Classifying paintings by artist is not like standard object recognition. Style is a diffuse signal encoded simultaneously in brushstroke texture, colour temperature, spatial frequency, and compositional habit. This project investigates how well deep learning can learn these cues across 23 artists from the WikiArt dataset - and where it fails.

---

## Results

| Metric | Score |
|---|---|
| **Macro F1** | **0.860** |
| Accuracy | 0.874 |
| Cohen's Kappa | 0.867 |
| Macro ROC AUC | 0.991 |
| MCC | 0.867 |

Final evaluation on a held-out test set of 1,984 images. Validation macro F1 (0.862) and test macro F1 (0.860) are tightly aligned, confirming no optimistic bias from model selection.

---

## Dataset

- **Source:** [WikiArt](https://www.wikiart.org/) - 23-class subset
- **Size:** 13,172 paintings after duplicate removal (perceptual hashing + Hamming distance, threshold = 10)
- **Split:** 70% train / 15% validation / 15% test
- **Input resolution:** 224 × 224 (ImageNet-compatible)
- **Labels:** one-hot encoded for compatibility with categorical cross-entropy, label smoothing, and macro F1
- **Class imbalance:** ~3.9× ratio (Van Gogh: 1,322 images vs. Dalí: 336)
- **Imbalance handling:** balanced class weighting - loss contributions scaled inversely to class frequency, preferred over over/undersampling
- **Primary metric:** macro F1, chosen over accuracy to give equal weight to all 23 artists regardless of support

---

## Models

### 1. CNN from Scratch (Baseline)
Four-block custom CNN trained from scratch to establish a ceiling without pretrained representations.

- Architecture: 4× [Conv2D(3×3) → Conv2D(3×3) → BN → ReLU → Dropout], filter progression 32→64→128→256
- Blocks 1–3: MaxPooling(2×2); Block 4: GlobalAveragePooling
- L2 regularisation (λ = 1e-4), Adam (lr = 1e-3), label smoothing = 0.1
- Callbacks: EarlyStopping (patience=8), ReduceLROnPlateau (factor=0.5, patience=4)

**Augmentation ablation:**

| Strategy | Val F1 | Train F1 | Gap |
|---|---|---|---|
| No augmentation | 0.633 | 0.681 | 0.048 |
| Geometric (flip + rotation + zoom) | 0.458 | 0.553 | 0.095 |
| Photometric (flip + brightness±10% + contrast±10%) | 0.627 | 0.653 | 0.026 |

Geometric transforms consistently hurt - rotation and zoom destroy compositional cues that are genuine discriminative features in painting style. Photometric augmentation was carried forward for all subsequent models.

---

### 2. EfficientNetV2S - Progressive Fine-tuning
Primary backbone. 21M parameters, pretrained on ImageNet-21k. Adapted via a 3-phase progressive unfreezing strategy to minimise catastrophic forgetting.

| Phase | Unfrozen scope | LR | Val F1 |
|---|---|---|---|
| 1 | Head only (frozen backbone) | 1e-3 | 0.678 |
| 2 | + block6 layers (≥288) - SE attention | 1e-4 | 0.850 |
| 3 | + block5 layers (≥155) - mid-level textures | 5e-5 | **0.851** |

Block6 was selected first because it contains the deepest Squeeze-and-Excitation attention sub-blocks - the layers most responsible for high-level semantic representations that need adapting from natural photos to painterly features. Block5 adds mid-level texture detectors relevant to brushstroke patterns.

---

### 3. MobileNetV3Large - Ensemble Partner
5.4M parameters with depthwise separable convolutions. Selected specifically for its different inductive bias relative to EfficientNetV2S, making it likely to disagree on the cases where EfficientNetV2S is wrong.

| Phase | Unfrozen scope | LR | Val F1 |
|---|---|---|---|
| 1 | Head only | 1e-3 | 0.713 |
| 2 | Last 3 expanded conv blocks (≥140) | 1e-4 | **0.803** |

---

### 4. Ensemble (50/50 Softmax Averaging)
The two best checkpoints combined by averaging softmax output distributions.

| Configuration | Val F1 |
|---|---|
| EfficientNetV2S alone | 0.851 |
| MobileNetV3Large alone | 0.803 |
| Ensemble 60/40 (V2S-favouring) | 0.858 |
| **Ensemble 50/50** | **0.862** |
| **→ Test set (final)** | **0.860** |

Equal weighting outperforming 60/40 confirms the two models are making genuinely complementary errors. Giving more weight to the stronger model suppresses MobileNet's corrective contribution on the cases where it disagrees correctly.

---

## Nearest-Neighbour Misclassification Analysis

Beyond standard evaluation, the notebook includes a feature-space explainability tool: for each misclassified test image, the pipeline retrieves the most visually similar training image from the predicted class using cosine distance on L2-normalised ensemble features.

**How it works:**
1. Feature extractors are built from the penultimate dropout layers of both models (EfficientNetV2S: 1,280-dim; MobileNetV3Large: 960-dim)
2. Features are concatenated into a 2,240-dim vector and L2-normalised
3. A full training index is built at inference time
4. For each misclassification, the nearest neighbour is retrieved from the predicted class's subset of the index

This makes individual errors interpretable - you can see *what the model saw* that led it to confuse a Pissarro for a Monet.

---

## Failure Analysis

The confusion matrix does not reveal a single dominant failure mode. Observed confusions are largely art-historically interpretable:

- **Pissarro → Monet** (9 errors): both Impressionists, near-identical brushwork tradition
- **Dalí → scattered** (Cézanne, Monet, Roerich): Surrealism has no consistent anchor in the feature space; model defaults to surface-level cues
- **Cézanne → Van Gogh / Renoir**: shared Post-Impressionist palette and visible brushstroke structure
- **Kustodiev → Repin**: both Russian Realists with nearly identical subject matter and tonal palette
- **Roerich → Van Gogh**: shared bold, saturated colour and rhythmic landscape composition

At its boundaries, artist classification is not purely a modelling problem - it reflects genuine ambiguity in artistic style that would challenge non-specialist human observers equally.

---

## Project Structure

```
wikiart-painter-classification/
│
├── eda.ipynb                # Dataset inspection: class balance, colour analysis, duplicate detection
├── group21_models.ipynb     # Full training pipeline: CNN → fine-tuning → ensemble → evaluation
│
└── README.md
```

**`eda.ipynb`** - Run locally on the downloaded dataset before training. Covers:
- Class distribution and imbalance analysis
- Per-artist colour profile (mean RGB, warmth index, grayscale-as-RGB detection)
- Duplicate detection via perceptual hashing with threshold calibration (distance 0–10)
- Corrupted image check

Findings here directly motivated the augmentation strategy and duplicate removal threshold used in training.

**`group21_models.ipynb`** - Designed for Kaggle (P100 GPU). Sections:
1. Data loading, split verification, class distribution
2. Class weighting
3. CNN from scratch (3 augmentation ablations)
4. EfficientNetV2S (Phase 1 → 2 → 3 + regularised variant)
5. MobileNetV3Large (Phase 1 → 2 + regularised variant)
6. Ensemble evaluation (50/50 and 60/40)
7. Final test set evaluation + confusion matrix
8. Nearest-neighbour misclassification analysis

---

## Reproducing Results

All experiments run on Kaggle with fixed seeds:

```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

Dataset used: [`piccini/wikiart-clean`](https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link) on Kaggle.

**Requirements:**
```
tensorflow>=2.12
tensorflow-probability
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
imagehash
optuna
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square&logo=keras)
![Kaggle](https://img.shields.io/badge/Trained_on-Kaggle_P100-20BEFF?style=flat-square&logo=kaggle)

---

## Authors

Carlos Amorim · Duarte Neves · Margarida Ourives · Ricardo Trindade · Rodrigo Amaral

*MSc. Data Science & Advanced Analytics - Nova IMS, 2025/2026*  
*Deep Learning - Prof. Mauro Castelli*
