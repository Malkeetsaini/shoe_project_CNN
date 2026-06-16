# Multi-Task Shoe Classification using ResNet18

## Overview

This project is a Deep Learning based Multi-Task Image Classification system built using PyTorch and ResNet18.

The model predicts three different outputs from a single shoe image:

1. Main Category
2. Subcategory
3. Shoe Type

The project uses Transfer Learning and Fine-Tuning with a pretrained ResNet18 model.

---

# Features

## Main Category Prediction

Predicts:

* Boots
* Shoes
* Slippers

---

## Subcategory Prediction

Predicts:

* Over the Knee
* Boat Shoes
* Slipper Flats
* Clogs and Mules
* Prewalker Boots

---

## Shoe Type Prediction

Predicts:

* sneaker
* formal
* sandal
* boot
* slipper

---

# Tech Stack

* Python
* PyTorch
* Torchvision
* ResNet18
* Google Colab

---

# Project Architecture

```text
Input Image
      ↓
ResNet18 Backbone
      ↓
Shared Feature Extractor
      ↓
 ┌──────────────┬──────────────┬──────────────┐
 │ Category Head│ Subcategory  │ Shoe Type    │
 │              │ Head         │ Head         │
 └──────────────┴──────────────┴──────────────┘
```

---

# Folder Structure

```text
shoe_project/

│
├── data/
│   ├── Boots/
│   ├── Shoes/
│   └── Slippers/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   └── config.py
│
├── saved_models/
│
└── README.md
```

---

# Dataset Structure

```text
Boots/
    Over the Knee/
        ALDO/
        Cole Haan/

Shoes/
    Boat Shoes/
        ECCO/
        Vans/

Slippers/
    Slipper Flats/
        UGG/
        Columbia/
```

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd shoe_project
```

---

## Install Dependencies

```bash
pip install torch torchvision tqdm scikit-learn
```

---

# Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

# Training

Go to source folder:

```bash
cd src
```

Run training:

```bash
python train.py
```

The trained model will be saved inside:

```text
saved_models/model.pth
```

---

# Prediction

Run prediction:

```bash
python predict.py
```

Example Output:

```text
Category: Boots
Subcategory: Over the Knee
Type: boot
```

---

# Model Details

## Backbone

* ResNet18 (Pretrained on ImageNet)

## Learning Type

* Transfer Learning
* Fine-Tuning

## Loss Function

* CrossEntropyLoss

## Optimizer

* Adam

---

# Data Augmentation

Recommended augmentations:

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
```

---

# Future Improvements

* EfficientNet
* Vision Transformers (ViT)
* Similarity Search
* Recommendation System
* CLIP Embeddings
* FAISS Vector Database
* FastAPI Deployment
* Streamlit UI

---

# Real World Use Cases

* Fashion Recommendation Systems
* Ecommerce Product Classification
* Shoe Search Engines
* Visual Product Discovery
* AI Fashion Assistants

---

# Results

This project demonstrates how a single CNN backbone can solve multiple classification tasks simultaneously using Multi-Task Learning.

---

# Author

Malkeet Saini

---

# License

This project is open-source and available under the MIT License.
