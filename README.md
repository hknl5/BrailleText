# BrailleText: Arabic Braille Character Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionEmpower Team** | Samsung Innovation Campus AI Capstone Project

A deep learning solution for converting Arabic Braille characters to digital text, enhancing accessibility for visually impaired Arabic speakers.

---

##  Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Team](#team)

---

##  Overview

BrailleText is an AI-powered system that automatically classifies Arabic Braille characters using Convolutional Neural Networks (CNN). The project addresses the scarcity of tools for converting Arabic Braille to text, aiming to improve literacy and accessibility for the visually impaired community.

**Key Features:**
- High-accuracy CNN model for Arabic Braille character recognition
- Support for 33 Arabic Braille character classes
- Optimized architecture with 88% accuracy on test data
- Comprehensive preprocessing and data augmentation pipeline

---

##  Background

Arabic Braille enables visually impaired individuals to read and write, yet automated conversion tools remain limited. This project leverages deep learning to bridge the gap between Braille and digital text, providing a foundation for assistive technology development.

**Motivation:**
- Enhance accessibility for Arabic-speaking visually impaired individuals
- Automate Braille-to-text conversion with high reliability
- Develop a scalable solution for educational and practical applications

---

##  Dataset

### Data Sources

The dataset combines images from two primary sources to ensure comprehensive coverage of Arabic Braille characters:

1. **[Arabic Braille Characters Dataset](https://www.kaggle.com/datasets/nagwaelaraby/arabic-braille-characters)**
   - 31 classes of Arabic Braille
   - Image dimensions: 50×50 pixels
   - Balanced representation across classes

2. **[Braille Dataset 2](https://www.kaggle.com/datasets/shemescobal/braille-dataset-2/data)**
   - 2 additional classes (ئ, ز)
   - Original dimensions: 224×224 pixels (resized to 50×50)

### Dataset Statistics

- **Total Images:** 3,267
- **Samples per Class:** 99
- **Image Size:** 50×50 pixels (RGB)
- **Number of Classes:** 33

### Data Split

| Subset | Images | Percentage |
|--------|--------|------------|
| Training | 2,112 | 64.6% |
| Validation | 528 | 16.2% |
| Testing | 627 | 19.2% |

### Preprocessing Pipeline

1. **Image Resizing:** All images standardized to 50×50 pixels
2. **Normalization:** Pixel values normalized to [0, 1] range
3. **Quality Preservation:** Pre-existing augmentations (zoom, rotation, noise) retained
4. **Batch Processing:** Default batch size of 32 images

---

##  Model Architecture

### CNN Design

The model employs a sequential CNN architecture optimized for Braille character recognition:

```
Input (50×50×3)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2) + Dropout (0.25)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2) + Dropout (0.25)
    ↓
Conv2D (128 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2) + Dropout (0.25)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU + Dropout (0.25)
    ↓
Dense (33 units) + Softmax
```

### Architecture Details

| Layer Type | Parameters | Output Shape |
|------------|------------|--------------|
| Conv2D | 32 filters, 3×3 kernel | (48, 48, 32) |
| MaxPooling2D | 2×2 pool size | (24, 24, 32) |
| Conv2D | 64 filters, 3×3 kernel | (22, 22, 64) |
| MaxPooling2D | 2×2 pool size | (11, 11, 64) |
| Conv2D | 128 filters, 3×3 kernel | (9, 9, 128) |
| MaxPooling2D | 2×2 pool size | (4, 4, 128) |
| Flatten | - | (2048,) |
| Dense | 128 units | (128,) |
| Output Dense | 33 units | (33,) |

### Training Configuration

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Epochs:** 25
- **Metric:** Accuracy
- **Regularization:** Dropout (0.25) after each layer

### Model Comparison

Two CNN variants were evaluated:

| Model | Kernel Size | Validation Accuracy | Training Time |
|-------|-------------|---------------------|---------------|
| **Model 1** | 3×3 | **98%** | **~36 seconds** |
| Model 2 | 5×5 | 96% | ~40 seconds |

**Selected Model:** Model 1 (3×3 kernel) demonstrated superior performance in both accuracy and efficiency.

---

## Results

### Performance Metrics

- **Test Accuracy:** 88%
- **Training Accuracy:** 98%
- **Training Time:** 36 seconds per epoch

### Confusion Matrix

A confusion matrix visualization reveals the model's classification performance, highlighting true positives, false positives, and false negatives across all 33 character classes.

---

##  Team

**VisionEmpower Team** - Samsung Innovation Campus AI Course

| Name | Role |
|------|------|
| **Ibrahim Almushaiqeh** | Project Lead & Coordination |
| **Alhanouf Alqahtani** | Data Collection & Preprocessing |
| **Aljawharah Alanazi** | Model Development |
| **Ibrahim Alzahem** | Model Development |
| **Batool Alsaloom** | Evaluation & Testing |
| **Reehal Alsaloom** | Evaluation & Testing |

---
##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**© 2024 VisionEmpower Team. All rights reserved.**

*This project was developed as part of the Samsung Innovation Campus AI Capstone Program.*