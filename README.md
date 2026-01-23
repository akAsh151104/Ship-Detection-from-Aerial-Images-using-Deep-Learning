# 🚢 Deep Learning Project: Tiny Ship Detection

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-YOLOv8%20%7C%20SAHI-orange)

## 🌟 Project Overview

This project addresses the difficult computer vision challenge of **detecting tiny ships in large-scale satellite and aerial imagery**. 

In standard object detection tasks, objects usually occupy a significant portion of the image. However, in satellite imagery, a ship might only be a few pixels wide (e.g., 10x10 pixels) within a massive 4000x4000 pixel image. Standard models resize these images down to 640x640, causing these tiny ships to effectively vanish (becoming less than 1 pixel), rendering detection impossible.

Our solution implements a robust pipeline that combines **high-resolution training**, **architectural modifications**, and **Slicing Aided Hyper Inference (SAHI)** to detect these micro-objects with high precision.

## ⚙️ Methodology: How We Perform It

Our approach is divided into three main stages:

### 1. Data Preparation & Tiling
Instead of resizing large satellite images and losing detail, we process data by **tiling**. We split large training images into smaller, manageable chips (e.g., 640x640 or 1024x1024) while keeping the original resolution. This ensures the ships remain visible to the network.

### 2. Advanced Model Training (YOLOv8/11)
We utilize the state-of-the-art YOLO (You Only Look Once) architecture. However, we don't just use the "out-of-the-box" model. We train multiple variants, including a specialized "Tiny Object" model and a "Loss Comparison" model to find the optimal convergence strategy.

### 3. Inference with SAHI
For testing and deployment, we use **SAHI (Slicing Aided Hyper Inference)**.
*   **The Problem**: If we feed a full 4K image to YOLO, it downsamples it, killing small details.
*   **The Solution**: SAHI automatically slices the large inference image into overlapping patches (e.g., 640x640).
*   **Process**: YOLO detects objects in each individual patch.
*   **Result**: SAHI merges these detections back onto the original large image, handling Non-Maximum Suppression (NMS) to remove duplicates at the boundaries.

---

## 🔬 The "Extra Part": Optimizations for Tiny Objects

To specifically target **tiny objects (< 32x32 pixels)**, we implemented several advanced engineering optimizations beyond standard YOLO training:

### 1. Architectural Modification: The P2 Layer
Standard YOLO models detect objects at three scales: P3 (stride 8), P4 (stride 16), and P5 (stride 32).
*   **Issue**: At P3 (stride 8), an 8x8 pixel object becomes just 1 pixel in the feature map.
*   **Our Fix**: We introduced a **P2 detection head (stride 4)**. This layer retains much higher resolution feature maps, allowing the model to "see" and regress bounding boxes for objects that would otherwise disappear in deeper layers.

### 2. Extreme Training Resolution
We train our specialized models at **1280 pixels** (double the standard 640px). This quadruples the number of pixels available to the model, providing significantly more detail for feature extraction on small targets.

### 3. Aggressive Augmentation Strategy
Small objects are hard to learn because they lack texture. We force the model to learn robust features using:
*   **Copy-Paste (0.3)**: Randomly copying ships and pasting them elsewhere to increase object density.
*   **Aggressive Scaling (0.9)**: Randomly scaling images from 0.1x to 2.0x to handle scale variations.
*   **MixUp (0.15) & Mosaic (1.0)**: Combining images to prevent overfitting to background context.

### 4. Loss Function Engineering
Bounding box regression is unstable for small boxes (a 1-pixel error is a huge percentage error for a tiny box). We conducted a comparative study patching the loss function to use:
*   **GIoU (Generalized IoU)**
*   **DIoU (Distance IoU)**
*   **CIoU (Complete IoU)**
*   **Box Loss Weight Boosting**: We increased the box loss gain to **12.0** (vs standard 7.5) to force the model to prioritize localization accuracy over classification confidence.

---

## 📂 Project Structure

```
DEEP_LEARNING_PROJECT_TINY_SHIP_DETECTION/
├── train_main_yolov8l.py      # Training script comparing Loss Functions (GIoU, DIoU, CIoU)
├── train_small_object.py      # "Extreme" optimization training (P2 Layer, 1280px, High Augmentation)
├── sahi_test.py               # Inference script using SAHI for large image slicing
├── test_all.py                # Generates visual comparison grids of all trained models
├── mega_pipeline2/            # Configuration and output for tiled data processing
│   ├── data_tiled.yaml        # Dataset configuration
│   └── sahi_out_predict_*/    # Output predictions
└── README.md                  # Project documentation
```

## 💻 Execution Commands

### 1. Train for Tiny Objects (The Optimized Approach)
Run the specialized training script with P2 layer and high-res settings.
```bash
python train_small_object.py
```

### 2. Train Loss Comparison Model
Train the model to compare how different geometric loss functions affect performance.
```bash
python train_main_yolov8l.py
```

### 3. Run SAHI Inference
Detect ships on large test images using slicing.
```bash
python sahi_test.py
```

### 4. Visualize Results
Generate a grid comparing all your models.
```bash
python test_all.py
```

---
*Created for the Deep Learning Project on Tiny Ship Detection.*
