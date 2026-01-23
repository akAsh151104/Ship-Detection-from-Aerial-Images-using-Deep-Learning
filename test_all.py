#!/usr/bin/env python3
"""
Test script to compare all trained loss-function models.

1. Finds all 'yolov8l_*.pt' models in the training output directory.
2. Runs prediction with each model on a single test image.
3. ADDS NO TEXT LABELS to bounding boxes (labels=False).
4. ADDS the RED LOSS FUNCTION NAME (e.g., "GIOU") to each image.
5. Adds small padding/spacing between images in the grid.
6. Stitches all prediction images into a single comparison grid.
"""

import os
import cv2
import numpy as np
import math
from pathlib import Path
from ultralytics import YOLO
import torch

# --- Configuration ---

# 1. Directory where your 'yolov8l_*.pt' models are saved
MODELS_DIR = "/home/amit/DEEP_LEARNING_PROJECT/yolo11_loss_comparison_all_Adv6"

# 2. Path to the single image you want to test
IMAGE_TO_TEST = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/test/images/boat675.png"

# 3. Directory to save the final comparison grid
OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/yolo11_loss_comparison_all_Adv6/comparison_results"

# 4. Name for the final grid image
FINAL_GRID_IMAGE_NAME = "all_losses_comparison_grid_with_title_noboxlabels_padded.png" # Updated name

# 5. Inference parameters (same as your training)
IMG_SIZE = 640
CONF_THRESHOLD = 0.10  # Use a LOW confidence to see all detections
IOU_THRESHOLD = 0.4

# 6. Grid layout
NUM_COLS = 4
CELL_PADDING = 10 # <<< ADDED: Padding in pixels between images

# --- End Configuration ---

def create_comparison_grid():
    """
    Finds models, runs inference, and creates a comparison grid
    WITH a title on each image, WITHOUT labels on boxes, and WITH padding.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Find all trained models
    models_dir_path = Path(MODELS_DIR)
    model_paths = sorted(list(models_dir_path.glob('yolov8l_*.pt')))
    
    # Move 'native' to the front if it exists
    native_path = models_dir_path / 'yolov8l_native.pt'
    if native_path in model_paths:
        model_paths.remove(native_path)
        model_paths.insert(0, native_path)

    if not model_paths:
        print(f"❌ Error: No models found in {MODELS_DIR}")
        print("Please check the path and ensure training is complete.")
        return

    print(f"Found {len(model_paths)} models to compare:")
    for p in model_paths:
        print(f"  - {p.name}")
        
    if not Path(IMAGE_TO_TEST).exists():
        print(f"❌ Error: Test image not found at {IMAGE_TO_TEST}")
        return

    annotated_images = []
    
    print("\n" + "="*50)
    print("Running predictions...")
    print("="*50)

    # 2. Loop, predict, and annotate
    for i, model_path in enumerate(model_paths):
        model_name = model_path.stem.replace('yolov8l_', '').upper()
        print(f"Processing ({i+1}/{len(model_paths)}): {model_name} (Title: YES, Box Labels: NO)...")
        
        try:
            model = YOLO(model_path)
            
            # Run prediction
            results = model.predict(
                source=IMAGE_TO_TEST,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                save=False,
                verbose=False
            )

            img_annotated = results[0].plot(
                labels=False,
                line_width=2
            )
            
            cv2.putText(
                img_annotated,
                model_name,
                (10, 40), # Position (top-left)
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,      # Font scale
                (0, 0, 255), # Color (Red)
                3,        # Thickness
                cv2.LINE_AA
            )
            
            annotated_images.append(img_annotated)
            
        except Exception as e:
            print(f"  Failed to process {model_name}: {e}")
            # Add a placeholder error image
            img_h, img_w = (IMG_SIZE, IMG_SIZE)
            placeholder = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"ERROR: {model_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            annotated_images.append(placeholder)

    if not annotated_images:
        print("❌ No images were processed.")
        return

    # 3. Stitch images into a grid with padding
    print("\nStitching images into a grid with padding...")
    
    img_h, img_w, _ = annotated_images[0].shape
    
    # Create a blank image for padding
    padding_col = np.zeros((img_h, CELL_PADDING, 3), dtype=np.uint8)
    padding_row = np.zeros((CELL_PADDING, (img_w + CELL_PADDING) * NUM_COLS - CELL_PADDING, 3), dtype=np.uint8)
    
    num_models = len(annotated_images)
    num_rows = math.ceil(num_models / NUM_COLS)
    
    # Fill in the list with blank images to make a full grid
    total_cells = num_rows * NUM_COLS
    while len(annotated_images) < total_cells:
        # Create a blank canvas matching image size, then add error text later
        blank_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        annotated_images.append(blank_canvas) # Append a blank image

    rows_list_with_padding = []
    for r in range(num_rows):
        start_idx = r * NUM_COLS
        end_idx = start_idx + NUM_COLS
        row_images = annotated_images[start_idx:end_idx]
        
        # Horizontally stack images with padding
        row_with_padding = []
        for j, img in enumerate(row_images):
            row_with_padding.append(img)
            if j < NUM_COLS - 1: # Add padding between images, not after the last one
                row_with_padding.append(padding_col)
        
        # Concatenate images and padding for this row
        current_row_hstacked = cv2.hconcat(row_with_padding)
        rows_list_with_padding.append(current_row_hstacked)
        
        if r < num_rows - 1: # Add padding between rows, not after the last row
            rows_list_with_padding.append(padding_row)
            
    final_grid = cv2.vconcat(rows_list_with_padding)
    
    # 4. Save the final grid
    final_path = Path(OUTPUT_DIR) / FINAL_GRID_IMAGE_NAME
    cv2.imwrite(str(final_path), final_grid)

    print("\n" + "="*50)
    print("✅ Comparison Complete (Title: YES, Box Labels: NO, Padded)! Image generated:")
    print(f"Grid saved to: {final_path}")
    print("="*50)


if __name__ == "__main__":
    create_comparison_grid()