# import cv2
# import numpy as np
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# from pathlib import Path

# # 1. Load your trained model
# model_path = "/home/amit/DEEP_LEARNING_PROJECT/yolo11n_p2_tiny_tiled/p2_tiny_run6/weights/best.pt"
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=model_path,
#     confidence_threshold=0.3, # You can adjust this threshold
#     device="cuda:0",
# )

# # 2. Load your large test image
# # large_image_path = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/test/images/boat579.png"
# large_image_path = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/train/images/boat591.png"

# # 3. Run sliced prediction
# result = get_sliced_prediction(
#     large_image_path,
#     detection_model,
#     slice_height=640,
#     slice_width=640,
#     overlap_height_ratio=0.25,
#     overlap_width_ratio=0.25
# )

# # --- Manual Drawing Section ---
# # Load the original image with OpenCV
# image = cv2.imread(large_image_path)
# if image is None:
#     print(f"Error: Could not load image from {large_image_path}")
#     exit()

# # Define drawing parameters
# LINE_THICKNESS = 1      # Very thin line for bounding box
# COLOR = (0, 0, 255)     # BGR for Red

# # Iterate through detections and draw them
# for obj in result.object_prediction_list:
#     bbox = obj.bbox.to_xyxy() # Get bbox in [xmin, ymin, xmax, ymax] format
#     xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

#     # Draw bounding box
#     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, LINE_THICKNESS)
    
#     # --- All text and label drawing lines have been removed ---

# # Save the output image
# output_dir = Path("test_output_manual_draw")
# output_dir.mkdir(parents=True, exist_ok=True)
# output_image_path = output_dir / Path(large_image_path).name
# cv2.imwrite(str(output_image_path), image)
# print(f"Manually drawn prediction (boxes only) saved to {output_image_path}")



#!/usr/bin/env python3
"""
predict_with_sahi.py

This script uses SAHI (Slicing Aided Hyper Inference) to run inference
on large images, which is ideal for tiny object detection.
"""

import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict

# -----------------------
# CONFIG
# -----------------------
# Path to the model you just trained
MODEL_PATH = "/home/amit/DEEP_LEARNING_PROJECT/yolov8s_p2_v1/yolov8s_p2_safe_best.pt"

# What to run inference on (can be a single image or a folder)
SOURCE_IMAGES_DIR ="/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/train/images/boat591.png"

# Where to save the results
OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/sahi_inference_results"

# Slicing parameters
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# Model parameters
CONF_THRESHOLD = 0.01  # <-- Use a very low threshold for tiny objects
IOU_THRESHOLD = 0.5    # NMS IoU

# -----------------------
# SCRIPT
# -----------------------
def main():
    print(f"Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load the Ultralytics model into SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda:0",  # or "cpu"
    )

    print(f"Running SAHI inference on: {SOURCE_IMAGES_DIR}")
    print(f"Slicing into {SLICE_WIDTH}x{SLICE_HEIGHT} patches...")
    
    # Run sliced prediction
    predict(
        model=detection_model,
        source=SOURCE_IMAGES_DIR,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        project=OUTPUT_DIR,
        name="sahi_sliced_run",
        save_images=True,       # Save images with boxes
        save_txt=True,          # Save label files
        exist_ok=True
    )

    print("\n" + "="*30)
    print("✅ SAHI Inference Complete!")
    print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'sahi_sliced_run')}")
    print("="*30)

if __name__ == "__main__":
    main()