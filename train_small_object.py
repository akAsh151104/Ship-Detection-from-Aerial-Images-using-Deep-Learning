# # #!/usr/bin/env python3
# # """
# # EXTREME Tiny Object Detection for YOLO11
# # Optimized for very small objects (< 32x32 pixels)
# # Features:
# # - P2 detection layer (stride 4) for micro-objects
# # - Aggressive multi-scale training
# # - Enhanced augmentations for small targets
# # - Optimized anchor boxes for tiny objects
# # - Higher resolution training (1280px)
# # """

# # import os
# # import shutil
# # import time
# # import yaml
# # from pathlib import Path
# # from ultralytics import YOLO

# # # ============================
# # # CONFIG
# # # ============================
# # DATA_YAML = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/data.yaml"
# # BASE_MODEL = "yolo11n.pt"
# # OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/yolo11_tiny_object_training_2"
# # TRAIN_IMG_SIZE = 1280  # Higher resolution for tiny objects
# # BATCH_SIZE = 2         # Reduced for higher resolution

# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # ============================
# # # EXTREME Hyperparameters for VERY TINY objects
# # # ============================
# # HYP = {
# #     # ----- optimizer (more conservative for high-res) -----
# #     "lr0": 0.005,           # Lower LR for stability at 1280px
# #     "lrf": 0.005,           # Final LR
# #     "momentum": 0.937,
# #     "weight_decay": 0.0005,
# #     "warmup_epochs": 5.0,   # Longer warmup for stability
# #     "warmup_momentum": 0.5,
# #     "warmup_bias_lr": 0.05,

# #     # ----- loss weights (BOOSTED for tiny objects) -----
# #     "box": 12.0,            # ↑↑ MAXIMUM box loss weight
# #     "cls": 0.3,             # Lower class weight
# #     "dfl": 2.0,             # Higher DFL for precise localization

# #     # ----- augmentations (AGGRESSIVE for small objects) -----
# #     "degrees": 15.0,        # More rotation
# #     "translate": 0.2,       # More translation
# #     "scale": 0.9,           # AGGRESSIVE scale (0.1x to 2.0x range)
# #     "shear": 2.0,           # Small shear
# #     "perspective": 0.0005,  # Slight perspective
# #     "fliplr": 0.5,
# #     "flipud": 0.1,          # Some vertical flips
# #     "mosaic": 1.0,          # ALWAYS use mosaic
# #     "mixup": 0.15,          # Moderate mixup
# #     "copy_paste": 0.3,      # ↑ MORE copy-paste for tiny objects

# #     # HSV augmentation (moderate)
# #     "hsv_h": 0.015,
# #     "hsv_s": 0.7,
# #     "hsv_v": 0.4,

# #     # ----- Small object specific -----
# #     "erasing": 0.0,         # NO random erasing (would kill tiny objects)
# #     "crop_fraction": 1.0,   # Use full image
    
# #     # ----- training runtime -----
# #     "workers": 8,
# #     "close_mosaic": 15,     # Disable mosaic in last 15 epochs for fine-tuning
# # }

# # # ============================
# # # Create Custom Model Config with P2 Layer
# # # ============================
# # def create_p2_model_config():
# #     """
# #     Create a custom YOLO11 config with P2 layer (stride=4) for tiny objects.
# #     P2 layer detects objects as small as 4x4 pixels at 640px input.
# #     """
# #     model_cfg = """
# # # YOLO11 with P2 head for tiny object detection
# # # Ultralytics YOLO 🚀, AGPL-3.0 license

# # # Parameters
# # nc: 1  # number of classes (will be overridden by dataset)
# # scales:  # model compound scaling constants
# #   n: [0.50, 0.25, 1024]  # [depth, width, max_channels]
# #   s: [0.50, 0.50, 1024]
# #   m: [0.50, 1.00, 512]
# #   l: [1.00, 1.00, 512]   # Using 'l' variant
# #   x: [1.00, 1.50, 512]

# # # YOLO11n backbone with P2 output
# # backbone:
# #   # [from, repeats, module, args]
# #   - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
# #   - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4  ← KEEP THIS for tiny objects
# #   - [-1, 2, C3k2, [256, False, 0.25]]
# #   - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
# #   - [-1, 2, C3k2, [512, False, 0.25]]
# #   - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
# #   - [-1, 2, C3k2, [512, True, 0.25]]
# #   - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
# #   - [-1, 2, C3k2, [1024, True, 0.25]]
# #   - [-1, 1, SPPF, [1024, 5]]  # 9
# #   - [-1, 2, C2PSA, [1024, 0.25]]  # 10

# # # YOLO11n head with P2, P3, P4, P5
# # head:
# #   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
# #   - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
# #   - [-1, 2, C3k2, [512, False, 0.25]]  # 13

# #   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
# #   - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
# #   - [-1, 2, C3k2, [256, False, 0.25]]  # 16 (P3/8-small)

# #   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
# #   - [[-1, 1], 1, Concat, [1]]  # cat backbone P2
# #   - [-1, 2, C3k2, [128, False, 0.25]]  # 19 (P2/4-tiny) ← NEW HEAD

# #   - [-1, 1, Conv, [128, 3, 2]]
# #   - [[-1, 16], 1, Concat, [1]]  # cat head P3
# #   - [-1, 2, C3k2, [256, False, 0.25]]  # 22 (P3/8-small)

# #   - [-1, 1, Conv, [256, 3, 2]]
# #   - [[-1, 13], 1, Concat, [1]]  # cat head P4
# #   - [-1, 2, C3k2, [512, False, 0.25]]  # 25 (P4/16-medium)

# #   - [-1, 1, Conv, [512, 3, 2]]
# #   - [[-1, 10], 1, Concat, [1]]  # cat head P5
# #   - [-1, 2, C3k2, [1024, True, 0.25]]  # 28 (P5/32-large)

# #   - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
# # """
    
# #     config_path = Path(OUTPUT_DIR) / "yolo11l_p2.yaml"
# #     config_path.parent.mkdir(parents=True, exist_ok=True)
    
# #     with open(config_path, "w") as f:
# #         f.write(model_cfg)
    
# #     print(f"✓ Created P2 model config: {config_path}")
# #     return str(config_path)

# # # ============================
# # # Enhanced Dataset YAML
# # # ============================
# # def create_enhanced_yaml(yaml_path):
# #     """Create enhanced dataset config for tiny objects."""
# #     with open(yaml_path, "r") as f:
# #         data = yaml.safe_load(f)
    
# #     # Add small object optimizations
# #     data['overlap_mask'] = True  # Better handling of overlapping objects
# #     data['mask_ratio'] = 4       # Mask downsampling ratio
    
# #     new_yaml = yaml_path.replace(".yaml", "_tiny_enhanced.yaml")
# #     with open(new_yaml, "w") as f:
# #         yaml.dump(data, f, sort_keys=False)
    
# #     return new_yaml

# # # Prepare enhanced dataset
# # ENHANCED_DATA_YAML = create_enhanced_yaml(DATA_YAML)

# # # ============================
# # # Initialize Model with P2 Architecture
# # # ============================
# # print("\n" + "="*60)
# # print("  EXTREME TINY OBJECT DETECTION - YOLO11 + P2 LAYER")
# # print("="*60)
# # print(f"📊 Dataset: {ENHANCED_DATA_YAML}")
# # print(f"💾 Output: {OUTPUT_DIR}")
# # print(f"🖼️  Resolution: {TRAIN_IMG_SIZE}px (HIGH-RES mode)")
# # print(f"🎯 Features: P2 detection layer (stride=4) for micro-objects")
# # print("="*60 + "\n")

# # # Create P2-enabled model config
# # p2_config = create_p2_model_config()

# # # Load base model and prepare for P2 training
# # print(f"Loading base model: {BASE_MODEL}")
# # model = YOLO(BASE_MODEL)

# # # Override model architecture with P2 config
# # print(f"Applying P2 architecture from: {p2_config}")
# # model_p2 = YOLO(p2_config).load(BASE_MODEL)  # Load weights into P2 architecture

# # print("\n🚀 Starting training with EXTREME tiny object optimizations...\n")
# # t0 = time.time()

# # results = model_p2.train(
# #     data=ENHANCED_DATA_YAML,
# #     epochs=300,                      # ↑ More epochs for convergence
# #     imgsz=TRAIN_IMG_SIZE,           # 1280px for tiny objects
# #     batch=BATCH_SIZE,                # 8 (adjusted for high-res)
# #     device=0,
# #     project=OUTPUT_DIR,
# #     name="yolo11l_p2_extreme_tiny",
# #     exist_ok=True,
    
# #     # Multi-scale training (CRITICAL for tiny objects)
# #     multi_scale=True,                # Train at multiple scales
    
# #     # Optimization settings
# #     patience=80,                     # More patience
# #     save=True,
# #     save_period=10,                  # Save every 10 epochs
    
# #     # Validation settings
# #     val=True,
# #     plots=True,
    
# #     # Performance
# #     amp=True,                        # Automatic mixed precision
# #     fraction=1.0,                    # Use 100% of dataset
    
# #     # Reproducibility
# #     seed=42,
# #     deterministic=False,             # Faster training
    
# #     # Unpack all hyperparameters
# #     **HYP
# # )

# # train_time = time.time() - t0
# # hours = train_time / 3600

# # print("\n" + "="*60)
# # print(f"✅ Training Complete in {hours:.2f} hours ({train_time/60:.1f} minutes)")
# # print("="*60)

# # # ============================
# # # Save Best Model
# # # ============================
# # best = results.save_dir / "weights" / "best.pt"
# # last = results.save_dir / "weights" / "last.pt"

# # save_path = os.path.join(OUTPUT_DIR, "yolo11_p2_tiny_best.pt")
# # save_path_onnx = os.path.join(OUTPUT_DIR, "yolo11_p2_tiny_best.onnx")

# # if os.path.exists(best):
# #     shutil.copy(best, save_path)
# #     print(f"✓ Saved best model: {save_path}")
# # else:
# #     shutil.copy(last, save_path)
# #     print(f"⚠️  Using last checkpoint: {save_path}")

# # # ============================
# # # Comprehensive Validation
# # # ============================
# # print("\n" + "="*60)
# # print("📊 Running Comprehensive Validation...")
# # print("="*60)

# # # Load best model for validation
# # best_model = YOLO(save_path)

# # # Validate at training resolution
# # print(f"\n1️⃣  Validation at {TRAIN_IMG_SIZE}px (training resolution)...")
# # val_high = best_model.val(
# #     data=ENHANCED_DATA_YAML,
# #     imgsz=TRAIN_IMG_SIZE,
# #     batch=4,
# #     conf=0.001,              # Very low conf for tiny objects
# #     iou=0.6,
# #     verbose=True
# # )

# # # Validate at multiple scales
# # print(f"\n2️⃣  Validation at 1920px (ultra high-res)...")
# # val_ultra = best_model.val(
# #     data=ENHANCED_DATA_YAML,
# #     imgsz=1920,
# #     batch=2,
# #     conf=0.001,
# #     iou=0.6,
# #     verbose=True
# # )

# # # Print comprehensive results
# # print("\n" + "="*60)
# # print("📈 VALIDATION RESULTS - TINY OBJECT DETECTION")
# # print("="*60)

# # metrics_high = val_high.results_dict
# # metrics_ultra = val_ultra.results_dict

# # print(f"\n🎯 At {TRAIN_IMG_SIZE}px (Training Resolution):")
# # print(f"   Precision:  {metrics_high.get('metrics/precision(B)', 0):.4f}")
# # print(f"   Recall:     {metrics_high.get('metrics/recall(B)', 0):.4f}")
# # print(f"   mAP50:      {metrics_high.get('metrics/mAP50(B)', 0):.4f}")
# # print(f"   mAP50-95:   {metrics_high.get('metrics/mAP50-95(B)', 0):.4f}")

# # print(f"\n🎯 At 1920px (Ultra High-Res):")
# # print(f"   Precision:  {metrics_ultra.get('metrics/precision(B)', 0):.4f}")
# # print(f"   Recall:     {metrics_ultra.get('metrics/recall(B)', 0):.4f}")
# # print(f"   mAP50:      {metrics_ultra.get('metrics/mAP50(B)', 0):.4f}")
# # print(f"   mAP50-95:   {metrics_ultra.get('metrics/mAP50-95(B)', 0):.4f}")

# # print("\n" + "="*60)

# # # ============================
# # # Export Model (Optional)
# # # ============================
# # print("\n📦 Exporting model to ONNX for deployment...")
# # try:
# #     best_model.export(format="onnx", imgsz=TRAIN_IMG_SIZE, dynamic=True)
# #     print(f"✓ ONNX model exported")
# # except Exception as e:
# #     print(f"⚠️  ONNX export failed: {e}")

# # # ============================
# # # Usage Instructions
# # # ============================
# # print("\n" + "="*60)
# # print("🎉 TINY OBJECT DETECTION MODEL READY!")
# # print("="*60)
# # print(f"\n📁 Model saved to: {save_path}")
# # print(f"\n💡 INFERENCE TIPS for VERY SMALL OBJECTS:")
# # print(f"   1. Use HIGH resolution: imgsz=1920 or higher")
# # print(f"   2. Use LOW confidence: conf=0.001 to 0.01")
# # print(f"   3. Use NMS threshold: iou=0.6 or lower")
# # print(f"   4. For large images, use SAHI (slicing aided inference):")
# # print(f"      pip install sahi")
# # print(f"      sahi predict --slice_width 640 --slice_height 640")
# # print(f"\n📝 Example inference code:")
# # print(f"   from ultralytics import YOLO")
# # print(f"   model = YOLO('{save_path}')")
# # print(f"   results = model.predict('image.jpg', imgsz=1920, conf=0.001, iou=0.6)")
# # print(f"   results[0].show()  # Display results")
# # print("\n" + "="*60 + "\n")



# """
# train_p2_model.py

# Optimized training script for tiny objects using a P2 model.
# - Uses YOLOv8s-P2 (designed for small objects)
# - Uses 640px training (optimial for P2)
# - Larger batch size for stable training
# - AMP enabled
# - Injects tiny anchors into dataset YAML (safe, good for v5-style models)
# - Gentle but effective augmentations for small objects
# """

# import os
# import shutil
# import time
# import yaml
# from pathlib import Path
# import torch
# from ultralytics import YOLO

# # -----------------------
# # USER CONFIG (editable)
# # -----------------------
# DATA_YAML = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/data.yaml"  # original dataset YAML
# # Prefer a P2 variant for tiny objects; n-p2 is widely available
# BASE_MODEL = "yolov8s-p2.pt"  # will auto-map to 'yolov8n-p2.pt' if unavailable
# OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/yolov8s_p2_v1"
# TRAIN_IMG_SIZE = 640      # 640 is recommended for P2 models
# BATCH_SIZE = 8            # INCREASED BATCH SIZE (try 16 if VRAM allows)
# EPOCHS = 200
# SEED = 42
# WORKERS = 6

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------
# # Environment memory helpers
# # -----------------------
# # Reduce fragmentation risk
# # Use the new allocator config env var (avoids deprecation warning)
# os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
# torch.cuda.empty_cache()

# # -----------------------
# # Hyperparameters (safe + tiny-object focused)
# # These are still good and can be used directly
# # -----------------------
# HYP = {
#     # optimizer
#     "lr0": 0.005,         # conservative initial LR for stability
#     "lrf": 0.01,
#     "momentum": 0.937,
#     "weight_decay": 0.0005,
#     "warmup_epochs": 3.0,

#     # loss weighting (slightly higher box weight helps tiny objects)
#     "box": 8.0,
#     "cls": 0.3,
#     "dfl": 1.5,

#     # augmentations
#     "degrees": 6.0,
#     "translate": 0.06,
#     "scale": 0.5,
#     "shear": 0.0,
#     "perspective": 0.0005,
#     "fliplr": 0.5,
#     "flipud": 0.0,
#     "mosaic": 0.5,        # reduced mosaic (0.0-1.0)
#     "mixup": 0.05,
#     "copy_paste": 0.1,
#     "hsv_h": 0.015,
#     "hsv_s": 0.7,
#     "hsv_v": 0.4,

#     # small-object specific
#     # label_smoothing is deprecated in recent Ultralytics; omit to avoid warnings
#     "close_mosaic": 15,   # disable mosaic in last N epochs
# }

# # -----------------------
# # Create Custom YOLO11 P2 Model Config (adds stride-4 detection head)
# # -----------------------
# def create_p2_model_config(output_dir: str) -> str:
#         """
#         Create a YOLO11 model YAML with an added P2 (stride=4) detection layer
#         for improved tiny object detection.

#         Returns the path to the generated YAML.
#         """
#         model_cfg = """
# # YOLO11 with P2 head for tiny object detection
# # Ultralytics YOLO 🚀

# # Parameters
# nc: 1  # overridden by dataset YAML
# scales:
#     n: [0.50, 0.25, 1024]
#     s: [0.50, 0.50, 1024]
#     m: [0.50, 1.00, 512]
#     l: [1.00, 1.00, 512]
#     x: [1.00, 1.50, 512]

# # Backbone
# backbone:
#     - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
#     - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4  ← keep P2 for tiny objects
#     - [-1, 2, C3k2, [256, False, 0.25]]
#     - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
#     - [-1, 2, C3k2, [512, False, 0.25]]
#     - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
#     - [-1, 2, C3k2, [512, True, 0.25]]
#     - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
#     - [-1, 2, C3k2, [1024, True, 0.25]]
#     - [-1, 1, SPPF, [1024, 5]]
#     - [-1, 2, C2PSA, [1024, 0.25]]

# # Head with P2, P3, P4, P5
# head:
#     - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#     - [[-1, 6], 1, Concat, [1]]            # +P4
#     - [-1, 2, C3k2, [512, False, 0.25]]

#     - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#     - [[-1, 4], 1, Concat, [1]]            # +P3
#     - [-1, 2, C3k2, [256, False, 0.25]]    # (P3/8)

#     - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#     - [[-1, 1], 1, Concat, [1]]            # +P2
#     - [-1, 2, C3k2, [128, False, 0.25]]    # (P2/4) ← NEW

#     - [-1, 1, Conv, [128, 3, 2]]
#     - [[-1, 16], 1, Concat, [1]]           # +P3
#     - [-1, 2, C3k2, [256, False, 0.25]]    # (P3/8)

#     - [-1, 1, Conv, [256, 3, 2]]
#     - [[-1, 13], 1, Concat, [1]]           # +P4
#     - [-1, 2, C3k2, [512, False, 0.25]]    # (P4/16)

#     - [-1, 1, Conv, [512, 3, 2]]
#     - [[-1, 10], 1, Concat, [1]]           # +P5
#     - [-1, 2, C3k2, [1024, True, 0.25]]    # (P5/32)

#     - [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
# """

#         out_dir = Path(output_dir)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         cfg_path = out_dir / "yolo11_p2.yaml"
#         with open(cfg_path, "w") as f:
#                 f.write(model_cfg)
#         print(f"✓ Created P2 model config: {cfg_path}")
#         return str(cfg_path)

# # -----------------------
# # Tiny anchors injection: create a new YAML with anchors for tiny objects
# # This is safe. It will be used by anchor-based models (like v5)
# # and ignored by anchor-free models (like v8).
# # -----------------------
# def inject_tiny_anchors(yaml_path: str) -> str:
#     """
#     Read dataset yaml, add a tiny-anchor list (key 'anchors') and write a new yaml.
#     Returns path to the new yaml.
#     """
#     p = Path(yaml_path)
#     if not p.exists():
#         raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

#     with open(p, "r") as f:
#         data = yaml.safe_load(f)

#     # Provide tiny anchors (width,height pairs).
#     tiny_anchors = [
#         [4, 6, 8, 10, 12, 14],
#         [18, 22, 28, 32, 40, 44],
#         [56, 64, 80, 96, 128, 144],
#     ]
    
#     data2 = dict(data)  # shallow copy
#     data2.setdefault("anchors", tiny_anchors)

#     new_yaml = str(p.with_name(p.stem + "_tiny_anchors.yaml"))
#     with open(new_yaml, "w") as f:
#         yaml.dump(data2, f, sort_keys=False)

#     print(f"✓ Tiny-anchors dataset YAML created: {new_yaml}")
#     return new_yaml

# # -----------------------
# # Safety: check GPU memory availability and warn
# # -----------------------
# def check_gpu():
#     if not torch.cuda.is_available():
#         print("⚠️  CUDA not available. Training will run on CPU (very slow).")
#         return
#     info = torch.cuda.get_device_properties(0)
#     total_gb = info.total_memory / (1024 ** 3)
#     print(f"GPU detected: {info.name} — {total_gb:.1f} GB total VRAM")
#     # free memory (best-effort)
#     torch.cuda.empty_cache()

# # -----------------------
# # Training pipeline
# # -----------------------
# def main():
#     check_gpu()
#     # Create enhanced dataset YAML with tiny anchors
#     data_yaml = inject_tiny_anchors(DATA_YAML)

#     # Resolve a valid base model path or fallback to an available local checkpoint
#     def _resolve_base_model(preferred: str) -> str:
#         # Absolute paths to likely available checkpoints in this workspace
#         workspace_models = [
#             "/home/amit/DEEP_LEARNING_PROJECT/yolo11n.pt",
#             "/home/amit/DEEP_LEARNING_PROJECT/yolo11l.pt",
#             "/home/amit/DEEP_LEARNING_PROJECT/yolov8n.pt",
#             "/home/amit/DEEP_LEARNING_PROJECT/yolov8l.pt",
#             "/home/amit/DEEP_LEARNING_PROJECT/yolov8x.pt",
#         ]

#         # If the preferred is an absolute path and exists, use it
#         if os.path.isabs(preferred) and os.path.exists(preferred):
#             return preferred

#         # If the preferred is a relative filename that exists in CWD, use it
#         rel_path = os.path.join(os.getcwd(), preferred)
#         if os.path.exists(rel_path):
#             return rel_path

#         # Map less-common aliases to stable, available ones
#         if preferred == "yolov8s-p2.pt":
#             # Prefer a local checkpoint if present to avoid download issues
#             local_default = "/home/amit/DEEP_LEARNING_PROJECT/yolo11n.pt"
#             if os.path.exists(local_default):
#                 print("⚠️  'yolov8s-p2.pt' not found. Using local 'yolo11n.pt' instead.")
#                 return local_default
#             print("⚠️  'yolov8s-p2.pt' not found. Falling back to 'yolov8n.pt'.")
#             preferred = "yolov8n.pt"

#         # If the preferred is a known Ultralytics alias (e.g., 'yolov8n.pt'), let Ultralytics fetch it
#         # Supported aliases include P2 variant for tiny objects as well
#         known_aliases = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
#                          "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}
#         if preferred in known_aliases:
#             return preferred  # Ultralytics will download if missing

#         # Otherwise, fallback to the first existing local checkpoint from our workspace list
#         for p in workspace_models:
#             if os.path.exists(p):
#                 print(f"⚠️  '{preferred}' not found. Falling back to: {p}")
#                 return p

#         # Last resort: use a widely available alias
#         print(f"⚠️  '{preferred}' not found and no local checkpoints detected. Falling back to 'yolov8n.pt'.")
#         return "yolov8n.pt"

#     resolved_model = _resolve_base_model(BASE_MODEL)
#     # Build P2 architecture and load available pretrained weights into it
#     p2_cfg = create_p2_model_config(OUTPUT_DIR)
#     print(f"Loading base model (transferred into P2 arch): {resolved_model}")
#     model = YOLO(p2_cfg).load(resolved_model)

#     # Print important config
#     print("Training config:")
#     print(f"  output: {OUTPUT_DIR}")
#     print(f"  imgsz:  {TRAIN_IMG_SIZE}")
#     print(f"  batch:  {BATCH_SIZE}")
#     print(f"  epochs: {EPOCHS}")
#     print(f"  base model: {BASE_MODEL}")
#     print(f"  device: GPU (if available)")

#     # Clear caches before starting
#     torch.cuda.empty_cache()

#     t0 = time.time()
#     # Train
#     results = model.train(
#         data=data_yaml,
#         epochs=EPOCHS,
#         imgsz=TRAIN_IMG_SIZE,
#         batch=BATCH_SIZE,
#         project=OUTPUT_DIR,
#         name="yolov8s_p2_safe", # Updated name
#         exist_ok=True,
#         seed=SEED,
#         workers=WORKERS,
#         plots=True,
#         patience=50,
#         amp=True,              # mixed precision (reduces memory & speeds up)
#         multi_scale=True,      # ENABLED multi-scale for better accuracy
#         save=True,
#         device=0,
#         # pass the custom hyperparameters block into trainer
#         **HYP
#     )
#     train_time = time.time() - t0
#     print(f"✓ Training finished in {train_time/3600:.2f} hours")

#     # Save best/last checkpoint to a well-known path
#     best_path = results.save_dir / "weights" / "best.pt"
#     last_path = results.save_dir / "weights" / "last.pt"
#     dest = Path(OUTPUT_DIR) / "yolov8s_p2_safe_best.pt" # Updated name
    
#     if best_path.exists():
#         shutil.copy(best_path, dest)
#         print(f"✓ Best model saved to: {dest}")
#     elif last_path.exists():
#         shutil.copy(last_path, dest)
#         print(f"⚠️  Best model missing; copied last.pt to: {dest}")
#     else:
#         print("✗ No checkpoint found to save.")

#     # Run validation at training resolution with a low conf for tiny objects
#     try:
#         print("→ Running validation on test split (training resolution) with low conf for tiny objects...")
#         best_model = YOLO(str(dest))
#         val = best_model.val(
#             data=data_yaml,
#             split='test',
#             imgsz=TRAIN_IMG_SIZE, # Updated imgsz
#             batch=max(1, BATCH_SIZE//1), 
#             conf=0.001, 
#             iou=0.6, 
#             verbose=True
#         )
#         md = getattr(val, "results_dict", None) or {}
#         print("Validation metrics (training resolution):")
#         print(f"  Precision:  {md.get('metrics/precision(B)', 'N/A')}")
#         print(f"  Recall:     {md.get('metrics/recall(B)', 'N/A')}")
#         print(f"  mAP50:      {md.get('metrics/mAP50(B)', 'N/A')}")
#         print(f"  mAP50-95:   {md.get('metrics/mAP50-95(B)', 'N/A')}")
#     except Exception as e:
#         print(f"⚠️  Validation failed: {e}")

#     # Optional: quick inference example
#     print("\nInference example (adjust paths):")
#     print(f"  from ultralytics import YOLO")
#     print(f"  model = YOLO('{dest}')")
#     print(f"  model.predict(source='your_test_images_folder_or.jpg', imgsz={TRAIN_IMG_SIZE}, conf=0.01, iou=0.3, augment=True, save=True)")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
train_p2_model_1280px.py

Optimized training script for tiny objects using a P2 model.
- Uses YOLOv8s-P2 (designed for small objects)
- Uses 1280px training (HIGH RESOLUTION for tiny objects)
- Smaller batch size for VRAM stability
- AMP enabled
- Injects tiny anchors into dataset YAML
- More aggressive 'copy_paste' augmentation
"""

import os
import shutil
import time
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

# -----------------------
# USER CONFIG (editable) - MODIFIED
# -----------------------
DATA_YAML = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/data.yaml"  # original dataset YAML
BASE_MODEL = "yolov8s-p2.pt"  # will auto-map to 'yolov8n-p2.pt' if unavailable
OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/yolov8s_p2_v2_1280px" # New output dir
TRAIN_IMG_SIZE = 1280         # <-- CHANGED: Higher resolution for tiny objects
BATCH_SIZE = 2                # <-- CHANGED: Reduced for 1280px VRAM
EPOCHS = 200
SEED = 42
WORKERS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Environment memory helpers
# -----------------------
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
torch.cuda.empty_cache()

# -----------------------
# Hyperparameters (safe + tiny-object focused) - MODIFIED
# -----------------------
HYP = {
    # optimizer
    "lr0": 0.005,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,

    # loss weighting (slightly higher box weight helps tiny objects)
    "box": 8.0,
    "cls": 0.3,
    "dfl": 1.5,

    # augmentations
    "degrees": 6.0,
    "translate": 0.06,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0005,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.5,
    "mixup": 0.05,
    "copy_paste": 0.3,      # <-- CHANGED: Increased from 0.1
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,

    # small-object specific
    "close_mosaic": 15,
}

# -----------------------
# Create Custom YOLO11 P2 Model Config (adds stride-4 detection head)
# (This function is identical to your script)
# -----------------------
def create_p2_model_config(output_dir: str) -> str:
        """
        Create a YOLO11 model YAML with an added P2 (stride=4) detection layer
        """
        model_cfg = """
# YOLO11 with P2 head for tiny object detection
nc: 1
scales:
    n: [0.50, 0.25, 1024]
    s: [0.50, 0.50, 1024]
    m: [0.50, 1.00, 512]
    l: [1.00, 1.00, 512]
    x: [1.00, 1.50, 512]

backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 2, C3k2, [256, False, 0.25]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 2, C3k2, [512, False, 0.25]]
    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 2, C3k2, [512, True, 0.25]]
    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 2, C3k2, [1024, True, 0.25]]
    - [-1, 1, SPPF, [1024, 5]]
    - [-1, 2, C2PSA, [1024, 0.25]]

head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 2, C3k2, [512, False, 0.25]]
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]]
    - [-1, 2, C3k2, [256, False, 0.25]]
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 1], 1, Concat, [1]]
    - [-1, 2, C3k2, [128, False, 0.25]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [[-1, 16], 1, Concat, [1]]
    - [-1, 2, C3k2, [256, False, 0.25]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 13], 1, Concat, [1]]
    - [-1, 2, C3k2, [512, False, 0.25]]
    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 10], 1, Concat, [1]]
    - [-1, 2, C3k2, [1024, True, 0.25]]
    - [[19, 22, 25, 28], 1, Detect, [nc]]
"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "yolo11_p2.yaml"
        with open(cfg_path, "w") as f:
                f.write(model_cfg)
        print(f"✓ Created P2 model config: {cfg_path}")
        return str(cfg_path)

# -----------------------
# Tiny anchors injection
# (This function is identical to your script)
# -----------------------
def inject_tiny_anchors(yaml_path: str) -> str:
    """
    Read dataset yaml, add a tiny-anchor list (key 'anchors') and write a new yaml.
    Returns path to the new yaml.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    tiny_anchors = [
        [4, 6, 8, 10, 12, 14],
        [18, 22, 28, 32, 40, 44],
        [56, 64, 80, 96, 128, 144],
    ]
    
    data2 = dict(data)  # shallow copy
    data2.setdefault("anchors", tiny_anchors)

    new_yaml = str(p.with_name(p.stem + "_tiny_anchors.yaml"))
    with open(new_yaml, "w") as f:
        yaml.dump(data2, f, sort_keys=False)

    print(f"✓ Tiny-anchors dataset YAML created: {new_yaml}")
    return new_yaml

# -----------------------
# Safety: check GPU memory
# (This function is identical to your script)
# -----------------------
def check_gpu():
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training will run on CPU (very slow).")
        return
    info = torch.cuda.get_device_properties(0)
    total_gb = info.total_memory / (1024 ** 3)
    print(f"GPU detected: {info.name} — {total_gb:.1f} GB total VRAM")
    torch.cuda.empty_cache()

# -----------------------
# Training pipeline
# (This function is identical to your script, with updated names)
# -----------------------
def main():
    check_gpu()
    data_yaml = inject_tiny_anchors(DATA_YAML)

    def _resolve_base_model(preferred: str) -> str:
        workspace_models = [
            "/home/amit/DEEP_LEARNING_PROJECT/yolo11n.pt",
            "/home/amit/DEEP_LEARNING_PROJECT/yolo11l.pt",
            "/home/amit/DEEP_LEARNING_PROJECT/yolov8n.pt",
            "/home/amit/DEEP_LEARNING_PROJECT/yolov8l.pt",
            "/home/amit/DEEP_LEARNING_PROJECT/yolov8x.pt",
        ]
        if os.path.isabs(preferred) and os.path.exists(preferred):
            return preferred
        rel_path = os.path.join(os.getcwd(), preferred)
        if os.path.exists(rel_path):
            return rel_path
        if preferred == "yolov8s-p2.pt":
            local_default = "/home/amit/DEEP_LEARNING_PROJECT/yolo11n.pt"
            if os.path.exists(local_default):
                print("⚠️  'yolov8s-p2.pt' not found. Using local 'yolo11n.pt' instead.")
                return local_default
            print("⚠️  'yolov8s-p2.pt' not found. Falling back to 'yolov8n.pt'.")
            preferred = "yolov8n.pt"
        known_aliases = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                         "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}
        if preferred in known_aliases:
            return preferred
        for p in workspace_models:
            if os.path.exists(p):
                print(f"⚠️  '{preferred}' not found. Falling back to: {p}")
                return p
        print(f"⚠️  '{preferred}' not found and no local checkpoints detected. Falling back to 'yolov8n.pt'.")
        return "yolov8n.pt"

    resolved_model = _resolve_base_model(BASE_MODEL)
    p2_cfg = create_p2_model_config(OUTPUT_DIR)
    print(f"Loading base model (transferred into P2 arch): {resolved_model}")
    model = YOLO(p2_cfg).load(resolved_model)

    print("Training config:")
    print(f"  output: {OUTPUT_DIR}")
    print(f"  imgsz:  {TRAIN_IMG_SIZE} (High-Res)")
    print(f"  batch:  {BATCH_SIZE} (Low-Batch)")
    print(f"  epochs: {EPOCHS}")
    print(f"  base model: {BASE_MODEL}")
    print(f"  device: GPU (if available)")

    torch.cuda.empty_cache()

    t0 = time.time()
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=TRAIN_IMG_SIZE, # <-- Uses the 1280 setting
        batch=BATCH_SIZE,     # <-- Uses the 2 setting
        project=OUTPUT_DIR,
        name="yolov8s_p2_1280px", # <-- Updated name
        exist_ok=True,
        seed=SEED,
        workers=WORKERS,
        plots=True,
        patience=50,
        amp=True,
        multi_scale=True,
        save=True,
        device=0,
        **HYP
    )
    train_time = time.time() - t0
    print(f"✓ Training finished in {train_time/3600:.2f} hours")

    best_path = results.save_dir / "weights" / "best.pt"
    last_path = results.save_dir / "weights" / "last.pt"
    dest = Path(OUTPUT_DIR) / "yolov8s_p2_1280px_best.pt" # <-- Updated name
    
    if best_path.exists():
        shutil.copy(best_path, dest)
        print(f"✓ Best model saved to: {dest}")
    elif last_path.exists():
        shutil.copy(last_path, dest)
        print(f"⚠️  Best model missing; copied last.pt to: {dest}")
    else:
        print("✗ No checkpoint found to save.")

    try:
        print("→ Running validation on test split (training resolution) with low conf...")
        best_model = YOLO(str(dest))
        val = best_model.val(
            data=data_yaml,
            split='test',
            imgsz=TRAIN_IMG_SIZE, # <-- Validates at 1280
            batch=max(1, BATCH_SIZE//1), 
            conf=0.001, 
            iou=0.6, 
            verbose=True
        )
        md = getattr(val, "results_dict", None) or {}
        print("Validation metrics (1280px resolution):")
        print(f"  Precision:  {md.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall:     {md.get('metrics/recall(B)', 'N/A')}")
        print(f"  mAP50:      {md.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95:   {md.get('metrics/mAP50-95(B)', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Validation failed: {e}")

    print("\nInference example (standard):")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{dest}')")
    print(f"  model.predict(source='image.jpg', imgsz={TRAIN_IMG_SIZE}, conf=0.01, iou=0.3)")
    print("\nRECOMMENDATION: Use SAHI for better tiny object inference (see next script).")


if __name__ == "__main__":
    main()