#!/usr/bin/env python3
"""
YOLOv8l Training with Different Box Loss Functions
Compares GIoU, DIoU, CIoU by patching the box regression loss.
Writes a single CSV summary with metrics and paths.
"""

import os
import csv
import shutil
import time
import inspect
import re
import random
from datetime import datetime
import torch
from ultralytics import YOLO

# --- Configuration ---
DATA_YAML = '/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/data.yaml'

# Use an Ultralytics YOLOv8 checkpoint for stability with loss patching
BASE_MODEL = 'yolo11l.pt'
# -----------------------------------------------------

EPOCHS = 300
IMG_SIZE = 640
BATCH_SIZE = 16
# Updated output directory for this experiment (as requested)
OUTPUT_DIR = '/home/amit/DEEP_LEARNING_PROJECT/yolo11_loss_comparison_all_Adv6'

# Fixed global seed for fair comparisons across variants
GLOBAL_SEED = 42

# Standard hyperparameters for all runs
# These weights (box, cls, dfl) are applied by the Ultralytics
# trainer *after* the loss function returns its raw value.
HYPERPARAMS = {
    # Optimization
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,

    # Loss weights
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    # Augmentations
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'flipud': 0.0,
    'mosaic': 1.0,
    'mixup': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,

    # Runtime
    'device': '0',
    'workers': 8,
}

# These are the 'iou_type' values supported
SUPPORTED_LOSS_TYPES = [
    # Baseline (unpatched) to anchor the comparison
    'Native',
    # Base variants
    'IoU',
    'GIoU',
    'DIoU',
    'CIoU',
    'EIoU',
    'SIoU',
    'WIoU@2.0',
    'AlphaIoU@2.0',
    'FocalIoU@2.0',
    'GWD',
    'WIoUv1',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV summary path
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'evaluation_summary.csv')


def _slugify(text: str) -> str:
    """Sanitize a string for filesystem paths: lowercase, replace non-alnum with underscores."""
    text = str(text)
    text = text.replace(' ', '_')
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text


def set_global_seed(seed: int = 42):
    """Set seeds for reproducibility across Python, NumPy, and Torch."""
    try:
        import numpy as np
    except Exception:
        np = None
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# -------------------------
# IoU variant implementations (vectorized torch, x1y1x2y2)
# (All functions from iou_giou to iou_wiou_v1 are unchanged)
# -------------------------
def _box_iou_basic(box1, box2, eps=1e-7):
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + eps
    iou = inter / union
    return iou, inter, union


def iou_giou(box1, box2, eps=1e-7):
    iou, inter, union = _box_iou_basic(box1, box2, eps)
    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    c = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
    giou = iou - (c - union) / c
    return giou.clamp(-1.0, 1.0)


def iou_diou(box1, box2, eps=1e-7):
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
    diou = iou - rho2 / c2
    return diou.clamp(-1.0, 1.0)


def iou_ciou(box1, box2, eps=1e-7):
    import math
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
    diou_penalty = rho2 / c2
    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)
    ciou = iou - diou_penalty - alpha * v
    return ciou.clamp(-1.0, 1.0)


def iou_plain(box1, box2, eps=1e-7):
    """Plain IoU similarity in [0, 1]."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    return iou.clamp(0.0, 1.0)


def iou_eiou(box1, box2, eps=1e-7):
    """Efficient IoU approximation."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)

    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2

    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    cw = (c_x2 - c_x1).clamp(min=eps)
    ch = (c_y2 - c_y1).clamp(min=eps)

    dx2 = (cx1 - cx2) ** 2
    dy2 = (cy1 - cy2) ** 2
    dw2 = (w1 - w2) ** 2
    dh2 = (h1 - h2) ** 2

    pen_center = (dx2 / (cw ** 2 + eps)) + (dy2 / (ch ** 2 + eps))
    pen_size = (dw2 / (cw ** 2 + eps)) + (dh2 / (ch ** 2 + eps))
    eiou = iou - (pen_center + pen_size)
    return eiou.clamp(-1.0, 1.0)


def iou_alpha(box1, box2, alpha=2.0, eps=1e-7):
    """Alpha-IoU score: IoU^alpha to emphasize higher overlaps."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    return torch.pow(iou.clamp(0.0, 1.0), float(alpha)).clamp(0.0, 1.0)


def iou_focal(box1, box2, gamma=2.0, eps=1e-7):
    """Focal-IoU style similarity: IoU^gamma (alias of AlphaIoU)."""
    return iou_alpha(box1, box2, alpha=gamma, eps=eps)


def iou_siou(box1, box2, eps=1e-7):
    """Simplified SIoU-like similarity."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)

    cx1 = (box1[:, 0] + box1[:, 2]) / 2
    cy1 = (box1[:, 1] + box1[:, 3]) / 2
    cx2 = (box2[:, 0] + box2[:, 2]) / 2
    cy2 = (box2[:, 1] + box2[:, 3]) / 2

    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    cw = (c_x2 - c_x1).clamp(min=eps)
    ch = (c_y2 - c_y1).clamp(min=eps)

    dx = (cx2 - cx1).abs()
    dy = (cy2 - cy1).abs()
    angle = torch.atan2(dy, dx + eps)  # [0, pi/2]

    dist_cost = (dx / cw + dy / ch) * 0.5
    angle_cost = torch.sin(angle) ** 2
    w_cost = ((w1 - w2) / cw) ** 2
    h_cost = ((h1 - h2) / ch) ** 2
    shape_cost = (w_cost + h_cost) * 0.5

    penalty = dist_cost + 0.5 * angle_cost + shape_cost
    siou = iou - penalty
    return siou.clamp(-1.0, 1.0)


def iou_wiou(box1, box2, beta=2.0, eps=1e-7):
    """Weighted-IoU similarity approximation (Simple version)."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    iou = iou.clamp(0.0, 1.0)
    weight = 1.0 - torch.exp(-float(beta) * (iou ** 2))
    wiou = iou * weight
    return wiou.clamp(0.0, 1.0)


def iou_gwd(box1, box2, eps=1e-7):
    """Gaussian Wasserstein Distance similarity."""
    cx1 = (box1[:, 0] + box1[:, 2]) / 2.0
    cy1 = (box1[:, 1] + box1[:, 3]) / 2.0
    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    cx2 = (box2[:, 0] + box2[:, 2]) / 2.0
    cy2 = (box2[:, 1] + box2[:, 3]) / 2.0
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    cw = (c_x2 - c_x1).clamp(min=eps)
    ch = (c_y2 - c_y1).clamp(min=eps)
    
    dist_center_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    dist_shape_sq = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
    w_dist_sq = dist_center_sq + dist_shape_sq

    c_diag_sq = cw ** 2 + ch ** 2
    norm_w_dist_sq = w_dist_sq / c_diag_sq.clamp(min=eps)
    
    gwd_sim = 1.0 / (1.0 + norm_w_dist_sq)
    return gwd_sim.clamp(0.0, 1.0)


def iou_wiou_v1(box1, box2, eps=1e-7):
    """Wise-IoU v1 (static distance-based weight) similarity."""
    iou, _, _ = _box_iou_basic(box1, box2, eps)
    iou = iou.clamp(0.0, 1.0)

    c_x1 = torch.min(box1[:, 0], box2[:, 0])
    c_y1 = torch.min(box1[:, 1], box2[:, 1])
    c_x2 = torch.max(box1[:, 2], box2[:, 2])
    c_y2 = torch.max(box1[:, 3], box2[:, 3])
    cw = (c_x2 - c_x1).clamp(min=eps)
    ch = (c_y2 - c_y1).clamp(min=eps)
    c_diag_sq = cw ** 2 + ch ** 2 + eps
    
    cx1 = (box1[:, 0] + box1[:, 2]) / 2.0
    cy1 = (box1[:, 1] + box1[:, 3]) / 2.0
    cx2 = (box2[:, 0] + box2[:, 2]) / 2.0
    cy2 = (box2[:, 1] + box2[:, 3]) / 2.0
    
    dist_center_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    r_wiou = torch.exp(dist_center_sq / c_diag_sq).detach()
    loss_wiou = r_wiou * (1.0 - iou)
    sim_wiou = 1.0 - loss_wiou
    return sim_wiou.clamp(-1.0, 1.0)


def get_iou_variant(name: str):
    raw = (name or '').strip()
    lname = raw.lower()

    alpha = None
    gamma = None
    beta = None

    def parse_param(default_val):
        if '@' in raw:
            try:
                return float(raw.split('@', 1)[1])
            except Exception:
                return default_val
        return default_val

    if lname.startswith('alphaiou'):
        alpha = parse_param(2.0)
    if lname.startswith('focaliou'):
        gamma = parse_param(2.0)
    if lname.startswith('wiou'): # This handles 'WIoU@2.0'
        beta = parse_param(2.0)

    if lname == 'gwd':
        return iou_gwd
    if lname == 'wiouv1':
        return iou_wiou_v1
    if lname == 'iou':
        return iou_plain
    if lname == 'giou':
        return iou_giou
    if lname == 'diou':
        return iou_diou
    if lname == 'ciou':
        return iou_ciou
    if lname == 'eiou':
        return iou_eiou
    if lname == 'siou':
        return iou_siou
    if lname.startswith('wiou'): # Handles 'WIoU@2.0'
        def _wiou_wrap(b1, b2, eps=1e-7, _b=beta if beta is not None else 2.0):
            return iou_wiou(b1, b2, beta=_b, eps=eps)
        return _wiou_wrap
    if lname.startswith('alphaiou'):
        def _alpha_wrap(b1, b2, eps=1e-7, _a=alpha if alpha is not None else 2.0):
            return iou_alpha(b1, b2, alpha=_a, eps=eps)
        return _alpha_wrap
    if lname.startswith('focaliou'):
        def _focal_wrap(b1, b2, eps=1e-7, _g=gamma if gamma is not None else 2.0):
            return iou_focal(b1, b2, gamma=_g, eps=eps)
        return _focal_wrap

    return None


# ------------------------------------------------------------------
# --- FIXED PATCHING FUNCTION ---
# ------------------------------------------------------------------
def patch_bbox_loss(loss_type: str) -> tuple[str, bool]:
    """
    Patch Ultralytics BboxLoss.forward to replace the box loss
    with the selected IoU variant.
    
    This is a critical fix. The previous version was buggy.
    """
    
    # Native baseline: no patching
    if (loss_type or '').strip().lower() in {'native', 'baseline', 'ultralytics'}:
        return 'native', False

    variant_fn = get_iou_variant(loss_type)
    used_variant = (loss_type or '').lower()
    if variant_fn is None:
        print(f"Warning: Could not find IoU variant '{loss_type}'. Skipping patch.")
        return used_variant or 'unknown', False

    try:
        from ultralytics.yolo.utils.loss import BboxLoss
    except Exception:
        try:
            from ultralytics.utils.loss import BboxLoss  # legacy path
        except Exception:
            print("Error: Could not import BboxLoss from ultralytics.")
            return used_variant, False

    # Save the original function one time
    if not hasattr(BboxLoss, 'original_forward'):
        BboxLoss.original_forward = BboxLoss.forward

    def new_forward(self, pred_bboxes, pred_dist, anchor_points, target_bboxes,
                    target_scores, target_scores_sum, fg_mask):
        """
        Patched forward function.
        Replaces the box loss calculation but re-uses the original DFL
        loss calculation to ensure stability.
        """
        
        # --- 1. Guard for empty masks (same as original) ---
        if fg_mask is None or (isinstance(fg_mask, torch.Tensor) and fg_mask.numel() > 0 and fg_mask.sum() == 0):
            # Call the original function
            return BboxLoss.original_forward(self, pred_bboxes, pred_dist, anchor_points,
                                            target_bboxes, target_scores, target_scores_sum, fg_mask)

        # --- 2. Calculate DFL Loss (using original function) ---
        # We MUST do this, as re-implementing DFL is complex.
        # We call the original_forward to get both native losses.
        # The DFL loss (result[1]) will be kept.
        # The native box loss (result[0]) will be used for scaling.
        try:
            result_native = BboxLoss.original_forward(
                self, pred_bboxes, pred_dist, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            native_box_loss = result_native[0].detach()
            native_dfl_loss = result_native[1] # Keep this!
        except Exception as e:
            print(f"Error during original_forward call: {e}")
            # Fallback
            return BboxLoss.original_forward(self, pred_bboxes, pred_dist, anchor_points,
                                            target_bboxes, target_scores, target_scores_sum, fg_mask)

        # --- 3. Calculate CUSTOM Box Loss (our logic) ---
        try:
            b_pred = pred_bboxes[fg_mask]
            b_tgt = target_bboxes[fg_mask]

            def to_xyxy(b):
                if ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).all():
                    return b
                cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2].clamp(min=0), b[:, 3].clamp(min=0)
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                return torch.stack([x1, y1, x2, y2], dim=1)

            b_pred_xyxy = to_xyxy(b_pred)
            b_tgt_xyxy = to_xyxy(b_tgt)

            sim = variant_fn(b_pred_xyxy, b_tgt_xyxy)
            sim = sim.clamp(-1.0, 1.0)
            ts_fg = target_scores[fg_mask]
            denom = ts_fg.sum().clamp_min(1e-7)
            
            # Raw custom box loss (unweighted)
            box_loss_custom = ((1.0 - sim) * ts_fg).sum() / denom

            # --- 4. Rescale custom loss (IMPORTANT) ---
            # We rescale our custom loss to match the magnitude of the
            # native box loss. This prevents the optimizer from
            # exploding or vanishing.
            scale_match = (native_box_loss / (box_loss_custom.detach() + 1e-12)).clamp(min=0.1, max=10.0)
            box_loss = box_loss_custom * scale_match

        except Exception as e:
            # If our custom logic fails, fallback to native
            print(f"Error in custom loss calculation ({loss_type}): {e}")
            return (native_box_loss, native_dfl_loss)

        # --- 5. Return the correct tuple ---
        # (Custom_Box_Loss, Native_DFL_Loss)
        # The main trainer will apply the HYPERPARAMS weights
        # (e.g., 7.5 * box_loss, 1.5 * dfl_loss)
        return (box_loss, native_dfl_loss)

    # Apply the patch
    BboxLoss.forward = new_forward
    return used_variant, True
# ------------------------------------------------------------------
# --- END OF FIX ---
# ------------------------------------------------------------------


def _write_results_row(csv_path: str, row: dict, field_order: list[str]):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if not exists:
            w.writeheader()
        w.writerow(row)

print("="*80)
print("YOLOv8l TRAINING - Different Box Loss Functions (FIXED)")
print("="*80)
print(f"Dataset: {DATA_YAML}")
print(f"Base Model: {BASE_MODEL}")
print(f"Epochs: {EPOCHS}")
print(f"Output: {OUTPUT_DIR}")
print(f"\nComparing {len(SUPPORTED_LOSS_TYPES)} Loss Functions:")
for i, loss_type in enumerate(SUPPORTED_LOSS_TYPES, 1):
    print(f"  {i}. {loss_type}")

training_results = {}

# --- This loop is UNCHANGED ---
for loss_type in SUPPORTED_LOSS_TYPES:
    print(f"\n{'='*80}")
    print(f"Training with {loss_type.upper()} Loss")
    print(f"{'='*80}")

    try:
        # Global seed for this run
        run_seed = GLOBAL_SEED
        set_global_seed(run_seed)
        print(f"   → Seed: {run_seed}")

        # Load a fresh model from the checkpoint each time
        model = YOLO(BASE_MODEL)
        
        # --- Patch BboxLoss.forward to use selected IoU variant ---
        # This will now apply our FIXED patch
        used_variant, patched_ok = patch_bbox_loss(loss_type)
        if not patched_ok and used_variant != 'native':
            print(f"   ! Warning: '{loss_type}' not supported for patching. Skipping this variant.")
            training_results[loss_type] = {'status': 'skipped', 'error': 'unsupported-variant'}
            continue
        elif patched_ok:
             print(f"   → Successfully patched BboxLoss with {loss_type.upper()}")
        else:
             print(f"   → Running with 'Native' (unpatched) loss.")

        print(f"   → Starting training...")
        t0 = time.time()
        run_name = _slugify(f'yolov8l_{loss_type.lower()}_loss')
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=OUTPUT_DIR,
            name=run_name,
            exist_ok=True,
            seed=run_seed,
            plots=True,
            patience=20,
            **HYPERPARAMS
        )
        train_time_s = time.time() - t0

        print(f"   ✓ Training completed")

        # Save the final best model
        best_path = results.save_dir / 'weights' / 'best.pt'
        last_path = results.save_dir / 'weights' / 'last.pt'
        output_path = os.path.join(OUTPUT_DIR, _slugify(f'yolov8l_{loss_type}.pt'))

        src_weights = None
        if os.path.exists(best_path):
            src_weights = str(best_path)
        elif os.path.exists(last_path):
            src_weights = str(last_path)

        if src_weights is not None and os.path.exists(src_weights):
            shutil.copy(src_weights, output_path)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✓ Saved: {output_path} ({file_size:.2f} MB)")

            # Evaluate on test set and log metrics
            print("   → Running evaluation on test set...")
            t1 = time.time()
            val_res = model.val(data=DATA_YAML, split='test', imgsz=IMG_SIZE, batch=BATCH_SIZE, verbose=False)
            val_time_s = time.time() - t1

            metrics_dict = {}
            try:
                md = getattr(val_res, 'results_dict', None) or getattr(val_res, 'metrics', None)
                if isinstance(md, dict):
                    metrics_dict = md
                elif hasattr(md, 'results_dict'):
                    metrics_dict = md.results_dict
            except Exception:
                metrics_dict = {}

            mAP50 = metrics_dict.get('metrics/mAP50(B)', metrics_dict.get('map50', None))
            mAP5095 = metrics_dict.get('metrics/mAP50-95(B)', metrics_dict.get('map', None))
            precision = metrics_dict.get('metrics/precision(B)', metrics_dict.get('precision', None))
            recall = metrics_dict.get('metrics/recall(B)', metrics_dict.get('recall', None))

            row = {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'loss_type': loss_type,
                'used_variant': used_variant or 'default',
                'seed': run_seed,
                'epochs': EPOCHS,
                'imgsz': IMG_SIZE,
                'batch': BATCH_SIZE,
                'weights_path': output_path,
                'size_mb': f"{file_size:.2f}",
                'train_time_s': round(train_time_s, 2),
                'val_time_s': round(val_time_s, 2),
                'precision': precision,
                'recall': recall,
                'mAP50': mAP50,
                'mAP50-95': mAP5095,
            }
            _write_results_row(
                SUMMARY_CSV,
                row,
                ['timestamp', 'loss_type', 'used_variant', 'seed', 'epochs', 'imgsz', 'batch', 'weights_path', 'size_mb', 'train_time_s', 'val_time_s', 'precision', 'recall', 'mAP50', 'mAP50-95']
            )

            training_results[loss_type] = {'status': 'success', 'path': output_path, 'size_mb': file_size}
        else:
            print(f"   ✗ Model weights not found at: {best_path} or {last_path}")
            training_results[loss_type] = {'status': 'failed', 'error': 'Model file not found'}

    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        training_results[loss_type] = {'status': 'failed', 'error': str(e)}

# --- Summary ---
print(f"\n{'='*80}")
print("TRAINING SUMMARY")
print(f"{'='*80}")
successful = sum(1 for r in training_results.values() if r.get('status') == 'success')
print(f"\nCompleted: {successful}/{len(SUPPORTED_LOSS_TYPES)} models")
print("\nModels:")
for loss_type, result in training_results.items():
    if result.get('status') == 'success':
        print(f"   ✓ {loss_type.upper():12s} - {result.get('size_mb', 0):.2f} MB")
    else:
        print(f"   ✗ {loss_type.upper():12s} - Failed")

print(f"\n{'='*80}")