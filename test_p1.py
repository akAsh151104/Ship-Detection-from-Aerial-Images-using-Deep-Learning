#!/usr/bin/env python3
"""
predict_sahi.py
Run SAHI sliced inference using trained YOLO model.
HARD-CODED PATH VERSION (no argparse).
"""

import os
import csv
import cv2
from pathlib import Path
from tqdm import tqdm
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
import torch


# ----------------- HARD CODED PATHS -----------------

MODEL_PATH = "/home/amit/DEEP_LEARNING_PROJECT/mega_pipeline/models/p1p2_1280/weights/best.pt"
# MODEL_PATH = "/home/amit/DEEP_LEARNING_PROJECT/yolo11_loss_comparison_all_Adv6/yolov8l_native_loss/weights/best.pt"
# MODEL_PATH = "yolo11n.pt"



# root test folder contains `images/` and `labels/`
TEST_ROOT = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/test"
TEST_IMAGES = os.path.join(TEST_ROOT, "images")

OUTPUT_DIR = "/home/amit/DEEP_LEARNING_PROJECT/mega_pipeline2/sahi_out_predict_6"

SLICE_SIZE = 512
OVERLAP = 0.2
CONF = 0.10

# best  at  0.24

# ----------------------------------------------------


def makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sahi_infer(weights: str, img_folder: Path, outp: Path,
               slice_size=512, overlap=0.2, conf=0.001):

    makedirs(outp)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    model = UltralyticsDetectionModel(
        model_path=weights,
        confidence_threshold=conf,
        device=dev
    )

    imgs = sorted(img_folder.glob("*.*"))
    print(f"Found {len(imgs)} test images")

    # metrics accumulators
    per_image_stats = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    labels_dir = Path(TEST_ROOT) / "labels"

    for im_p in tqdm(imgs, desc="SAHI Inference"):
        try:
            res = get_sliced_prediction(
                str(im_p),
                detection_model=model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                postprocess_type="NMS",
                verbose=False
            )
        except Exception as e:
            print("Error during SAHI:", im_p, e)
            continue

        im = cv2.imread(str(im_p))
        vis = im.copy()

        for ann in res.object_prediction_list:
            cls = ann.category.id
            sc = ann.score.value
            x1, y1, x2, y2 = map(int, [
                ann.bbox.minx, ann.bbox.miny,
                ann.bbox.maxx, ann.bbox.maxy
            ])

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            # cv2.putText(vis, f"{cls}:{sc:.3f}", (x1, max(0, y1-6)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # save visualization
        out_vis = outp / f"{im_p.stem}_vis.jpg"
        cv2.imwrite(str(out_vis), vis)

        # save raw detections
        raw_txt = outp / f"{im_p.stem}.txt"
        with open(raw_txt, "w") as f:
            for ann in res.object_prediction_list:
                cls = ann.category.id
                sc = ann.score.value
                x1, y1, x2, y2 = ann.bbox.minx, ann.bbox.miny, ann.bbox.maxx, ann.bbox.maxy
                f.write(f"{cls} {sc:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        # ---------- metric computation for this image ----------
        # load GT YOLO label from labels/ folder
        gt_path = labels_dir / f"{im_p.stem}.txt"
        gt_boxes = []  # (cls, x1,y1,x2,y2)
        if gt_path.exists():
            with open(gt_path, "r") as gf:
                for line in gf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    gcls = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    H, W = im.shape[:2]
                    gx1 = (cx - w / 2.0) * W
                    gy1 = (cy - h / 2.0) * H
                    gx2 = (cx + w / 2.0) * W
                    gy2 = (cy + h / 2.0) * H
                    gt_boxes.append((gcls, gx1, gy1, gx2, gy2))

        total_gt += len(gt_boxes)

        def iou(box1, box2):
            ax1, ay1, ax2, ay2 = box1
            bx1, by1, bx2, by2 = box2
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        img_tp = 0
        img_fp = 0
        used_gt = set()
        iou_thr = 0.5

        for ann in res.object_prediction_list:
            cls = ann.category.id
            x1, y1, x2, y2 = ann.bbox.minx, ann.bbox.miny, ann.bbox.maxx, ann.bbox.maxy

            best_iou = 0.0
            best_gt_idx = -1
            for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gi in used_gt:
                    continue
                if gcls != cls:
                    continue
                i = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if i > best_iou:
                    best_iou = i
                    best_gt_idx = gi

            if best_iou >= iou_thr and best_gt_idx >= 0:
                img_tp += 1
                used_gt.add(best_gt_idx)
            else:
                img_fp += 1

        total_tp += img_tp
        total_fp += img_fp
        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
        img_recall = img_tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0

        per_image_stats.append({
            "image": im_p.name,
            "tp": img_tp,
            "fp": img_fp,
            "gt": len(gt_boxes),
            "precision": img_prec,
            "recall": img_recall,
        })

    # overall metrics
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    # very simple accuracy: correct detections over all predictions + gt (not ideal for detection)
    overall_acc = total_tp / (total_tp + total_fp + (total_gt - total_tp)) if (total_tp + total_fp + (total_gt - total_tp)) > 0 else 0.0

    # write csv summary
    csv_path = outp / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as cf:
        fieldnames = ["image", "tp", "fp", "gt", "precision", "recall"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image_stats:
            writer.writerow(row)
        writer.writerow({
            "image": "OVERALL",
            "tp": total_tp,
            "fp": total_fp,
            "gt": total_gt,
            "precision": overall_prec,
            "recall": overall_recall,
        })

    print("✓ SAHI prediction completed!")
    print("Overall precision:", overall_prec)
    print("Overall recall:", overall_recall)
    print("Overall accuracy (approx):", overall_acc)
    print("CSV metrics saved to:", csv_path)
    print("Results saved to:", outp)


if __name__ == "__main__":

    model_path = Path(MODEL_PATH)
    img_folder = Path(TEST_IMAGES)
    out_folder = Path(OUTPUT_DIR)

    if not model_path.exists():
        print("Model not found:", model_path)
        exit()

    if not img_folder.exists():
        print("Image folder not found:", img_folder)
        exit()

    sahi_infer(str(model_path), img_folder, out_folder,
               slice_size=SLICE_SIZE, overlap=OVERLAP, conf=CONF)
