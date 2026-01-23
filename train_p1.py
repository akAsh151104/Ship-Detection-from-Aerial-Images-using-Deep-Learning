#!/usr/bin/env python3
"""
mega_tiny_detector.py
Fully automated pipeline for tiny object detection:
  1) optional upsample
  2) tile dataset (images + crop labels)
  3) merge original images+labels into tiled dataset (KEEP ORIGINALS)
  4) inject tiny anchors to dataset YAML
  5) create P1+P2 model YAML
  6) train with Ultralytics YOLO
  7) run SAHI sliced inference on test images (folder) and save visuals & raw detections

Edit TOP CONFIG section below to match your environment.
"""
import os
import sys
import shutil
import time
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm

# ML libs (may be heavy)
import cv2
import torch
from ultralytics import YOLO

# SAHI
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# ------------------ TOP CONFIG ------------------
# dataset original (images + labels in YOLO format)
SRC = Path("/home/amit/DEEP_LEARNING_PROJECT/custom_dataset")  # original dataset root
# expect: SRC/images/*.jpg and SRC/labels/*.txt and dataset yaml at DATA_YAML

DATA_YAML = "/home/amit/DEEP_LEARNING_PROJECT/custom_dataset/data.yaml"  # original dataset yaml
WORK = Path("/home/amit/DEEP_LEARNING_PROJECT/mega_pipeline2")  # pipeline workspace
OUT_MODEL_DIR = WORK / "models2"
TILED_DIR = WORK / "tiled_dataset2"
UPS_DIR = WORK / "upscaled_dataset2"   # upsampled images (if using)
SAHI_OUT = WORK / "sahi_out"
TEST_IMGS = Path("/home/amit/DEEP_LEARNING_PROJECT/test_images2")  # folder of images to run SAHI on

# training params
# BASE_MODEL = "yolov8s-p2.pt"   # will fallback if not found locally
BASE_MODEL = "yolo11n.pt"   # will fallback if not found locally

IMG_SZ = 640
BATCH = 1
EPOCHS = 300
SEED = 42
WORKERS = 6
DEVICE = 7   # GPU device number or "cpu"

# tiling params
TILE = 512
OL = 0.2      # overlap fraction
MIN_A = 8     # min area px to keep bounding box inside tile

# upsample
UPSCALE = True   # set False to skip upscaling
SCALE = 2        # 2x upsample (bicubic). If False, skip.

# SAHI inference params
SLICE_SIZE = 512
OVERLAP = 0.2
CONF = 0.001
IOU = 0.4

# hyperparameters (tiny-object focused)
HYP = {
    "lr0": 0.002,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "box": 8.0,
    "cls": 0.3,
    "dfl": 1.5,
    "degrees": 6.0,
    "translate": 0.06,
    "scale": 0.5,
    "perspective": 0.0005,
    "fliplr": 0.5,
    "mosaic": 0.6,
    "mixup": 0.0,
    "copy_paste": 0.,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "close_mosaic": 25,
}
# ------------------------------------------------

def makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def upsample_images(src_root: Path, dst_root: Path, scale: int = 2):
    """Bicubic upsample images, copy labels (normalized coords unchanged)."""
    makedirs(dst_root)
    # Process train, val, test splits
    for split in ['train', 'val', 'test']:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        imgs_in = split_dir / "images"
        lbls_in = split_dir / "labels"
        if not imgs_in.exists():
            continue
        imgs_out = dst_root / split / "images"
        lbls_out = dst_root / split / "labels"
        makedirs(imgs_out); makedirs(lbls_out)
        for p in tqdm(sorted(imgs_in.glob("*.*")), desc=f"upsampling {split}"):
            im = cv2.imread(str(p))
            if im is None:
                continue
            h,w = im.shape[:2]
            nw, nh = int(w*scale), int(h*scale)
            rsz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(imgs_out / p.name), rsz)
            lbl = lbls_in / (p.stem + ".txt")
            if lbl.exists():
                with open(lbl, "r") as f:
                    data = f.read()
                with open(lbls_out / lbl.name, "w") as f2:
                    f2.write(data)
    print("✓ upsampling done")

def tile_dataset(src_root: Path, out_root: Path, tile_size: int=512, overlap: float=0.2, min_area:int=8):
    """
    Create overlapping tiles and corresponding YOLO labels for each tile.
    src_root expected to have train/val/test splits with 'images' and 'labels' subfolders.
    """
    # Process train, val, test splits
    for split in ['train', 'val', 'test']:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        imgs_in = split_dir / "images"
        lbls_in = split_dir / "labels"
        if not imgs_in.exists():
            continue
        imgs_out = out_root / split / "images"
        lbls_out = out_root / split / "labels"
        makedirs(imgs_out); makedirs(lbls_out)

        def load_yolo(lbl_p):
            boxes=[]
            if not lbl_p.exists(): return boxes
            with open(lbl_p,"r") as f:
                for L in f:
                    sp=L.strip().split()
                    if len(sp)<5: continue
                    cls=int(sp[0]); cx,cy,w,h = map(float,sp[1:5])
                    boxes.append((cls,cx,cy,w,h))
            return boxes

        def yolo_to_xyxy(b, iw, ih):
            cls,cx,cy,w,h = b
            x1=(cx - w/2)*iw
            y1=(cy - h/2)*ih
            x2=(cx + w/2)*iw
            y2=(cy + h/2)*ih
            return cls, int(x1), int(y1), int(x2), int(y2)

        def xyxy_to_yolo(x1,y1,x2,y2,tw,th):
            w = x2-x1; h = y2-y1
            cx = x1 + w/2; cy = y1 + h/2
            return cx/tw, cy/th, w/tw, h/th

        imgs = sorted(imgs_in.glob("*.*"))
        for p in tqdm(imgs, desc=f"tiling {split}"):
            im = cv2.imread(str(p))
            if im is None: continue
            ih, iw = im.shape[:2]
            lbl_p = lbls_in / (p.stem + ".txt")
            boxes = load_yolo(lbl_p)
            boxes_xy = [yolo_to_xyxy(b, iw, ih) for b in boxes]

            step = int(tile_size * (1 - overlap))
            xs = list(range(0, max(1, iw - tile_size + 1), step))
            ys = list(range(0, max(1, ih - tile_size + 1), step))
            if xs[-1] != max(0, iw - tile_size): xs.append(max(0, iw - tile_size))
            if ys[-1] != max(0, ih - tile_size): ys.append(max(0, ih - tile_size))

            for y in ys:
                for x in xs:
                    crop = im[y:y+tile_size, x:x+tile_size]
                    if crop.shape[0]==0 or crop.shape[1]==0: continue
                    out_name = f"{p.stem}_tx{x}_ty{y}.jpg"
                    out_img_p = imgs_out / out_name
                    out_lbl_p = lbls_out / (out_name.replace(".jpg", ".txt"))
                    cv2.imwrite(str(out_img_p), crop)
                    out_lines=[]
                    for cls,x1,y1,x2,y2 in boxes_xy:
                        ix1 = max(x1, x); iy1 = max(y1, y)
                        ix2 = min(x2, x+tile_size); iy2 = min(y2, y+tile_size)
                        if ix2 <= ix1 or iy2 <= iy1: continue
                        area = (ix2-ix1)*(iy2-iy1)
                        if area < min_area: continue
                        nx1, ny1 = ix1 - x, iy1 - y
                        nx2, ny2 = ix2 - x, iy2 - y
                        cx,cy,w_,h_ = xyxy_to_yolo(nx1,ny1,nx2,ny2,tile_size,tile_size)
                        out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w_:.6f} {h_:.6f}\n")
                    with open(out_lbl_p,"w") as f:
                        f.writelines(out_lines)
    print("✓ tiling done")

def merge_original_into_tiled(orig_root: Path, tiled_root: Path):
    """
    Copy original images and labels into the tiled dataset folder,
    so training includes both originals and tiles.
    """
    # Process train, val, test splits
    for split in ['train', 'val', 'test']:
        split_dir = orig_root / split
        if not split_dir.exists():
            continue
        orig_imgs = split_dir / "images"
        orig_lbls = split_dir / "labels"
        if not orig_imgs.exists():
            continue
        tiled_imgs = tiled_root / split / "images"
        tiled_lbls = tiled_root / split / "labels"
        makedirs(tiled_imgs); makedirs(tiled_lbls)
        # copy images
        for p in tqdm(sorted(orig_imgs.glob("*.*")), desc=f"merging {split} images"):
            dst = tiled_imgs / p.name
            if not dst.exists():
                try:
                    shutil.copy(p, dst)
                except Exception:
                    pass
        # copy labels
        for p in tqdm(sorted(orig_lbls.glob("*.txt")), desc=f"merging {split} labels"):
            dst = tiled_lbls / p.name
            if not dst.exists():
                try:
                    shutil.copy(p, dst)
                except Exception:
                    pass
    print("✓ Original images+labels merged into tiled dataset")

def inject_tiny_anchors(yaml_path: str) -> str:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    with open(p,"r") as f:
        data = yaml.safe_load(f)
    tiny_anchors = [
        [4,6,8,10,12,14],
        [18,22,28,32,40,44],
        [56,64,80,96,128,144],
    ]
    data2 = dict(data)
    data2.setdefault("anchors", tiny_anchors)
    new_yaml = str(p.with_name(p.stem + "_tiny_anchors.yaml"))
    with open(new_yaml,"w") as f:
        yaml.dump(data2, f, sort_keys=False)
    print("✓ tiny anchors injected ->", new_yaml)
    return new_yaml

def create_p1p2_cfg(out_dir: Path, nc=1) -> str:
    cfg = f"""
# P1+P2 style YOLO config (nc overwritten by trainer typically)
nc: {nc}
backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]     # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]     # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]     # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]    # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]       # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]      # cat backbone P4
  - [-1, 3, C2f, [512]]            # 12
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]      # cat backbone P3
  - [-1, 3, C2f, [256]]            # 15
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]      # cat backbone P2
  - [-1, 3, C2f, [128]]            # 18 (P2/4-small)w
  
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]     # cat head P3
  - [-1, 3, C2f, [256]]            # 21 (P3/8-medium)
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]     # cat head P4
  - [-1, 3, C2f, [512]]            # 24 (P4/16-large)
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]      # cat backbone P5
  - [-1, 3, C2f, [1024]]           # 27 (P5/32-xlarge)
  
  - [[18, 21, 24, 27], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""
    makedirs(out_dir)
    cfg_p = out_dir / "yolo_p1p2.yaml"
    with open(cfg_p, "w") as f:
        f.write(cfg)
    print("✓ created model cfg ->", cfg_p)
    return str(cfg_p)

def resolve_base_model(preferred: str) -> str:
    # prefer absolute or local paths; otherwise fallback to common checkpoint names
    workspace = [
        "/home/amit/DEEP_LEARNING_PROJECT/yolo11n.pt",
        "/home/amit/DEEP_LEARNING_PROJECT/yolo11l.pt",
        "/home/amit/DEEP_LEARNING_PROJECT/yolov8n.pt",
        "/home/amit/DEEP_LEARNING_PROJECT/yolov8l.pt",
        "/home/amit/DEEP_LEARNING_PROJECT/yolov8x.pt",
    ]
    if os.path.isabs(preferred) and os.path.exists(preferred):
        return preferred
    if os.path.exists(preferred):
        return preferred
    for p in workspace:
        if os.path.exists(p):
            print(f"⚠ fallback -> {p}")
            return p
    print("⚠ preferred base not found, using 'yolov8n.pt' (ultralytics will fetch if needed)")
    return "yolov8n.pt"

def check_gpu():
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Training will run on CPU (very slow).")
    else:
        info = torch.cuda.get_device_properties(0)
        print(f"GPU detected: {info.name} — {info.total_memory/(1024**3):.1f} GB")
        torch.cuda.empty_cache()

def train_model(cfg_p: str, data_yaml: str, out_dir: Path):
    check_gpu()
    base = resolve_base_model(BASE_MODEL)
    print("Loading model config and base weights:", cfg_p, " <- ", base)
    model = YOLO(cfg_p).load(base)
    print("→ training: imgsz", IMG_SZ, "batch", BATCH, "epochs", EPOCHS)
    t0 = time.time()
    res = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SZ,
        batch=BATCH,
        project=str(out_dir),
        name="p1p2_1280",
        exist_ok=True,
        seed=SEED,
        workers=WORKERS,
        plots=True,
        patience=80,
        amp=False,
        multi_scale=True,
        save=True,
        device=DEVICE,
        **HYP
    )
    print("train done (hrs):", (time.time()-t0)/3600)
    # copy best or last
    try:
        sd = getattr(res, "save_dir", None)
        if sd is None:
            sd = Path(out_dir) / "p1p2_1280"
        else:
            sd = Path(sd)
        best = sd / "weights" / "best.pt"
        last = sd / "weights" / "last.pt"
        dst = Path(out_dir) / "p1p2_1280_best.pt"
        if best.exists():
            shutil.copy(best, dst); print("✓ Best model saved ->", dst)
        elif last.exists():
            shutil.copy(last, dst); print("⚠ best missing, copied last ->", dst)
        else:
            print("✗ No checkpoint found in:", sd)
        return str(dst) if dst.exists() else None
    except Exception as e:
        print("warn saving model:", e)
        return None

def sahi_infer(weights: str, img_folder: Path, outp: Path, slice_size=512, overlap=0.2, conf=0.001):
    makedirs(outp)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = UltralyticsDetectionModel(model_path=weights, confidence_threshold=conf, device=dev)
    imgs = sorted(img_folder.glob("*.*"))
    for im_p in tqdm(imgs, desc="sahi inference"):
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
            print("sahi error for", im_p, e); continue
        im = cv2.imread(str(im_p))
        vis = im.copy()
        for ann in res.object_prediction_list:
            cls = ann.category.id
            sc = ann.score.value
            x1,y1,x2,y2 = map(int, [ann.bbox.min_x, ann.bbox.min_y, ann.bbox.max_x, ann.bbox.max_y])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{cls}:{sc:.3f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        out_vis = outp / (im_p.stem + "_sahi_vis.jpg")
        cv2.imwrite(str(out_vis), vis)
        # save raw detections
        detf = outp / (im_p.stem + "_sahi.txt")
        with open(detf,"w") as f:
            for ann in res.object_prediction_list:
                cls = ann.category.id
                sc = ann.score.value
                x1,y1,x2,y2 = ann.bbox.min_x, ann.bbox.min_y, ann.bbox.max_x, ann.bbox.max_y
                f.write(f"{cls} {sc:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    print("✓ SAHI inference done ->", outp)

def update_dataset_yaml_for_tiled(orig_yaml: str, tiled_root: Path, out_yaml: str):
    """
    Create a dataset YAML pointing to tiled dataset for training.
    It copies most keys from original (nc, names) but replaces train/val/test paths to tiled dirs.
    """
    p = Path(orig_yaml)
    if not p.exists():
        raise FileNotFoundError("orig yaml missing: " + orig_yaml)
    with open(p,"r") as f: d = yaml.safe_load(f)
    d2 = dict(d)
    # Replace train/val/test with tiled dataset paths
    d2['train'] = str(tiled_root / 'train' / 'images')
    d2['val'] = str(tiled_root / 'val' / 'images')
    if (tiled_root / 'test' / 'images').exists():
        d2['test'] = str(tiled_root / 'test' / 'images')
    outp = Path(out_yaml)
    with open(outp, "w") as f:
        yaml.dump(d2, f, sort_keys=False)
    print("✓ created tiled dataset yaml ->", outp)
    return str(outp)

# ------------------ MAIN PIPELINE ------------------
def run_all():
    makedirs(WORK); makedirs(OUT_MODEL_DIR); makedirs(TILED_DIR); makedirs(SAHI_OUT)
    # step 0: optionally upsample
    ds_src = SRC
    if UPSCALE:
        print("→ Upscaling images (bicubic) ...")
        upsample_images(SRC, UPS_DIR, scale=SCALE)
        ds_src = UPS_DIR
    # step 1: tile dataset
    print("→ Tiling dataset ...")
    tile_dataset(ds_src, TILED_DIR, tile_size=TILE, overlap=OL, min_area=MIN_A)
    # step 1b: merge original images + labels into tiled dataset
    print("→ Merging original images+labels into tiled dataset ...")
    # Use the original root if not upscaled, otherwise merge from the upscaled originals
    merge_src = ds_src if ds_src.exists() else SRC
    merge_original_into_tiled(merge_src, TILED_DIR)
    # step 2: create dataset yaml for tiled dataset
    tiled_yaml = WORK / "data_tiled.yaml"
    tiled_yaml = update_dataset_yaml_for_tiled(DATA_YAML, TILED_DIR, tiled_yaml)
    # step 3: inject anchors
    print("→ Injecting tiny anchors ...")
    tiny_yaml = inject_tiny_anchors(tiled_yaml)
    # step 4: create model cfg
    print("→ Creating P1+P2 model config ...")
    cfg_p = create_p1p2_cfg(OUT_MODEL_DIR, nc=1)
    # step 5: train
    print("→ Starting training ... (this may take long)")
    best = train_model(cfg_p, tiny_yaml, OUT_MODEL_DIR)
    if best is None:
        print("⚠ Training did not produce a saved model. Exiting inference.")
        return
    # step 6: SAHI inference on TEST_IMGS
    if TEST_IMGS.exists() and any(TEST_IMGS.glob("*.*")):
        print("→ Running SAHI inference on test images ...")
        sahi_infer(best, TEST_IMGS, SAHI_OUT, slice_size=SLICE_SIZE, overlap=OVERLAP, conf=CONF)
    else:
        print("⚠ Test images folder missing or empty:", TEST_IMGS)
    print("→ Pipeline finished. Results in:", WORK)

# allow command-line control if wanted
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mega tiny detector pipeline")
    parser.add_argument("--no-up", action="store_true", help="skip upsampling")
    parser.add_argument("--only-tiles", action="store_true", help="only run tiling")
    parser.add_argument("--only-train", action="store_true", help="assume tiled dataset exists and only train")
    parser.add_argument("--only-infer", action="store_true", help="only run SAHI inference (requires model path)")
    parser.add_argument("--model", type=str, default=None, help="path to model for --only-infer")
    args = parser.parse_args()

    # apply flags
    if args.no_up:
        UPSCALE = False
    if args.only_tiles:
        makedirs(WORK); makedirs(TILED_DIR)
        ds_src = SRC if not UPSCALE else UPS_DIR
        if UPSCALE and not UPS_DIR.exists():
            upsample_images(SRC, UPS_DIR, SCALE)
            ds_src = UPS_DIR
        tile_dataset(ds_src, TILED_DIR, tile_size=TILE, overlap=OL, min_area=MIN_A)
        # merge originals as well
        merge_original_into_tiled(ds_src, TILED_DIR)
        sys.exit(0)
    if args.only_train:
        makedirs(OUT_MODEL_DIR)
        tiled_yaml = WORK / "data_tiled.yaml"
        if not tiled_yaml.exists():
            tiled_yaml = update_dataset_yaml_for_tiled(DATA_YAML, TILED_DIR, WORK / "data_tiled.yaml")
        tiny_yaml = inject_tiny_anchors(str(tiled_yaml))
        cfg_p = create_p1p2_cfg(OUT_MODEL_DIR, nc=1)
        train_model(cfg_p, tiny_yaml, OUT_MODEL_DIR)
        sys.exit(0)
    if args.only_infer:
        if args.model is None:
            print("provide --model <weights.pt>") ; sys.exit(1)
        makedirs(SAHI_OUT)
        sahi_infer(args.model, TEST_IMGS, SAHI_OUT, slice_size=SLICE_SIZE, overlap=OVERLAP, conf=CONF)
        sys.exit(0)

    # default: run full pipeline
    run_all()
