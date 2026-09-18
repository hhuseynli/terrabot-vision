"""
TerraBot Vision - Plant Disease Detection Training Script
=========================================================
Trains a YOLOv8n model on 5 plant disease datasets with ESP32-CAM
augmentation for real-world robustness.

Requirements:
    pip install roboflow ultralytics scikit-learn albumentations pyyaml

Usage:
    python train.py                  # auto-detects GPU or CPU
    python train.py --device cpu     # force CPU
    python train.py --device 0       # force GPU 0
    python train.py --device mps     # Mac Apple Silicon GPU
    python train.py --epochs 100     # custom epochs
    python train.py --batch 8        # smaller batch for low RAM
"""

import os
import sys
import hashlib
import shutil
import yaml
import random
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"
MERGED_DIR = SCRIPT_DIR / "merged_dataset"

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")

ROBOFLOW_DATASETS = [
    ("joseph-nelson", "plantdoc", 4, "plantdoc"),
    ("graduation-project-2023", "plants-diseases-detection-and-classification", 12, "plants_disease"),
    ("advanced-ai-4rgba", "plant-disease-detection-3anip-wf15n", 1, "advanced_ai"),
    ("freelance-0dspz", "yolov8-g25jg", 1, "plantdoc_yolov8"),
    ("plant-disease-detection-using-yolo", "plant-disease-hcbov", 1, "plant_disease_yolo"),
]

UNIFIED_NAMES = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf",
    "Bell_pepper leaf spot", "Bell_pepper leaf", "Blueberry leaf",
    "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight",
    "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
    "Potato leaf late blight", "Potato leaf", "Raspberry leaf",
    "Soyabean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
    "Tomato Early blight leaf", "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus",
    "Tomato leaf", "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot", "grape leaf",
]
NAME_TO_ID = {n: i for i, n in enumerate(UNIFIED_NAMES)}


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------
def download_datasets():
    print("\n=== Step 1: Downloading datasets ===")
    from roboflow import Roboflow
    import threading

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    def download(ws, proj, ver, folder):
        path = DATASETS_DIR / folder
        if path.exists() and (path / "data.yaml").exists():
            print(f"  SKIP (exists): {folder}")
            return
        if path.exists():
            shutil.rmtree(path)
        try:
            project = rf.workspace(ws).project(proj)
            version = project.version(ver)
            version.download("yolov8", location=str(path))
            print(f"  DONE: {folder}")
        except Exception as e:
            print(f"  FAILED {folder}: {e}")

    threads = []
    for ws, proj, ver, folder in ROBOFLOW_DATASETS:
        t = threading.Thread(target=download, args=(ws, proj, ver, folder))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("  All datasets ready.")


# ---------------------------------------------------------------------------
# Step 2: Merge
# ---------------------------------------------------------------------------
def build_remap(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    remap = {}
    for old_id, name in enumerate(data.get("names", [])):
        if name in NAME_TO_ID:
            remap[old_id] = NAME_TO_ID[name]
        elif name == "Soybean leaf":
            remap[old_id] = NAME_TO_ID["Soyabean leaf"]
    return remap


def remap_labels(src_lbl, dst_lbl, remap):
    with open(src_lbl) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        old_cls = int(parts[0])
        if old_cls in remap:
            parts[0] = str(remap[old_cls])
            new_lines.append(" ".join(parts) + "\n")
    if new_lines:
        with open(dst_lbl, "w") as f:
            f.writelines(new_lines)
        return True
    return False


def merge_datasets():
    print("\n=== Step 2: Merging datasets ===")

    if (MERGED_DIR / "train" / "images").exists():
        n = len(list((MERGED_DIR / "train" / "images").glob("*")))
        print(f"  Merged dataset exists ({n} train images), skipping...")
        return

    (MERGED_DIR / "images").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    for _, _, _, ds_name in ROBOFLOW_DATASETS:
        ds_path = DATASETS_DIR / ds_name
        yaml_path = ds_path / "data.yaml"
        if not yaml_path.exists():
            print(f"  SKIP {ds_name}: no data.yaml")
            continue
        remap = build_remap(yaml_path)
        count = 0
        for split in ["train", "valid", "test"]:
            img_dir = ds_path / split / "images"
            lbl_dir = ds_path / split / "labels"
            if not img_dir.exists():
                continue
            for img_file in os.listdir(img_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = Path(img_file).stem
                ext = Path(img_file).suffix
                src_lbl = lbl_dir / (stem + ".txt")
                if not src_lbl.exists():
                    continue
                h = hashlib.md5(f"{ds_name}_{img_file}".encode()).hexdigest()[:12]
                dst_img = MERGED_DIR / "images" / f"{ds_name}_{h}{ext}"
                dst_lbl = MERGED_DIR / "labels" / f"{ds_name}_{h}.txt"
                if remap_labels(str(src_lbl), str(dst_lbl), remap):
                    shutil.copy2(str(img_dir / img_file), str(dst_img))
                    count += 1
        print(f"  {ds_name}: {count} images")
        total += count

    print(f"  Total: {total} images")

    # Split 90/10
    random.seed(42)
    all_imgs = list((MERGED_DIR / "images").glob("*"))
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * 0.9)

    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (MERGED_DIR / sub).mkdir(parents=True, exist_ok=True)

    for img in all_imgs[:split_idx]:
        lbl = MERGED_DIR / "labels" / (img.stem + ".txt")
        shutil.move(str(img), str(MERGED_DIR / "train" / "images" / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(MERGED_DIR / "train" / "labels" / lbl.name))

    for img in all_imgs[split_idx:]:
        lbl = MERGED_DIR / "labels" / (img.stem + ".txt")
        shutil.move(str(img), str(MERGED_DIR / "val" / "images" / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(MERGED_DIR / "val" / "labels" / lbl.name))

    shutil.rmtree(str(MERGED_DIR / "images"), ignore_errors=True)
    shutil.rmtree(str(MERGED_DIR / "labels"), ignore_errors=True)

    train_count = len(list((MERGED_DIR / "train" / "images").glob("*")))
    val_count = len(list((MERGED_DIR / "val" / "images").glob("*")))
    print(f"  Train: {train_count}, Val: {val_count}")

    data = {
        "path": str(MERGED_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": 29,
        "names": UNIFIED_NAMES,
    }
    with open(str(MERGED_DIR / "data.yaml"), "w") as f:
        yaml.dump(data, f)
    print("  Merge complete.")


# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
def get_esp32_augmentation():
    import albumentations as A
    return A.Compose([
        # Camera quality (OV2640 sensor)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 9)),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3)),
        ], p=0.4),
        # Sensor noise
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.12)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.35),
        # JPEG compression (ESP32 compresses at quality ~80)
        A.ImageCompression(quality_range=(20, 65), p=0.4),
        # Lighting
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.3, 0.3)),
            A.RandomToneCurve(scale=0.2),
            A.CLAHE(clip_limit=(1, 4)),
        ], p=0.5),
        # Color temperature
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
        ], p=0.4),
        # Shadows
        A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=0.15),
        # Resolution degradation
        A.Downscale(scale_range=(0.4, 0.8), p=0.25),
        # Partial occlusion (fingers, stems)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.03, 0.12),
            hole_width_range=(0.03, 0.12),
            fill=0, p=0.2,
        ),
    ])


def train(device, epochs, batch):
    print(f"\n=== Step 3: Training (device={device}, epochs={epochs}, batch={batch}) ===")
    from ultralytics import YOLO

    esp32_augment = get_esp32_augmentation()
    original_getitem = None

    def patch_dataset(trainer):
        nonlocal original_getitem
        dataset = trainer.train_loader.dataset
        if original_getitem is None:
            original_getitem = dataset.__class__.__getitem__

        def augmented_getitem(self, index):
            result = original_getitem(self, index)
            if "img" in result and isinstance(result["img"], np.ndarray):
                try:
                    img = result["img"]
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                        augmented = esp32_augment(image=img)["image"]
                        result["img"] = np.transpose(augmented, (2, 0, 1))
                    else:
                        result["img"] = esp32_augment(image=img)["image"]
                except Exception:
                    pass
            return result

        dataset.__class__.__getitem__ = augmented_getitem
        print("  [ESP32-CAM] OV2640 augmentation patched")

    model = YOLO("yolov8n.pt")
    model.add_callback("on_train_start", patch_dataset)

    results = model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        patience=30,
        device=device,
        project=str(SCRIPT_DIR / "runs"),
        name="terrabot_final",
        # Anti-overfitting
        freeze=10,
        dropout=0.15,
        weight_decay=0.001,
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        label_smoothing=0.05,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        erasing=0.3,
        close_mosaic=15,
        workers=4,
        verbose=True,
    )

    # Copy best model to server
    best_pt = SCRIPT_DIR / "runs" / "terrabot_final" / "weights" / "best.pt"
    server_pt = SCRIPT_DIR.parent / "server" / "best.pt"
    if best_pt.exists():
        server_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_pt), str(server_pt))
        print(f"\n  Best model copied to {server_pt}")

    print("\n=== TRAINING COMPLETE ===")
    print(f"  Best model: {best_pt}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def detect_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {name}")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple Silicon MPS detected")
        return "mps"
    else:
        print("  No GPU found, using CPU (this will be slow)")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="TerraBot Vision Training")
    parser.add_argument("--device", type=str, default=None, help="cpu, 0, mps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  TerraBot Vision - Plant Disease Detection Training")
    print("  5 datasets, 29 classes, ESP32-CAM OV2640 augmentation")
    print("=" * 60)

    # Detect device
    if args.device is None:
        args.device = detect_device()
    else:
        print(f"  Using device: {args.device}")

    # Auto batch size
    if args.batch is None:
        if args.device == "cpu":
            args.batch = 16
        elif args.device == "mps":
            args.batch = 16
        else:
            args.batch = 32

    download_datasets()
    merge_datasets()
    train(args.device, args.epochs, args.batch)


if __name__ == "__main__":
    main()
