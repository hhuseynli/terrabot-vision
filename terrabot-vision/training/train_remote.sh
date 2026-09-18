#!/bin/bash
# TerraBot Vision - Remote Training Script
# Run this on a Vast.ai instance with RTX 4090/4080/3090/3080
# Usage: bash train_remote.sh

set -e

echo "=== TerraBot Vision Training ==="
echo "Step 1: Installing dependencies..."
pip install roboflow ultralytics scikit-learn albumentations pyyaml

echo ""
echo "Step 2: Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available! Check GPU compatibility.'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('GPU OK')
"

echo ""
echo "Step 3: Downloading datasets..."
python3 << 'PYEOF'
from roboflow import Roboflow
import threading, os

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

downloads = [
    ("joseph-nelson", "plantdoc", 4, "plantdoc"),
    ("graduation-project-2023", "plants-diseases-detection-and-classification", 12, "plants_disease"),
    ("advanced-ai-4rgba", "plant-disease-detection-3anip-wf15n", 1, "advanced_ai"),
    ("freelance-0dspz", "yolov8-g25jg", 1, "plantdoc_yolov8"),
    ("plant-disease-detection-using-yolo", "plant-disease-hcbov", 1, "plant_disease_yolo"),
]

def download(ws, proj, ver, folder):
    try:
        project = rf.workspace(ws).project(proj)
        version = project.version(ver)
        version.download("yolov8", location=f"/root/datasets/{folder}")
        print(f"DONE: {folder}")
    except Exception as e:
        print(f"FAILED {folder}: {e}")

threads = []
for ws, proj, ver, folder in downloads:
    t = threading.Thread(target=download, args=(ws, proj, ver, folder))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
print("All datasets downloaded.")
PYEOF

echo ""
echo "Step 4: Merging datasets..."
python3 << 'PYEOF'
import os, hashlib, shutil, yaml, random
from pathlib import Path
from collections import Counter

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

def build_remap(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    remap = {}
    for old_id, name in enumerate(names):
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

MERGED = Path("/root/merged_dataset")
(MERGED / "images").mkdir(parents=True, exist_ok=True)
(MERGED / "labels").mkdir(parents=True, exist_ok=True)

datasets = {
    "plantdoc": "/root/datasets/plantdoc",
    "plants_disease": "/root/datasets/plants_disease",
    "advanced_ai": "/root/datasets/advanced_ai",
    "plantdoc_yolov8": "/root/datasets/plantdoc_yolov8",
    "plant_disease_yolo": "/root/datasets/plant_disease_yolo",
}

total = 0
for ds_name, ds_path in datasets.items():
    yaml_path = os.path.join(ds_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"SKIP {ds_name}: no data.yaml")
        continue
    remap = build_remap(yaml_path)
    count = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(ds_path, split, "images")
        lbl_dir = os.path.join(ds_path, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = Path(img_file).stem
            ext = Path(img_file).suffix
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(src_lbl):
                continue
            h = hashlib.md5(f"{ds_name}_{img_file}".encode()).hexdigest()[:12]
            dst_img = MERGED / "images" / f"{ds_name}_{h}{ext}"
            dst_lbl = MERGED / "labels" / f"{ds_name}_{h}.txt"
            if remap_labels(src_lbl, dst_lbl, remap):
                shutil.copy2(os.path.join(img_dir, img_file), dst_img)
                count += 1
    print(f"{ds_name}: {count} images")
    total += count

print(f"\nTotal: {total} images")

# Split 90/10
random.seed(42)
all_imgs = list((MERGED / "images").glob("*"))
random.shuffle(all_imgs)
split_idx = int(len(all_imgs) * 0.9)

for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    (MERGED / sub).mkdir(parents=True, exist_ok=True)

for img in all_imgs[:split_idx]:
    lbl = MERGED / "labels" / (img.stem + ".txt")
    shutil.move(str(img), MERGED / "train/images" / img.name)
    if lbl.exists():
        shutil.move(str(lbl), MERGED / "train/labels" / lbl.name)

for img in all_imgs[split_idx:]:
    lbl = MERGED / "labels" / (img.stem + ".txt")
    shutil.move(str(img), MERGED / "val/images" / img.name)
    if lbl.exists():
        shutil.move(str(lbl), MERGED / "val/labels" / lbl.name)

shutil.rmtree(MERGED / "images", ignore_errors=True)
shutil.rmtree(MERGED / "labels", ignore_errors=True)

train_count = len(list((MERGED / "train/images").glob("*")))
val_count = len(list((MERGED / "val/images").glob("*")))
print(f"Train: {train_count}, Val: {val_count}")

data = {
    "path": str(MERGED),
    "train": "train/images",
    "val": "val/images",
    "nc": 29,
    "names": UNIFIED_NAMES,
}
with open(MERGED / "data.yaml", "w") as f:
    yaml.dump(data, f)

print("Merge complete.")
PYEOF

echo ""
echo "Step 5: Starting training..."
python3 << 'PYEOF'
import cv2
import numpy as np
import albumentations as A
from ultralytics import YOLO

# ESP32-CAM OV2640 augmentation - simulates real camera conditions
esp32_augment = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 9)),
        A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3)),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(std_range=(0.03, 0.12)),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
    ], p=0.35),
    A.ImageCompression(quality_range=(20, 65), p=0.4),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.3, 0.3)),
        A.RandomToneCurve(scale=0.2),
        A.CLAHE(clip_limit=(1, 4)),
    ], p=0.5),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
    ], p=0.4),
    A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=0.15),
    A.Downscale(scale_range=(0.4, 0.8), p=0.25),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.03, 0.12),
        hole_width_range=(0.03, 0.12),
        fill=0, p=0.2,
    ),
])

original_getitem = None

def patch_dataset(trainer):
    global original_getitem
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
    print("[CUSTOM] ESP32-CAM OV2640 augmentation patched")

model = YOLO("yolov8n.pt")
model.add_callback("on_train_start", patch_dataset)

model.train(
    data="/root/merged_dataset/data.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    patience=30,
    device=0,
    project="/root/runs",
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
print("TRAINING COMPLETE")
print(f"Best model saved to: /root/runs/terrabot_final/weights/best.pt")
print(f"Copy it to your local machine:")
print(f"  scp -P <port> root@<host>:/root/runs/terrabot_final/weights/best.pt ./best.pt")
PYEOF

echo ""
echo "=== Training complete ==="
echo "Best model: /root/runs/terrabot_final/weights/best.pt"
echo "Download with: scp -P <port> root@<host>:/root/runs/terrabot_final/weights/best.pt ./best.pt"
