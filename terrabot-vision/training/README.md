# Training a Custom Plant Disease Model

## 1. Collect Images

Capture 200+ images from the ESP32-CAM using the client's screenshot feature (press `s`).
Include varied lighting, angles, and both healthy and diseased plants.

## 2. Label with Roboflow

1. Create a project at [roboflow.com](https://roboflow.com)
2. Upload your images
3. Label with two classes: `plant` and `disease`
4. Export in **YOLOv8** format

## 3. Organize Dataset

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Label format (one `.txt` per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].

## 4. Update Config

Edit `plant_disease.yaml` and set the `path:` to your dataset's absolute path.

## 5. Train

```bash
yolo detect train \
  data=plant_disease.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=terrabot_training \
  name=plant_disease_run
```

## 6. Deploy

```bash
cp terrabot_training/plant_disease_run/weights/best.pt ../server/best.pt
```

Then restart the YOLO server.
