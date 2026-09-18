# TerraBot Vision

Autonomous plant disease detection robot built with ESP32-CAM and YOLOv8. The robot captures images of plant leaves, sends them to a detection server, and identifies 29 types of plant diseases in real time.

## Architecture

```
ESP32-CAM (WiFi AP)  -->  Python Client  -->  Flask + YOLOv8 Server
   captures JPEG           fetches frames       runs inference
   serves via HTTP         draws bounding boxes  returns detections
   controls LED flash
```

## Project Structure

```
terrabot-vision/
├── esp32cam/              # Arduino firmware for ESP32-CAM
│   └── terrabot_cam.ino   # WiFi AP + JPEG streaming + LED control
├── server/                # Detection server
│   ├── yolo_server.py     # Flask API serving YOLOv8 model
│   └── requirements.txt   # Python dependencies
├── client/                # Client applications
│   ├── esp32_yolo_client.py  # ESP32-CAM live detection client
│   └── webcam_test.py        # Webcam-based testing client
└── training/              # Model training pipeline
    ├── train.py           # Full pipeline: download, merge, train
    ├── train_remote.sh    # One-shot script for GPU cloud (Vast.ai)
    ├── train_local.sh     # One-shot script for local training
    └── README.md          # Manual training guide
filters.py                 # ESP32-CAM image degradation simulator
```

## Setup

### ESP32-CAM Firmware

1. Install [Arduino IDE](https://www.arduino.cc/en/software) with ESP32 board support
2. Install the `esp32cam` library
3. Flash `terrabot-vision/esp32cam/terrabot_cam.ino` to your ESP32-CAM
4. Set your WiFi password in the `.ino` file before flashing

### Detection Server

```bash
cd terrabot-vision/server
pip install -r requirements.txt
```

Place your trained `best.pt` model in the `server/` directory, then:

```bash
python yolo_server.py
```

The server runs on `http://localhost:8000` with endpoints:
- `POST /detect` — upload an image, get bounding box detections
- `GET /health` — server status and loaded model info

### Client

Connect to the ESP32-CAM's WiFi AP (`TerraBot_CAM`), then:

```bash
cd terrabot-vision/client
python esp32_yolo_client.py
```

Controls: `q` quit, `l` toggle LED, `s` screenshot, `r`/`h` switch resolution.

For testing without hardware, use your webcam:

```bash
python webcam_test.py
```

## Training

The training pipeline merges 5 Roboflow datasets (29 plant disease classes) and applies ESP32-CAM sensor augmentation for real-world robustness.

```bash
export ROBOFLOW_API_KEY="your_api_key"
cd terrabot-vision/training
python train.py              # auto-detects GPU/CPU
python train.py --device mps # Apple Silicon
python train.py --epochs 100 --batch 8
```

### Classes (29)

Apple Scab Leaf, Apple leaf, Apple rust leaf, Bell pepper leaf spot, Bell pepper leaf, Blueberry leaf, Cherry leaf, Corn Gray leaf spot, Corn leaf blight, Corn rust leaf, Peach leaf, Potato leaf early blight, Potato leaf late blight, Potato leaf, Raspberry leaf, Soyabean leaf, Squash Powdery mildew leaf, Strawberry leaf, Tomato Early blight leaf, Tomato Septoria leaf spot, Tomato leaf bacterial spot, Tomato leaf late blight, Tomato leaf mosaic virus, Tomato leaf yellow virus, Tomato leaf, Tomato mold leaf, Tomato two spotted spider mites leaf, grape leaf black rot, grape leaf.

## License

MIT
