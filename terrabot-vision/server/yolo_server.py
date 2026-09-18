from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

print(f"[SERVER] Loaded model: {MODEL_PATH}")
print(f"[SERVER] Classes: {list(model.names.values())}")

# Warmup: first inference is slow due to model init
print("[SERVER] Warming up model...")
model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
print("[SERVER] Warmup complete.")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    conf_thresh = float(request.args.get("conf", 0.3))
    results = model(img, verbose=False, conf=conf_thresh)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls,
                "class_name": model.names[cls]
            })

    return jsonify({
        "success": True,
        "detections": detections,
        "count": len(detections),
        "model": MODEL_PATH
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "classes": list(model.names.values())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
