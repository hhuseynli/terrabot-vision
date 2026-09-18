import cv2
import numpy as np
import requests
import time

SERVER_URL = "http://localhost:8000/detect"


def detect_yolo(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(SERVER_URL, files=files, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"[SERVER] {e}")
        return None


def draw_boxes(frame, result):
    if not result or not result.get("success"):
        return frame

    for det in result.get("detections", []):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        name = det["class_name"]

        if "healthy" in name.lower():
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    count = result.get("count", 0)
    cv2.putText(frame, f"Detected: {count}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cv2.namedWindow("TerraBot Webcam Test", cv2.WINDOW_AUTOSIZE)
    print("Controls: q=quit, s=screenshot")

    fps_time = time.time()
    fps_count = 0
    current_fps = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_yolo(frame)
        frame = draw_boxes(frame, result)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_count
            fps_count = 0
            fps_time = time.time()
        cv2.putText(frame, f"FPS:{current_fps}  Latency:{(time.time()-t0)*1000:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("TerraBot Webcam Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVE] {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
