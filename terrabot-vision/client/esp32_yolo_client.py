import cv2
import urllib.request
import numpy as np
import requests
import time

ESP32_URL = "http://192.168.4.1/cam-hi.jpg"
SERVER_URL = "http://localhost:8000/detect"
LED_URL = "http://192.168.4.1/led"

ROTATE_180 = True
SHOW_FPS = True


def fetch_frame():
    try:
        resp = urllib.request.urlopen(ESP32_URL, timeout=5)
        buf = np.array(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(buf, -1)
        if frame is None:
            return None
        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame
    except Exception as e:
        print(f"[CAM] Error: {e}")
        return None


def detect_yolo(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(SERVER_URL, files=files, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"[SERVER] HTTP {r.status_code}")
        return None
    except Exception as e:
        print(f"[SERVER] Request failed: {e}")
        return None


def draw_boxes(frame, result):
    if not result or not result.get("success"):
        cv2.putText(frame, "NO DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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


def set_led(state: str):
    try:
        urllib.request.urlopen(f"{LED_URL}?state={state}", timeout=2)
    except Exception:
        pass


def main():
    cv2.namedWindow("TerraBot Vision", cv2.WINDOW_AUTOSIZE)

    print("Controls:")
    print("  q  Quit")
    print("  l  Toggle LED")
    print("  s  Save screenshot")
    print("  r  Switch to low-res")
    print("  h  Switch to high-res")

    global ESP32_URL
    led_on = False
    fps_time = time.time()
    fps_count = 0
    current_fps = 0

    while True:
        t0 = time.time()
        frame = fetch_frame()
        if frame is None:
            time.sleep(0.3)
            continue

        result = detect_yolo(frame)
        frame = draw_boxes(frame, result)

        if SHOW_FPS:
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_count
                fps_count = 0
                fps_time = time.time()
            cv2.putText(frame, f"FPS:{current_fps}  Latency:{(time.time()-t0)*1000:.0f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("TerraBot Vision", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('l'):
            led_on = not led_on
            set_led("on" if led_on else "off")
        elif key == ord('s'):
            fname = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVE] {fname}")
        elif key == ord('r'):
            ESP32_URL = "http://192.168.4.1/cam-lo.jpg"
            print("[RES] Switched to low-res")
        elif key == ord('h'):
            ESP32_URL = "http://192.168.4.1/cam-hi.jpg"
            print("[RES] Switched to high-res")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
