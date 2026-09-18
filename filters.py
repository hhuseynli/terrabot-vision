import cv2
import numpy as np
from pathlib import Path


def esp32cam_degrade(image):
    image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

    # Actually darken
    gamma = np.random.uniform(1.5, 3.0)

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    image = cv2.LUT(image, table)

    # Gaussian noise
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)

    image = np.clip(
        image.astype(np.int16) + noise,
        0,
        255
    ).astype(np.uint8)

    # Color noise
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] += np.random.normal(
        0, 10, hsv.shape[:2]
    )

    hsv[:, :, 0] += np.random.normal(
        0, 5, hsv.shape[:2]
    )

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # JPEG compression
    quality = np.random.randint(40, 75)

    _, enc = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    image = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Blur
    if np.random.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)

    return image


# -----------------------------
# Camera
# -----------------------------

print("Opening Mac webcam...")

cap = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    print("\nERROR: Could not open webcam.")
    print("\nCheck:")
    print("System Settings → Privacy & Security → Camera")
    print("Make sure your Terminal/Python application has permission.")
    exit()


# Don't force a resolution initially.
# Let macOS choose the camera's native mode.

print("Camera opened successfully.")
print("Reading frames...")


# Test camera
for i in range(30):

    ret, frame = cap.read()

    if ret and frame is not None:
        print(
            f"Camera working! "
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
        )
        break

else:
    print("\nERROR: Camera opened but no frames were received.")
    cap.release()
    exit()


# -----------------------------
# Output directory
# -----------------------------

output_dir = Path("webcam_esp32")
output_dir.mkdir(exist_ok=True)

counter = 0


print("\nControls:")
print("SPACE → capture degraded image")
print("Q     → quit")


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        continue

    cv2.imshow(
        "Mac Webcam - SPACE to capture",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        degraded = esp32cam_degrade(frame)

        filename = output_dir / f"capture_{counter:04d}.jpg"

        cv2.imwrite(
            str(filename),
            degraded
        )

        print(f"Saved: {filename}")

        cv2.imshow(
            "ESP32-CAM Simulation",
            degraded
        )

        counter += 1

    # Q
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Done.")