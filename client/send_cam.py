import cv2
import requests
import time

URL = "http://13.232.18.180:8000/upload"

cap = cv2.VideoCapture(0)

# Optional: reduce camera resolution at source
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (extra safety, very important)
    frame = cv2.resize(frame, (320, 240))

    # Encode with lower JPEG quality (BIG speed gain)
    _, img = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    )

    try:
        requests.post(
            URL,
            files={"file": img.tobytes()},
            timeout=1
        )
    except requests.exceptions.RequestException:
        pass  # ignore dropped frames

    # Limit FPS (~20 FPS)
    time.sleep(0.05)
