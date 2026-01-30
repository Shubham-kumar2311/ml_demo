from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_frame = None

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame
    data = await file.read()
    latest_frame = cv2.imdecode(
        np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
    )
    return {"ok": True}

def mjpeg_stream():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.02)
            continue

        _, jpg = cv2.imencode(
            ".jpg", latest_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg.tobytes() + b"\r\n"
        )

@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
