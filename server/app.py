from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np

app = FastAPI()
latest_frame = None

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_frame
    data = await file.read()
    latest_frame = cv2.imdecode(
        np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
    )
    return {"status": "ok"}

def stream():
    global latest_frame
    while True:
        if latest_frame is None:
            continue
        _, buffer = cv2.imencode(".jpg", latest_frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/video")
def video():
    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
