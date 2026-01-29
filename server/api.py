from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from server.app import process_frame

app = FastAPI()

@app.post("/gaze")
async def gaze(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    result = process_frame(img)
    return result
requirements.txt