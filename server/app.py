from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()

def generate_frames():
    cap = cv2.VideoCapture("input.mp4")  # OR live frames from client later
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
