import cv2
import requests

URL = "http://13.232.18.180:8000/upload"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    _, img = cv2.imencode(".jpg", frame)
    requests.post(URL, files={"file": img.tobytes()})
    