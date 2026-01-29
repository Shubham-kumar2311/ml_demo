import cv2
import requests

URL = "http://13.232.18.180:8000/gaze"

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    _, buf = cv2.imencode(".jpg", frame)

    r = requests.post(URL, files={"file": buf.tobytes()})
    print(r.json())

    cv2.imshow("Client Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
