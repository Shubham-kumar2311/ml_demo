import cv2
import math

from mediapipe.python.solutions import face_mesh

mp_face_mesh = face_mesh


face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

BLINK_THRESHOLD = 0.35

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def process_frame(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)

    if not output.multi_face_landmarks:
        return {"face": False}

    lm = output.multi_face_landmarks[0].landmark

    def pt(i):
        return int(lm[i].x * w), int(lm[i].y * h)

    iris = pt(468)
    left, right = pt(33), pt(133)
    top, bottom = pt(159), pt(145)

    v = distance(top, bottom)
    h = distance(left, right)
    blink = v / h < BLINK_THRESHOLD if h != 0 else False

    hor_ratio = abs(left[0]-iris[0]) / (abs(left[0]-iris[0])+abs(right[0]-iris[0]))
    gaze_h = "LEFT" if hor_ratio < 0.42 else "RIGHT" if hor_ratio > 0.58 else "CENTER"

    return {
        "face": True,
        "blink": blink,
        "gaze_horizontal": gaze_h
    }
