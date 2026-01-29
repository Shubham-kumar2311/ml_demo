import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import math

# 1. Setup Camera and Face Mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# 2. Variables for Blinking
blink_counter = 0
blink_counter_frame = 0  # To prevent double counting (Debouncing)
BLINK_THRESHOLD = 0.35   # If ratio is below this, it's a blink

# Helper function to calculate distance between two points (x, y)
def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # --- COORDINATE EXTRACTION ---
        def get_coords(id):
            x = int(landmarks[id].x * frame_w)
            y = int(landmarks[id].y * frame_h)
            return x, y

        # Gaze Points (Left Eye)
        iris_center = get_coords(468)
        left_corner = get_coords(33)   # Inner
        right_corner = get_coords(133) # Outer
        
        # Blink Points (Left Eye)
        top_lid = get_coords(159)
        bottom_lid = get_coords(145)

        # Draw visual markers
        cv2.circle(frame, iris_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, left_corner, 2, (0, 0, 255), -1)
        cv2.circle(frame, right_corner, 2, (0, 0, 255), -1)
        cv2.line(frame, top_lid, bottom_lid, (0, 200, 0), 1) # Vertical line for blink

        # =========================================
        # PART 1: BLINK DETECTION
        # =========================================
        
        # Calculate distances
        vertical_len = calculate_distance(top_lid, bottom_lid)
        horizontal_len = calculate_distance(left_corner, right_corner)
        
        # Calculate Ratio (Eye Aspect Ratio)
        if horizontal_len != 0:
            blink_ratio = vertical_len / horizontal_len
        else:
            blink_ratio = 1.0 # Default safe value

        # Blink Check Logic
        if blink_ratio < BLINK_THRESHOLD:
            # If counter is 0, this is a "fresh" blink
            if blink_counter_frame == 0:
                blink_counter += 1
                blink_counter_frame = 1 # Lock the counter
                print("BLINK DETECTED!")
        else:
            # Eye is open again
            if blink_counter_frame > 0:
                blink_counter_frame += 1
                # Wait 10 frames before allowing next blink (Debouncing)
                if blink_counter_frame > 10:
                    blink_counter_frame = 0

        # =========================================
        # PART 2: GAZE TRACKING
        # =========================================
        
        # Horizontal Gaze
        dist_to_left = abs(left_corner[0] - iris_center[0])
        dist_to_right = abs(right_corner[0] - iris_center[0])
        total_width = dist_to_left + dist_to_right
        
        hor_text = "CENTER"
        if total_width != 0:
            ratio_h = dist_to_left / total_width
            if ratio_h < 0.42: hor_text = "LEFT"
            elif ratio_h > 0.58: hor_text = "RIGHT"

        # Vertical Gaze
        dist_to_top = abs(top_lid[1] - iris_center[1])
        dist_to_bottom = abs(bottom_lid[1] - iris_center[1])
        total_height = dist_to_top + dist_to_bottom
        
        ver_text = "CENTER"
        if total_height != 0:
            ratio_v = dist_to_top / total_height
            if ratio_v < 0.35: ver_text = "UP"
            elif ratio_v > 0.65: ver_text = "DOWN"

        # =========================================
        # PART 3: DISPLAY
        # =========================================
        
        # Show Gaze Direction
        cv2.putText(frame, f"Gaze: {ver_text} - {hor_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show Blink Count
        cv2.putText(frame, f"Blinks: {blink_counter}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()