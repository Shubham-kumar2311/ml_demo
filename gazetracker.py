import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # --- KEY LANDMARKS FOR LEFT EYE ---
        # 468 is the center of the Iris
        # 33 is the inner corner (towards nose)
        # 133 is the outer corner (towards ear)
        # 159 is the top eyelid
        # 145 is the bottom eyelid
        
        # Helper function to get coordinates
        def get_coords(id):
            x = int(landmarks[id].x * frame_w)
            y = int(landmarks[id].y * frame_h)
            return x, y

        # Get coordinates
        iris_center = get_coords(468)
        left_corner = get_coords(33)   # Inner corner
        right_corner = get_coords(133) # Outer corner
        top_lid = get_coords(159)
        bottom_lid = get_coords(145)

        # Draw points for visualization
        cv2.circle(frame, iris_center, 3, (0, 255, 0), -1) # Green Iris
        cv2.circle(frame, left_corner, 2, (0, 0, 255), -1) # Red Corners
        cv2.circle(frame, right_corner, 2, (0, 0, 255), -1)

        # --- LOGIC FOR LEFT / RIGHT ---
        # Calculate horizontal distances
        dist_to_left = abs(left_corner[0] - iris_center[0])
        dist_to_right = abs(right_corner[0] - iris_center[0])
        
        # Total eye width
        eye_width = dist_to_left + dist_to_right
        
        # Avoid division by zero
        if eye_width != 0:
            # Ratio: 0 means looking full left, 1 means full right, 0.5 is center
            hor_ratio = dist_to_left / eye_width

            if hor_ratio < 0.42:
                hor_text = "LEFT"
            elif hor_ratio > 0.58:
                hor_text = "RIGHT"
            else:
                hor_text = "CENTER"
        
        # --- LOGIC FOR UP / DOWN ---
        # Calculate vertical distances
        dist_to_top = abs(top_lid[1] - iris_center[1])
        dist_to_bottom = abs(bottom_lid[1] - iris_center[1])
        
        # Ratio for vertical
        ver_total = dist_to_top + dist_to_bottom
        if ver_total != 0:
             ver_ratio = dist_to_top / ver_total
             
             if ver_ratio < 0.35: # Iris is very close to top lid
                 ver_text = "UP"
             elif ver_ratio > 0.65: # Iris is very close to bottom lid
                 ver_text = "DOWN"
             else:
                 ver_text = "CENTER"

        # Display the result
        cv2.putText(frame, f"H: {hor_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"V: {ver_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()