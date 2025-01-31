# face_swap_detection.py
import cv2
import dlib
import numpy as np

# Load pre-trained shape predictor model (dlib's facial landmark detector)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_landmarks(frame):
    """Detect facial landmarks using dlib."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append(shape)
    return landmarks, faces

def analyze_face_swap(frame, prev_landmarks):
    """Detect potential face swap by analyzing facial landmarks and texture anomalies."""
    landmarks, faces = detect_landmarks(frame)

    if not landmarks:
        return frame, prev_landmarks  # No faces detected, return unchanged frame

    for idx, (landmark, face) in enumerate(zip(landmarks, faces)):
        # Compare the current face landmarks with the previous ones (if any)
        if prev_landmarks is not None:
            # Check if landmarks are dramatically different from the previous frame (possible swap)
            for (x1, y1), (x2, y2) in zip(prev_landmarks[0], landmark.parts()):
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance > 10:  # Threshold to detect face swap anomalies
                    cv2.putText(frame, 'Face Swap Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break  # Stop once a face swap is detected
        
        # Draw landmarks on the frame
        for point in landmark.parts():
            cv2.circle(frame, (point.x, point.y), 1, (0, 255, 0), -1)

    return frame, landmarks

