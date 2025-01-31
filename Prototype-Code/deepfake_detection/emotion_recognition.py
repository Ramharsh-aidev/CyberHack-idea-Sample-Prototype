# emotion_recognition.py
from fer import FER
import cv2

# Initialize emotion detector from FER library
emotion_detector = FER()

def detect_emotion(frame):
    """Detect facial emotion using FER library."""
    # Use FER to detect the top emotion and its score
    emotion, score = emotion_detector.top_emotion(frame)
    return emotion, score
