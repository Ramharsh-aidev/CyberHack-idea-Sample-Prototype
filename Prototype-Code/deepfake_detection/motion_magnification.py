# motion_magnification.py
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def advanced_motion_magnification(frame, prev_frame, magnification_factor=10):
    """Amplify even smaller motions in the face."""
    if prev_frame is None:
        return frame

    # Convert frames to grayscale for optical flow calculation
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow Calculation (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Magnify the flow magnitude
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flow_magnitude = gaussian_filter(flow_magnitude, sigma=3)  # Apply Gaussian filter for smoothing

    # Enhance the motion by magnifying it
    magnified_frame = cv2.convertScaleAbs(frame, alpha=magnification_factor)
    
    return magnified_frame
