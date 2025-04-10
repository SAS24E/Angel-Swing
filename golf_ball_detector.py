# File Name: golf_ball_tracker.py
# Description: Detects golf balls in a single video frame using HSV color filtering and contour detection.
#              Designed to work with white golf balls in standard lighting conditions.
#              Returns possible ball positions and a threshold mask for visualization or debugging.
# Date: 04/09/25
# References: OpenCV, NumPy

import cv2
import numpy as np

# Function Name: detect_golf_ball
# Description: Applies HSV color filtering to detect white regions, finds contours, filters by area,
#              and estimates circular positions for possible golf balls. Returns a mask, candidate list,
#              and a placeholder for best detection (None by default).
# Parameter Description: 
#   frame (NumPy array) – The input BGR image frame from a webcam or video feed.
# Date: 04/09/25
# References: OpenCV
def detect_golf_ball(frame):
    # Convert the image to HSV and create a mask for white color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 200)
    upper_white = (180, 50, 255)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and create a list of candidates
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Lowered threshold for smaller balls
            (x, y), radius = cv2.minEnclosingCircle(contour)
            candidates.append((int(x), int(y), int(radius)))
    
    return None, candidates, mask
