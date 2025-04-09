import cv2
import numpy as np

def detect_golf_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 200)
    upper_white = (180, 50, 255)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Lowered threshold
            (x, y), radius = cv2.minEnclosingCircle(contour)
            candidates.append((int(x), int(y), int(radius)))
    
    return None, candidates, mask
