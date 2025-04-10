# File Name: golf_ball_detector.py
# Description: Uses a trained CNN model to detect golf balls from video frames by applying HSV filtering,
#              extracting candidate regions via contour detection, and classifying them based on confidence.
#              The most confident detection is returned alongside a list of candidates and the threshold mask.
# Date: 04/09/25
# References: TensorFlow, Keras, OpenCV, NumPy

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load the trained model ===
# Model Name: golf_ball_model.keras
# Description: Trained model for classifying golf balls based on cropped image patches.
# Date: 04/09/25
# References: TensorFlow, Keras, OpenCV
MODEL = load_model("golf_ball_model.keras")
IMG_SIZE = 128  # Must match your training image size
CONFIDENCE_THRESHOLD = 0.5

# Function Name: preprocess_crop
# Description: Resize, convert to grayscale, normalize, and reshape image for prediction.
# Parameter Description: crop (NumPy array) – Image patch to preprocess.
# Date: 04/09/25
# References: None
def preprocess_crop(crop):
    """Resize, convert to grayscale, normalize, and reshape for the model."""
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype("float32") / 255.0
    tensor = np.expand_dims(normalized, axis=0)
    tensor = np.expand_dims(tensor, axis=-1)
    return tensor

# Function Name: filter_contours
# Description: Filters contours based on area, preprocesses image crops, runs model prediction,
#              and tracks the best scoring candidate with confidence above threshold.
# Parameter Description: 
#   contours (list) – Contours from cv2.findContours.
#   frame (NumPy array) – Original video frame for cropping.
# Date: 04/09/25
# References: None
def filter_contours(contours, frame):
    candidates = []
    best_confidence = 0
    best_candidate = None

    for contour in contours:
        if cv2.contourArea(contour) > 300:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, r = int(x), int(y), int(radius)
            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, frame.shape[1])
            y2 = min(y + r, frame.shape[0])
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            input_tensor = preprocess_crop(crop)
            prediction = MODEL.predict(input_tensor, verbose=0)[0][0]

            candidates.append((x, y, r))  # for blue outline

            if prediction > best_confidence:
                best_confidence = prediction
                best_candidate = (x, y, r, prediction)

    return best_candidate, candidates

# Function Name: detect_golf_ball
# Description: Applies color filtering in HSV space to isolate white regions,
#              finds contours, and identifies the best circle candidate classified as a golf ball.
# Parameter Description: frame (NumPy array) – BGR image frame from webcam or video.
# Date: 04/09/25
# References: None
def detect_golf_ball(frame):
    """Detect circular shapes, classify them, and return the best match."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 200)
    upper_white = (180, 50, 255)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate, candidates = filter_contours(contours, frame)

    if best_candidate and best_candidate[3] >= CONFIDENCE_THRESHOLD:
        return best_candidate, candidates, mask
    else:
        return None, candidates, mask
