# File Name: label_golf_ball_data.py
# Description: Launches a live webcam feed and allows manual labeling of detected golf ball candidates.
#              Saves crops as either positive or negative samples to build a clean training dataset.
#              Keys: [Y] = golf ball, [N] = not a golf ball, [Q] = quit.
# Date: 04/09/25
# References: OpenCV, NumPy, golf_ball_detector

import cv2
import os
import numpy as np
from golf_ball_detector import detect_golf_ball

# Output folders
POS_DIR = "clean_dataset/positive"
NEG_DIR = "clean_dataset/negative"
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

IMG_SIZE = 128

# Function Name: preprocess_crop
# Description: Converts a cropped BGR image to grayscale, pads to square, resizes to 128x128,
#              sharpens the result, and returns the processed image.
# Parameter Description: crop (NumPy array) – Input image crop.
# Date: 04/09/25
# References: OpenCV, NumPy
def preprocess_crop(crop):
    """Resize, pad to square, sharpen, and return cleaned crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Pad to square
    h, w = gray.shape
    size = max(h, w)
    padded = np.full((size, size), 128, dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = gray

    # Resize to 128x128
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    return sharpened

# Function Name: save_crop
# Description: Saves the processed crop to the appropriate dataset folder based on label.
# Parameter Description:
#   image (NumPy array) – The processed image to save.
#   label (str) – 'y' for positive or 'n' for negative.
#   index (int) – Index for file naming.
# Date: 04/09/25
# References: None
def save_crop(image, label, index):
    path = POS_DIR if label == 'y' else NEG_DIR
    filename = os.path.join(path, f"{label}_{index:04}.png")
    cv2.imwrite(filename, image)
    print(f"✅ Saved to {filename}")

# Function Name: main
# Description: Initializes webcam feed and detection loop. Prompts the user to label each
#              candidate crop as a golf ball or not, and saves images accordingly.
# Parameter Description: None
# Date: 04/09/25
# References: OpenCV
def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    index = 0
    print("📸 Press [Y] for golf ball, [N] for not, [Q] to quit. Click the preview window first!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_ball, candidates, thresh = detect_golf_ball(frame)

        for (cx, cy, r) in candidates:
            x1 = max(cx - r, 0)
            y1 = max(cy - r, 0)
            x2 = min(cx + r, frame.shape[1])
            y2 = min(cy + r, frame.shape[0])
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 20 or crop.shape[1] < 20:
                print("⚠️ Skipped: crop too small.")
                continue

            processed = preprocess_crop(crop)

            zoomed = cv2.resize(processed, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("🔍 Zoomed Preview", zoomed)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                save_crop(processed, 'y', index)
                index += 1
            elif key == ord('n'):
                save_crop(processed, 'n', index)
                index += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("👋 Labeling complete.")
                return

        cv2.imshow("Live Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done.")

if __name__ == "__main__":
    main()
