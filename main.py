import cv2
import numpy as np
from golf_ball_tracker import detect_golf_ball

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No camera found.")
        return

    print("📡 Live Golf Ball Detection with Confidence Overlay")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_ball, candidates, thresh = detect_golf_ball(frame)

        # Show top 3 candidate circles (blue)
        top_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:3]
        for (cx, cy, r) in top_candidates:
            cv2.circle(frame, (cx, cy), r, (255, 0, 0), 1)

        # Show best match with confidence overlay
        if detected_ball:
            cx, cy, r, confidence = detected_ball

            # Choose color based on confidence
            if confidence >= 0.9:
                color = (0, 255, 0)
            elif confidence >= 0.7:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.circle(frame, (cx, cy), r, color, 2)
            cv2.circle(frame, (cx, cy), 2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.0%}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No confident detection", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Resize debug view to avoid lag
        debug_thresh = cv2.resize(thresh, (320, 240))

        cv2.imshow("Golf Ball Detection", frame)
        cv2.imshow("Threshold Debug", debug_thresh)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done.")

if __name__ == "__main__":
    main()
