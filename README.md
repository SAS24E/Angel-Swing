# 🏌️ Angel-Swing

_“Like an angel, this project aims to view and replay your swing in slow motion.”_

Angel-Swing is a computer vision project that helps golfers analyze their swing by detecting and tracking the golf ball using a webcam, machine learning, and OpenCV. It can classify shots with confidence overlays, collect labeled data, and train a custom deep learning model to identify golf ball positions in real-time.

---

## 📸 Features

- 🎯 **Real-time golf ball detection** using HSV masking and contour filtering  
- 🧠 **Deep learning classifier** (Keras + CNN) to distinguish ball vs. non-ball  
- 🗂️ **Dataset labeling GUI** for positive/negative crop collection  
- 🛠️ **Training pipeline** with model saving and accuracy reporting  
- 🔍 **Confidence overlay** to show detection strength visually  
- 📦 Fully modular — separate detection, training, and tracking scripts  

---

## 🧱 Project Structure

```
Angel-Swing/
├── main.py                     # Live detection with confidence overlay
├── golf_ball_detector.py      # Color filtering + contour detection
├── golf_ball_tracker.py       # Classifier-based tracking
├── dataset_builder.py         # Labeling interface for training data
├── train_golf_ball_classifier.py # CNN training & evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/SAS24E/Angel-Swing.git
cd Angel-Swing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts ctivate   # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Classifier

1. First, use the labeling tool:

```bash
python dataset_builder.py
```

Press:
- `Y` for golf ball
- `N` for not
- `Q` to quit

This creates a clean dataset in `clean_dataset/`.

2. Then train your model:

```bash
python train_golf_ball_classifier.py
```

After training, the model will be saved as `golf_ball_model.keras`.

---

## 🕵️ Run the Live Detector

```bash
python main.py
```

Features:
- Blue = top 3 candidates
- Green/Yellow/Red = detection with confidence level
- `'q'` = quit the live view

---

## 🧪 Example Output

_You can drop in screenshots or a GIF here showing a ball being detected with confidence overlay._

---

## 📝 Requirements

```text
opencv-python
tensorflow
numpy
scikit-learn
```

Use `pip freeze > requirements.txt` to update as needed.

---

## 🪪 License

[MIT License](https://choosealicense.com/licenses/mit/) (You can update this based on your preference)

---

## 🙏 Acknowledgments

- TensorFlow & Keras for deep learning
- OpenCV for real-time vision
- Inspiration: DIY Golf Sim projects + PiTrack + Garmin R10

---

## 📬 Future Plans

- Slow-motion swing playback
- Shot classification (fade/draw)
- Club recognition
- GUI frontend for easier use
