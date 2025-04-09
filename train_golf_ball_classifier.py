import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Parameters ===
IMG_SIZE = 128  # Match the resolution of your labeled images
DATASET_DIR = "clean_dataset"  # Use the new cleaned dataset
CATEGORIES = ["negative", "positive"]

# === Load your labeled images ===
def load_data():
    data, labels = [], []

    for label_index, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label_index)

    data = np.array(data, dtype="float32") / 255.0
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels)
    return data, labels

# === Define CNN ===
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Train and Save ===
def main():
    print("📥 Loading data...")
    data, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    print("🧠 Building model...")
    model = build_model()

    print("🚀 Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    print("💾 Saving model to golf_ball_model.keras")
    model.save("golf_ball_model.keras")

    print("✅ Done! Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"🎯 Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
