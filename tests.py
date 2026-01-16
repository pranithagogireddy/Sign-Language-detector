from flask import Flask, request, render_template, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
from gtts import gTTS
import pygame
import tempfile

app = Flask(__name__)

# Upload path
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels
model_path = "keras_model.h5"
labels_path = "labels.txt"
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)
labels = ["hello", "peace", "dislike", "good luck"]

# Constants
offset = 20
imgSize = 300

# Initialize pygame for audio playback
pygame.mixer.init()

def speak_text(text):
    """Generate and play TTS for the detected label."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            path = fp.name
            tts.save(path)

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

        os.remove(path)
    except Exception as e:
        print(f"Speech error: {e}")

@app.route('/')
def index():
    return render_template("detector.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    last_spoken_label = None
    detected_labels = []

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, _ = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
            except:
                continue

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if index is not None and 0 <= index < len(labels):
                label = labels[index]

                if label != last_spoken_label:
                    speak_text(label)
                    last_spoken_label = label
                    if label not in detected_labels:
                        detected_labels.append(label)
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            last_spoken_label = None

    cap.release()
    return jsonify({'detected_signs': detected_labels})

if __name__ == '__main__':
    app.run(debug=True)
