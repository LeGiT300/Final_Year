from flask import Flask, render_template
import threading

import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3


def gesture_recognition():
    engine = pyttsx3.init()

    cap = cv.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    # Load model without labels (since we'll get them from the file)
    classifier = Classifier("Model/keras_model.h5")

    if not cap.isOpened():
        print("Could not get camera")
        exit()

    offset = 20
    size = 300

    # Load labels from the text file
    with open('Model/labels.txt', 'r') as file:
        labels = file.readlines()  # Read all lines into a list
        labels = [' '.join(label.strip().split()[1:]) for label in labels]  # Join all parts except the first one

    while True:
        ret, frame = cap.read()
        imgOutput = frame.copy()
        hands, img = detector.findHands(frame)

        if hands:
            hand = hands[0]  # Because we are using just one hand
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((size, size, 3), np.uint8) * 255
            croppedImg = img[y - offset:y + h + offset, x - offset:x + w + offset]
            croppedShape = croppedImg.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = size / h
                wCal = math.ceil(k * w)
                imgResize = cv.resize(croppedImg, (wCal, size))
                ResizeShape = imgResize.shape
                wGap = math.ceil((size - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = size / w
                hCal = math.ceil(k * h)
                imgResize = cv.resize(croppedImg, (size, hCal))
                ResizeShape = imgResize.shape
                hGap = math.ceil((size - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {prediction}, Index: {index}")

            # Speak the predicted label using pyttsx3
            engine.say(labels[index])  # Use the label from the list
            engine.runAndWait()

            cv.putText(
                imgOutput,
                labels[index],  # Use the label from the list
                (x, y - 26),
                cv.FONT_HERSHEY_COMPLEX,
                1.7,
                (0, 0, 0),
                2,
            )
            cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv.imshow("Hand gesture", imgOutput)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


app = Flask(__name__)
gesture_thread = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    global gesture_thread
    if gesture_thread is None or not gesture_thread.is_alive():
        gesture_thread = threading.Thread(target=gesture_recognition)
        gesture_thread.start()
        return 'Recognition has started'
    else:
        return 'Recognition is already running'


if __name__ == '__main__':
    app.run(debug=True)
