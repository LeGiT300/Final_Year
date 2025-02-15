import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3  # Import pyttsx3 for text-to-speech


# Initialize text-to-speech engine\

def gesture_recognition(stream):
    engine = pyttsx3.init()

    cap = cv.VideoCapture(stream)
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
        labels = [label.strip().split()[1] for label in labels]  # Remove any trailing newlines

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

            cv.rectangle(
                imgOutput,
                (x - offset, y - offset - 50),
                (x - offset + 90, y - offset - 50 + 50),
                (255, 0, 255),
                cv.FILLED,
            )
            cv.putText(
                imgOutput,
                labels[index],  # Use the label from the list
                (x, y - 26),
                cv.FONT_HERSHEY_COMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv.imshow("Hand", croppedImg)
            cv.imshow("white", imgWhite)

        cv.imshow("Hand gesture", imgOutput)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
