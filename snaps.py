import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

if not cap.isOpened():
    print("Could not get camera")
    exit()

offset = 20
size = 300

folder = "img/learn"

while True:
    ret, frame = cap.read()
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

        cv.imshow("Hand", croppedImg)
        cv.imshow("white", imgWhite)

    cv.imshow("Hand gesture", frame)
    if cv.waitKey(1) == ord('S'):
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)

cap.release()
cv.destroyAllWindows()
