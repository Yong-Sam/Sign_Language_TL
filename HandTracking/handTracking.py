import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2, detectionCon=0.8)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break