import cv2
from cvzone.HandTrackingModule import HandDetector
import socket

# Parameters
width,height = 1280, 720

# IP WebCam
cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)

# 손 감지
detector = HandDetector(maxHands=2, detectionCon=0.8)

# 네트워크
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1", 5052)

while True:
    # 웹 캠에서 프레임 가져오기
    success, img = cap.read()
    # Hands
    hands, img = detector.findHands(img)

    data = []

    # 21개의 랜드마크 값들을 UDP 프로토콜을 사용하여 Unity에 보냄
    # Landmark values - (x,y,z)*21

    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmList = hand['lmList']
        print(lmList)
        for lm in lmList:
            data.extend([lm[0],height-lm[1],lm[2]])
            print(data)
            sock.sendto(str.encode(str(data)),serverAddressPort )

            img = cv2.resize(img, (0,0), None, 0.5, 0.5)
            cv2.imshow("Image",img)

            if cv2.waitKey(1) == ord("q"):
                break