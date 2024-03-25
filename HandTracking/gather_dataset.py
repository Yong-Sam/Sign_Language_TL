# from cvzone.HandTrackingModule import HandDetector
import cv2
import socket
import mediapipe as mp
import numpy as np

max_num_hands = 2
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

gesture = {
    0: 'Unknown', 1: 'Closed_Fist', 2: 'Open_Palm', 3: 'Pointing_Up', 4: 'Thumb_Down',
    5: 'Thumb_Up', 6: 'Victory', 7: 'ILoveYou', 8: 'Hello'
}  # 7가지 제스처, 제스처 데이터는 손가락 관절의 각도와 각각의 라벨 뜻함

# Gesture recognition data
file = np.genfromtxt('./data/gesture_train.task', delimiter=',')
print(file.shape)

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
# success, img = cap.read()
# h, w, _ = img.shape
# detector = HandDetector(detectionCon=0.8, maxHands=2)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# click함수로 데이터셋 추가
def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)


cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10 ,11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]

            # Nomarlize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt, nt->n',
                                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            angle = np.degrees(angle)  # convert radian to degree
            data = np.array([angle], dtype=np.float32)
            data = np.append(data, 8)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            sock.sendto(str.encode(str(data)), serverAddressPort)

# while True:
    #  Get image frame
    # success, img = cap.read()
    #  Find the hand and its landmarks
    # hands, img = detector.findHands(img)  #  with draw
    # hands = detector.findHands(img, draw=False)  #  without draw
    # data = []

    # if hands:
        #  Hand 1
        # hand = hands[0]
        # lmList = hand["lmList"]  # List of 21 Landmark points
        # for lm in lmList:
            # data.extend([lm[0], h - lm[1], lm[2]])

        # sock.sendto(str.encode(str(data)), serverAddressPort)

    # Display
    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

    np.savetxt('./data/gesture_recognizer.task', file, delimiter=',')