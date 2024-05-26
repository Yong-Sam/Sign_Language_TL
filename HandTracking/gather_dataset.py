import cv2
import mediapipe as mp
import numpy as np
import time, os

from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------
# # 글꼴 경로 설정
# font_path = 'C:\WINDOWS\Fonts\MapoBackpacking.ttf'
# # 폰트 이름 가져오기
# font_name = fm.FontProperties(fname=font_path).get_name()
# # 폰트 설정
# plt.rc('font', family=font_name)


# max_num_hands = 2
# # gesture = {
# #     11: 'Hello1', 12: 'Hello2', 13: 'I', 14: 'Name', 15: 'Meet1', 16: 'Meet2', 17: 'NiceTMY1', 18: 'NiceTMY2',
# #     # 모음 자음 추가해야 함(이름)
# # }
# continuous = {'안녕하세요 ': [0, 1], '만나서 ': [2, 3], '반갑습니다 ': [4, 5]}
# #핵심 이미지가 여러개인 수화 동작 저장
#
# one = {6: '저는 ', 7: '이름 '}
# #핵심 이미지가 하나인 수화 동작 저장
#
# list_of_key = list(continuous.keys())
# list_of_value = list(continuous.values())
# #핵심 이미지가 여러개인 단어인 경우,
# #단어 별 핵심 이미지들은 value에 저장, 한국어 단어는 key에 저장
# --------------------------------------------------------------------


actions = ['Hello'] # 연속된 동작
seq_length = 30
secs_for_action = 30 # 액션 녹화하는 시간

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# # Gesture recognition data
# file = np.genfromtxt('data/gesture_recognizer.csv', delimiter=',')
# print(file.shape)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

# def click(event, x, y, flags, param):
#     global data, file
#     if event == cv2.EVENT_LBUTTONDOWN:
#         file = np.vstack((file, data))
#         print(file.shape)
#
# cv2.namedWindow('Dataset')
# cv2.setMouseCallback('Dataset', click)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []
        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('training', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # lm.visibility = landmark가 이미지에서 보이는지

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Dataset', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

# np.savetxt('data/gesture_train_SL.csv', file, delimiter=',')
