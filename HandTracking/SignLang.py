import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import socket

# 글꼴 경로 설정
font_path = 'C:\WINDOWS\Fonts\MapoBackpacking.ttf'
# 폰트 이름 가져오기
font_name = fm.FontProperties(fname=font_path).get_name()
# 폰트 설정
plt.rc('font', family=font_name)

# # 소켓 설정
# HOST = '192.168.35.101'  # Unity가 실행되는 호스트의 IP 주소
# PORT = 50000         # Unity와 통신할 포트 번호
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((HOST, PORT))

max_num_hands = 2
gesture = {
    11: 'Hello1', 12: 'Hello2', 13: 'I', 14: '이름', 15: 'Meet1', 16: 'Meet2', 17: 'NiceTMY1', 18: 'NiceTMY2',
    # 모음 자음 추가해야 함
}

# continuous = {'감기 ':["cold1","cold2"], '아니오 ':["no1","no2"], '콧물 ':["runnynose1","runnynose2"],
#               '쓰러지다 ':["fall1","fall2"], '설사 ':["diarrhea1","diarrhea2"], '입원 ':["hospitalization1","hospitalization2","hospitalization3"],
#               '퇴원 ':["hospitalization3","hospitalization2","hospitalization1"],
#               '완쾌 ':["recovery1","recovery2","recovery3"], '소화불량 ' :["digestion1","digestion2","poor"], '변비 ':["constipation1","constipation2","constipation3"],
#               '소변 ':["urine1","urine2"], '수술 ':["surgery1","surgery2"],  '낫다 ':["","recovery3"]}
# #핵심 이미지가 여러개인 수화 동작 저장
#
# one = {'3day':'3일 ', 'yes':'네 ', 'head':'머리 ', 'stomach':'배 ', 'sick':'아프다 ','reset':'','medicine':'약 '}
# #핵심 이미지가 하나인 수화 동작 저장
#
# list_of_key = list(continuous.keys())
# list_of_value = list(continuous.values())
# #핵심 이미지가 여러개인 단어인 경우,
# #단어 별 핵심 이미지들은 value에 저장, 한국어 단어는 key에 저장
#
#b,g,r,a = 255,255,255,0


# 이미 표시된 단어들을 저장할 리스트
displayed_words = []

# displayed_words를 웹캠 화면에 표시하는 함수 정의
def display_words(img, words):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    margin = 20
    x = margin
    y = img.shape[0] - margin
    for word in words:
        text_size = cv2.getTextSize(word, font, font_scale, font_thickness)[0]
        cv2.putText(img, word, (x, y), font, font_scale, (0, 0, 0), font_thickness)
        x += text_size[0] + 2 # 단어 사이 간격 조정

# mediapipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Gesture recognition model
file = np.genfromtxt('./data/gesture_recognizer.csv', delimiter=',') # 수집한 데이터셋 파일 가져옴
angle = file[:,:-1].astype(np.float32)  # angle과 label을 데이터로 모아줌
label = file[:,-1].astype(np.float32)
knn = cv2.ml.KNearest_create()          # opencv의 k최근접 알고리즘 사용해서 학습시켜줌 -> knn모델에 데이터가 잘 학습 돼서 들어감
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():       # 웹캠에서 추가한 이미지 읽어오는데, 성공하면 위 코드 전부 실행
    ret, img = cap.read()
    if not ret:             # 성공하지 못했다면 다음 프레임으로 넘어감
        continue

    img = cv2.flip(img,1)   # 좌우 반전시켜주는 flipCode사용
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv의 BGR컬러 시스템을 cvtColor 이용해서 Mediapipe의 RGB컬러 시스템으로 변경해줌
    result = hands.process(img)     # mediapipe 모델에 넣어주기 전에 전처리 해줌
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 그 다음 이미지를 또 출력해야 하므로 다시 변환

    # 현재 프레임에서 인식된 단어들을 저장할 리스트
    detected_words = []

    if result.multi_hand_landmarks is not None:     # 전처리가 되고 모델추론까지 된 다음에 결과가 나오면 true가 되고, 손이 인식되지 않으면 false
        for res in result.multi_hand_landmarks:     # 카메라 프레임에서 계속해서 손을 감지하므로 for문 처리
            joint = np.zeros((21,3))                # 빨간 점들을 joint, 0~20까지 랜드마크로 21개가 있고, x,y,z 좌표 3개를 저장해서 21x3 개의 점 만들어줌
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]       # joint마다 랜드마크를 저장하는데, 각 랜드마크의 x,y,z 좌표 joint에 저장

            # compute angles between joints -> 각 joint로 벡터 계산해서 각도 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3] -> v2-v1 을 계산하면서 각각의 벡터 각도 계산 (각 관절의 벡터 구하는 과정)

            #Nomarlize v -> 각 벡터의 길이를 유클라디안 거리로 구해주고, 나눠주면 normalize됨
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product -> v1벡터와 v2벡터 내적(dot product)
            angle = np.arccos(np.einsum('nt, nt->n',
                                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                                         v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            # 내적 -> [v1 벡터 크기] x [v2 벡터 크기] x [두 벡터가 이루는 각의 cos값] => 위에서 벡터들 크기를 모두 1로 normalize 해줬으므로
            # 두 벡터의 내적 값 = [두 벡터가 이루는 각의 cos값] -> 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나옴

            # 이렇게 15개의 각도 구해서 angle 변수에 저장
            angle = np.degrees(angle) #convert radian(angle 값) to degree

            # inference gesture -> 위에서 학습시킨 knn모델로 inference하고 numpy array형태로 바꿔줌
            data = np.array([angle], dtype = np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때의 값을 구하고
            idx = int(results[0][0])                                  # 결과는 result 인덱스에 저장

            # 제스처 라벨 표시
            gesture_label = gesture.get(idx, 'Unknown')
            detected_words.append(gesture_label) # 현재 프레임에서 인식된 단어 추가

            # 중복된 단어 제거 후 한 문장으로 합치기
            unique_words = set(detected_words)
            sentence = ' '.join(unique_words)

            # 이미 표시된 단어와 중복되지 않는 단어만 표시
            new_words = [word for word in unique_words if word not in displayed_words]
            displayed_words.extend(new_words)

            # displayed_words를 웹캠 화면에 표시
            display_words(img, displayed_words)


            # Display
            cv2.putText(img, gesture_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('SignLanguage', img)

            # # Unity에 영상 데이터 전송
            # encoded_frame = cv2.imencode('.jpg', img)[1].tobytes()
            # sock.sendall(encoded_frame)

            if cv2.waitKey(1) == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()