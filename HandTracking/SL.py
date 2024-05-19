import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# class RequestHandler(BaseHTTPRequestHandler):
#     def _set_headers(self):
#         self.send_response(200)
#         self.send_header('Content-type', 'application/json')
#         self.end_headers()
#
#     def do_GET(self):
#         if self.path == '/latest-sign-language-code':
#             self._set_headers()
#             # Simulate a dynamic value; you might connect to a database or another data source in a real application
#             response = {
#                 'latest_code': 123  # Example static value
#             }
#             self.wfile.write(json.dumps(response).encode('utf-8'))
#         else:
#             self.send_error(404, "File Not Found: {}".format(self.path))
#
# def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
#     server_address = ('', port)
#     httpd = server_class(server_address, handler_class)
#     print(f'Server running on port {port}...')
#     httpd.serve_forever()
#
# if __name__ == '__main__':
#     run()



word_to_number = {
    'Hello': 1,
    'Meet': 2,
    'NiceTMY': 3,
    'I': 4,
    'Name': 5
}

actions = list(word_to_number) # 연속된 동작
seq_length = 30


# Mediapipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
seq = []
action_seq = []

# Unity 애플리케이션 주소
unity_app_address = 'http://localhost:5000'

model = keras.models.load_model('models/model.h5')

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] | action_seq[-2] | action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # 수어 단어에 맞는 번호를 Unity로 전송
            word_number = word_to_number.get(this_action, 0)
            if word_number:
                try:
                    response = requests.post(f'{unity_app_address}/receive_word', json={'word_number': word_number})
                    if response.status_code == 200:
                        print(f'Sent word number {word_number} to Unity application.')
                    else:
                        print(f'Failed to send word number to Unity application. Status code: {response.status_code}')
                except requests.exceptions.RequestException as e:
                    print(f'Error sending word number to Unity application: {e}')

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()