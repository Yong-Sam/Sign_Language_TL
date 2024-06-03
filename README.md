![image](https://github.com/Yong-Sam/Sign_Language_TL/assets/80036437/66c6d74d-76cb-4118-b922-b2dee55e9c7d)

---
### 개발과정
**[수화인식 > 텍스트]**

- 파이썬의 OpenCV와
- 머신러닝 알고리즘
    - Mediapipe로 인식한 손의 각 부분 벡터의 사이 각도를 구함/ 구체적으로 말하면, 각 제스처의 각도를 저장한 데이터셋을 RNN 알고리즘을 사용하여 알아내는 방법이다.
    - KNN알고리즘을 활용하여 저장된 Dataset의 정확도를 측정한다. / 정확도는 k=1~10 중, k=1일 때 가장 높은 정확도를 보인다.
- 그리고 딥러닝 알고리즘
    - Mediapipe로 인식한 손의 각 부분 벡터의 사이 각도를 구하고 이를 csv 파일로 저장한다. 이 데이터 셋을 KNN 최근접 알고리즘을 사용하여 알아낸다.
- 을 활용하여 핸드트랙킹

- 유니티와 파이썬의 Http통신 작업을 통해, 파이썬의 웹캠화면으로 인식한 수어를 유니티로 연동

- 유니티와 MR 기기를 연결하여 수화 인식 및 소통

</br>

**[음성인식 > 수화]**

- Microsoft Azure의 Speech to Text 프로그램을 활용하여 음성을 인식하여 텍스트 UI로 표시 (유니티에 연결)
- 유니티에서 텍스트를 분류하여 각 텍스트에 맞는 애니메이션 선택 > 유니티 화면에 해당 애니메이션 출력
