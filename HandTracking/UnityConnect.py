import subprocess

def run_sign_language_recognition():
    # 파이썬에서 외부 프로그램 실행
    process = subprocess.Popen(["python", "UnityConnect.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 실행 결과 가져오기
    output, error = process.communicate()
    if error:
        print("에러 발생:", error)
    else:
        print("결과:", output.decode("utf-8"))

run_sign_language_recognition()
