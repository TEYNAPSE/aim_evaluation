#!/usr/bin/env python3
import requests
import os
from PIL import Image

# 테스트용 이미지 생성
def create_test_image(filename, color):
    img = Image.new('RGB', (256, 256), color)
    img.save(filename)

# 테스트 이미지들 생성
test_images = [
    ('test_red.png', (255, 0, 0)),
    ('test_blue.png', (0, 0, 255)),
]

uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

for filename, color in test_images:
    filepath = os.path.join(uploads_dir, filename)
    create_test_image(filepath, color)
    print(f"테스트 이미지 생성: {filepath}")

# Flask 서버에 POST 요청 보내기
url = 'http://localhost:8081/evaluate'
data = {
    'topic': '색상 테스트',
    'prompt': '빨간색과 파란색 사각형',
    'ref_image': 'test_red.png'
}

files = []
for filename, _ in test_images:
    filepath = os.path.join(uploads_dir, filename)
    files.append(('images', (filename, open(filepath, 'rb'), 'image/png')))

try:
    print("서버에 요청 전송 중...")
    response = requests.post(url, data=data, files=files)
    print(f"응답 상태 코드: {response.status_code}")
    
    if response.status_code == 200:
        print("성공! 응답 받음")
        print(f"응답 길이: {len(response.text)} characters")
        # 응답 일부 출력
        if 'results' in response.text.lower():
            print("결과 페이지가 정상적으로 반환되었습니다.")
        else:
            print("예상과 다른 응답입니다.")
            print(response.text[:500])
    else:
        print(f"오류 발생: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"요청 중 오류 발생: {e}")

finally:
    # 파일 핸들 닫기
    for _, file_tuple in files:
        if len(file_tuple) > 1:
            file_tuple[1].close()

