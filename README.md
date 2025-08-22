# 평가 시스템 개요
이 평가 시스템은 생성된 이미지의 품질과 프롬프트와의 관련성을 객관적인 수치로 나타내는 것을 목표로 한다. 이를 위해 다음 세 가지 주요 지표를 사용하며, 각 지표를 정규화한 후 가중치를 적용하여 최종 점수를 계산한다. 

1. 텍스트-이미지 의미 일치도 (CLIP Score): 입력된 상세 프롬프트(내용)와 생성된 이미지 간의 의미적 관련성을 평가한다. 
2. 주제 의미 유사도 (BLIP2 Text Similarity): 생성된 이미지로부터 추출한 캡션과 핵심 프롬프트(주제) 간의 의미적 유사도를 평가한다. 
3. 이미지 시각 품질 유사도 (LPIPS): 동일한 프롬프트 내에서 생성된 다른 이미지 또는 기준 이미지와의 시각적 일관성과 품질을 평가한다. 

# 구현 단계별 상세 안내
## 1단계: 지표별 원시 점수(Raw Score) 계산
1. 텍스트-이미지 의미 일치도 (CLIP Score) 계산
목적: 프롬프트와 생성된 이미지 간의 의미적 일치도(코사인유사도)를 측정한다.
사용 모델: openai/clip-vit-base-patch32
측정 방식: cosine_similarity(embedding(ca ption), embedding(prompt)) or BLEURT / BERTScore
프로세스:
- Hugging Face의 transformers 라이브러리를 사용하여 CLIP 모델과 프로세서를 로드한다.
- 평가할 이미지와 상세 프롬프트(내용, content)를 모델에 입력한다.
- 모델 출력값인 이미지와 텍스트의 임베딩 간의 코사인 유사도를 계산하여 CLIP Score를 얻는다. 

2. 주제 의미 유사도 (BLIP2 Text Similarity) 계산
목적: 생성된 이미지가 핵심 주제를 얼마나 잘 반영하는지 평가한다.
사용 모델: Salesforce/blip2-opt-2.7b  및 문장 임베딩 모델(e.g., distiluse-base-multilingual-cased-v1)
측정 방식: cosine_similarity(clip(image), clip(text))
프로세스:
- transformers를 이용해 BLIP2 모델을 로드한다.
- 생성된 이미지를 BLIP2 모델에 입력하여 이미지에 대한 캡션(설명)을 생성한다. 
- sentence-transformers 라이브러리를 사용하여 생성된 캡션과 원본 주제 프롬프트(title) 간의 코사인 유사도를 계산한다.  이 값이 'Text Similarity' 점수가 된다.

3. 이미지 시각 품질 유사도 (LPIPS) 계산
목적: 동일 주제로 생성된 이미지들 간의 시각적 일관성을 측정한다. 점수가 낮을수록 시각적으로 유사함을 의미한다. 
사용 모델: alex 네트워크 기반 LPIPS 모델. 
측정 방식: lpips(real_img, generated _img) → 낮을수록 유사
프로세스:
- lpips 라이브러리를 사용하여 AlexNet 기반의 평가 모델을 로드한다.
- 두 개의 이미지(e.g., 기준 이미지와 생성된 이미지)를 모델에 입력한다.
- 모델이 출력하는 LPIPS 점수를 얻는다.

## 2단계: 지표별 점수 정규화 (Normalization)
계산된 각 지표의 원시 점수(Raw Score)는 스케일이 다르므로, 0과 1 사이의 값으로 변환하는 정규화 과정이 필요하다. 

### CLIP Score 정규화 (clip_norm) 
clip_score >= 0.4 이면: clip_norm = 1.0
clip_score < 0.4 이면: clip_norm = (clip_score + 1) / 1.4

### LPIPS Score 정규화 (lpips_norm) 
lpips_score <= 0.6 이면: lpips_norm = 1.0
0.6 < lpips_score <= 0.75 이면: lpips_norm = 0.9
0.75 < lpips_score <= 0.95 이면: lpips_norm = 0.89 - ((lpips_score - 0.75) / 0.2) * (0.89 - 0.1) (선형 감소)
lpips_score > 0.95 이면: lpips_norm = 0.0

### BLIP2 유사도 정규화 (blip2_norm) 
similarity >= 0.7 이면: blip2_norm = 1.0
similarity < 0.7 이면: blip2_norm = similarity / 0.7

## 3단계: 최종 점수 계산
정규화된 각 지표 점수에 가중치를 곱하여 합산한 후, 백분율로 변환하여 최종 점수를 산출한다.
최종 점수 계산식:
Final Score=(clip_norm×0.2+lpips_norm×0.3+blip2_norm×0.5)×100

# 평가 프로세스 진행
1. Blip2에 들어갈 주제 텍스트, Clip score에 들어갈 프롬프트 텍스트, 입력
2. 평가할 이미지들 첨부
3. 이미지 들 중 Lips score의 비교 대상으로 들어갈 대표 이미지 하나 선택 
4. 각 이미지 별 평가 항목 점수 - 점수(정규화된 점수), 최종 점수 표로 정리해서 사용자 인터페이스에 표시

# 구현 아키텍쳐
- python
- flask
- transformers 
- pytorch

** hugging face API -> ./.env 의 HUGGING_FACE_API 변수로 저장 **  

** 프로세스 실행 ** 
``` bash 
uv run main.py
```