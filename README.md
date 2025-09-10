# AIM (AI Image Evaluator)

## 목차
- [평가 시스템 개요](#평가-시스템-개요)
- [설치 및 실행](#설치-및-실행)
  - [Jupyter Notebook 실행](#jupyter-notebook-실행)
- [지표 설명](#지표-설명)
- [구현 단계별 상세 안내](#구현-단계별-상세-안내)
- [평가 프로세스 진행](#평가-프로세스-진행)
- [구현 아키텍처](#구현-아키텍처)

## 평가 시스템 개요
이 평가 시스템은 생성된 이미지의 품질과 프롬프트와의 관련성을 객관적인 수치로 나타내는 것을 목표로 한다. 이를 위해 다음 세 가지 주요 지표를 사용하며, 각 지표를 정규화한 후 가중치를 적용하여 최종 점수를 계산한다. 

## 지표 설명

1. **텍스트-이미지 의미 일치도 (CLIP Score)**: 입력된 상세 프롬프트(내용)와 생성된 이미지 간의 의미적 관련성을 평가한다. 
2. **주제 의미 유사도 (BLIP2 Text Similarity)**: 생성된 이미지로부터 추출한 캡션과 핵심 프롬프트(주제) 간의 의미적 유사도를 평가한다. 
3. **이미지 시각 품질 유사도 (LPIPS)**: 동일한 프롬프트 내에서 생성된 다른 이미지 또는 기준 이미지와의 시각적 일관성과 품질을 평가한다. 

## 설치 및 실행

### 필수 요구사항
- Python 3.12 이상
- uv (Python 패키지 관리자)
- Hugging Face API 토큰

### 환경 설정
1. 리포지토리 클론
```bash
git clone <repository-url>
cd aim
```

2. 환경 변수 설정
`.env` 파일을 생성하고 Hugging Face API 토큰을 설정한다:
```bash
# .env 파일
HUGGING_FACE_API=your_hugging_face_token_here
# 또는
HF_TOKEN=your_hugging_face_token_here
```

3. 종속성 설치
```bash
uv sync
```

### Jupyter Notebook 실행
더 상세한 분석과 실험을 위해 Jupyter Notebook을 사용할 수 있다:

#### 1. Jupyter 환경 설정
```bash
# Jupyter Lab 설치 및 실행
uv run jupyter lab
```

#### 2. Notebook 사용법

**Step 1: 환경 준비**
```python
# 환경 변수 로딩 및 경로 설정
import os, sys
from dotenv import load_dotenv, find_dotenv
ROOT = os.path.dirname(os.path.dirname(os.getcwd())) if os.path.basename(os.getcwd())=='notebooks' else os.getcwd()
if ROOT not in sys.path: sys.path.insert(0, ROOT)
load_dotenv(find_dotenv())
```

**Step 2: 입력 데이터 준비**
```python
# 위젯을 통한 대화형 입력
import ipywidgets as widgets
from IPython.display import display

# 주제 입력
topic_w = widgets.Textarea(
    value='A wooden cabin with a smoking chimney stands among snow-covered trees and a snowy mountain',
    description='Topic',
    layout=widgets.Layout(width='800px', height='80px')
)

# 프롬프트 입력
prompt_w = widgets.Textarea(
    value='Only one wooden cabin with one chimney, no other buildings or cabins, in a snowy forest with mountains',
    description='Prompt',
    layout=widgets.Layout(width='800px', height='80px')
)

# 이미지 업로드
files_w = widgets.FileUpload(accept='image/*', multiple=True, description='Upload Images')
```

**Step 3: 평가 실행**
업로드된 이미지들에 대해 자동으로 평가가 수행되며, 각 지표별 점수와 최종 점수가 표시된다.

#### 4. Notebook의 장점
- **대화형 분석**: 위젯을 통한 실시간 입력 및 결과 확인
- **상세한 시각화**: 이미지 썸네일 프리뷰 및 평가 결과 시각화
- **단계별 실행**: 각 평가 단계를 개별적으로 실행하고 결과 확인 가능
- **커스터마이징**: 평가 파라미터 조정 및 실험 가능

## 버전 2 새로운 기능

버전 2 평가 노트북(`AIM_Evaluation_v2.ipynb`)은 평가 결과에 대한 심층적인 분석을 제공하기 위해 다음과 같은 두 가지 주요 시각화 기능이 추가되었습니다.

### 1. CLIP 어텐션 히트맵 (Attention Heatmap)
CLIP 모델이 텍스트 프롬프트와 이미지 간의 유사성을 계산할 때 이미지의 어떤 영역에 집중하는지를 시각적으로 보여줍니다.

- **작동 원리**: CLIP의 Vision Transformer 모델 마지막 레이어에서 어텐션 가중치(attention weights)를 추출하여, 이미지의 특정 영역이 텍스트와 얼마나 관련이 있는지를 히트맵으로 표현합니다.
- **해석**: 히트맵에서 붉은색에 가까운 영역일수록 모델이 프롬프트와 강하게 연관 지어 판단한 부분입니다. 이를 통해 CLIP 점수가 왜 그렇게 산출되었는지에 대한 직관적인 근거를 파악할 수 있습니다.

### 2. BLIP2 생성 캡션 표시
BLIP2 모델이 평가 대상 이미지를 분석하여 생성한 자연어 캡션(설명)을 결과와 함께 제공합니다.

- **목적**: '주제 의미 유사도' 점수의 기반이 되는 모델의 이미지 이해도를 직접 확인할 수 있습니다.
- **활용**: 모델이 이미지의 핵심 요소를 정확히 파악하고 있는지, 또는 예상치 못한 방향으로 해석하고 있는지를 검토하여 평가 결과의 신뢰성을 높일 수 있습니다.

## 구현 단계별 상세 안내
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

## 평가 프로세스 진행
1. Blip2에 들어갈 주제 텍스트, Clip score에 들어갈 프롬프트 텍스트 입력
2. 평가할 이미지들 첨부
3. 이미지들 중 LPIPS score의 비교 대상으로 들어갈 대표 이미지 하나 선택 
4. 각 이미지별 평가 항목 점수 - 점수(정규화된 점수), 최종 점수 표로 정리해서 사용자 인터페이스에 표시

## 구현 아키텍처
- **Python**: 3.12 이상
- **Transformers**: Hugging Face 모델 라이브러리 
- **PyTorch**: 딥러닝 프레임워크
- **Jupyter**: 대화형 분석 환경
- **uv**: 패키지 관리자

### 주요 의존성
- `sentence-transformers`: 텍스트 임베딩
- `lpips`: 이미지 시각적 유사도 측정
- `pillow`: 이미지 처리
- `python-dotenv`: 환경 변수 관리

## 라이선스
MIT License

## 기여하기
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request