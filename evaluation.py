import torch
from PIL import Image
import lpips
from transformers import (
    CLIPProcessor,
    CLIPModel,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv

# .env 파일에서 환경 변수 로드 (어느 디렉터리에서 실행해도 탐색)
load_dotenv(find_dotenv())
# Support both legacy and standard HF token variable names
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
BLIP2_MODEL_NAME = os.getenv("BLIP2_MODEL") or "Salesforce/blip2-opt-2.7b"
ENABLE_CLIP = (os.getenv("DISABLE_CLIP", "0").lower() not in {"1", "true", "yes"})
ENABLE_BLIP2 = (os.getenv("DISABLE_BLIP2", "0").lower() not in {"1", "true", "yes"})
ENABLE_LPIPS = (os.getenv("DISABLE_LPIPS", "0").lower() not in {"1", "true", "yes"})

class ImageEvaluator:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        print("Loading models...")
        
        # Flags
        self.enable_clip = ENABLE_CLIP
        self.enable_blip2 = ENABLE_BLIP2
        self.enable_lpips = ENABLE_LPIPS

        # Caches
        self._caption_cache = {}
        self._topic_embedding_cache = {}

        # CLIP 모델 로드 (코사인 유사도 계산에 사용)
        if self.enable_clip:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=HUGGING_FACE_API_KEY,
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=HUGGING_FACE_API_KEY,
            )
        else:
            self.clip_model = None
            self.clip_processor = None
        
        # BLIP2 모델 로드 (환경 변수로 모델 크기 선택 가능)
        if self.enable_blip2:
            # Try a list of candidate model ids (env first, then known public models)
            candidates = [BLIP2_MODEL_NAME]
            if BLIP2_MODEL_NAME != "Salesforce/blip2-flan-t5-xl":
                candidates.append("Salesforce/blip2-flan-t5-xl")
            if BLIP2_MODEL_NAME != "Salesforce/blip2-opt-2.7b":
                candidates.append("Salesforce/blip2-opt-2.7b")

            loaded = False
            last_error: Exception | None = None
            for model_id in candidates:
                try:
                    self.blip_processor = Blip2Processor.from_pretrained(
                        model_id,
                        token=HUGGING_FACE_API_KEY,
                    )

                    if self.device == "cuda" and torch.cuda.is_available():
                        blip_dtype = torch.float16
                        device_map = "auto"
                    else:
                        blip_dtype = torch.float32
                        device_map = "cpu"

                    self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=blip_dtype,
                        device_map=device_map,
                        token=HUGGING_FACE_API_KEY,
                    )
                    print(f"Loaded BLIP2 model: {model_id}")
                    loaded = True
                    break
                except Exception as e:  # broad: cover 404, auth, dtype, etc.
                    last_error = e
                    print(f"Failed to load BLIP2 model '{model_id}': {e}")

            if not loaded:
                print("Disabling BLIP2 metric due to model load failure.")
                if last_error:
                    print(f"Reason: {last_error}")
                self.enable_blip2 = False
                self.blip_processor = None
                self.blip_model = None
        else:
            self.blip_processor = None
            self.blip_model = None
        
        # Sentence Transformer 모델 로드
        self.st_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=self.device)
        
        # LPIPS 모델 로드
        if self.enable_lpips:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_model = None
        
        print("Models loaded successfully.")

    def calculate_clip_score(self, image_path: str, prompt: str) -> float:
        """Return cosine similarity between image and text in [-1, 1]."""
        if not self.enable_clip:
            return 0.0
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        # Move tensors to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = self.clip_model.get_image_features(inputs['pixel_values'])
            text_features = self.clip_model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask')
            )

            # Normalize and compute cosine similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().item()

        return similarity

    def calculate_blip2_similarity(self, image_path: str, topic: str) -> float:
        if not self.enable_blip2:
            return 0.0
        image = Image.open(image_path).convert("RGB")

        # 이미지 캡션 생성 (모델과 동일 디바이스로 입력 이동)
        if image_path in self._caption_cache:
            caption = self._caption_cache[image_path]
        else:
            inputs = self.blip_processor(images=image, return_tensors="pt")
            blip_device = next(self.blip_model.parameters()).device
            inputs = {k: v.to(blip_device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = self.blip_model.generate(**inputs, max_new_tokens=30)
            caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            self._caption_cache[image_path] = caption

        # 주제와 캡션의 유사도 계산 (문장 임베딩 코사인 유사도)
        if topic in self._topic_embedding_cache:
            embedding1 = self._topic_embedding_cache[topic]
        else:
            embedding1 = self.st_model.encode(topic, convert_to_tensor=True)
            self._topic_embedding_cache[topic] = embedding1
        embedding2 = self.st_model.encode(caption, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        return similarity.item()

    def calculate_lpips_score(self, image_path1, image_path2):
        if not self.enable_lpips:
            return 0.0
        # Load with PIL to avoid cv2 dependency
        img_a = Image.open(image_path1).convert("RGB")
        img_b = Image.open(image_path2).convert("RGB")
        # Ensure same size
        if img_b.size != img_a.size:
            img_b = img_b.resize(img_a.size, Image.BILINEAR)

        def pil_to_lpips_tensor(img: Image.Image) -> torch.Tensor:
            np_img = np.array(img).astype(np.float32) / 255.0  # H, W, C in [0,1]
            tensor = torch.from_numpy(np_img).permute(2, 0, 1)  # C, H, W
            tensor = tensor * 2.0 - 1.0  # to [-1,1]
            return tensor.unsqueeze(0)  # 1, C, H, W

        img1 = pil_to_lpips_tensor(img_a).to(self.device)
        img2 = pil_to_lpips_tensor(img_b).to(self.device)
        return self.lpips_model(img1, img2).item()

    def evaluate(self, image_path, ref_image_path, prompt, topic):
        clip_score = self.calculate_clip_score(image_path, prompt)
        blip2_similarity = self.calculate_blip2_similarity(image_path, topic)
        lpips_score = self.calculate_lpips_score(image_path, ref_image_path)

        clip_norm = self.normalize_clip(clip_score)
        blip2_norm = self.normalize_blip2(blip2_similarity)
        lpips_norm = self.normalize_lpips(lpips_score)

        final_score = self.calculate_final_score(clip_norm, lpips_norm, blip2_norm)

        return {
            "clip_score": clip_score,
            "clip_norm": clip_norm,
            "blip2_similarity": blip2_similarity,
            "blip2_norm": blip2_norm,
            "lpips_score": lpips_score,
            "lpips_norm": lpips_norm,
            "final_score": final_score
        }

    def normalize_clip(self, score: float) -> float:
        """README 규칙 적용:
        - score >= 0.4 -> 1.0
        - else -> (score + 1) / 1.4
        """
        if score >= 0.4:
            return 1.0
        norm = (score + 1.0) / 1.4
        # Clamp to [0,1] just in case
        return max(0.0, min(1.0, norm))


    def normalize_blip2(self, score: float) -> float:
        """README 규칙 적용:
        - score >= 0.7 -> 1.0
        - else -> score / 0.7
        """
        if score >= 0.7:
            return 1.0
        return max(0.0, score / 0.7)

    def normalize_lpips(self, score: float) -> float:
        """README 규칙 적용:
        - score <= 0.6 -> 1.0
        - 0.6 < score <= 0.75 -> 0.9
        - 0.75 < score <= 0.95 -> 0.89 - ((score - 0.75) / 0.2) * (0.89 - 0.1)
        - score > 0.95 -> 0.0
        """
        import torch
from PIL import Image
import lpips
from transformers import (
    CLIPProcessor,
    CLIPModel,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv

# .env 파일에서 환경 변수를 로드합니다. find_dotenv()는 어느 디렉토리에서 실행하든 .env 파일을 찾습니다.
load_dotenv(find_dotenv())

# Hugging Face API 토큰을 환경 변수에서 가져옵니다. (레거시 및 표준 이름 모두 지원)
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
# 사용할 BLIP2 모델을 환경 변수에서 가져옵니다. 지정하지 않으면 기본 모델을 사용합니다.
BLIP2_MODEL_NAME = os.getenv("BLIP2_MODEL") or "Salesforce/blip2-opt-2.7b"
# 각 평가 지표(CLIP, BLIP2, LPIPS)를 비활성화하는 환경 변수 설정
ENABLE_CLIP = (os.getenv("DISABLE_CLIP", "0").lower() not in {"1", "true", "yes"})
ENABLE_BLIP2 = (os.getenv("DISABLE_BLIP2", "0").lower() not in {"1", "true", "yes"})
ENABLE_LPIPS = (os.getenv("DISABLE_LPIPS", "0").lower() not in {"1", "true", "yes"})

class ImageEvaluator:
    """이미지 평가를 위한 모델과 로직을 캡슐화한 클래스입니다."""
    def __init__(self, device: str = 'cpu'):
        """
        평가에 필요한 모든 AI 모델을 초기화하고 로드합니다.
        :param device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
        """
        self.device = device
        print("평가 모델을 로드하는 중입니다...")
        
        # 각 평가 지표 활성화 여부 플래그
        self.enable_clip = ENABLE_CLIP
        self.enable_blip2 = ENABLE_BLIP2
        self.enable_lpips = ENABLE_LPIPS

        # 계산된 결과를 저장하여 중복 계산을 피하기 위한 캐시
        self._caption_cache = {}
        self._topic_embedding_cache = {}

        # CLIP 모델 로드 (이미지-텍스트 유사도 측정)
        if self.enable_clip:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=HUGGING_FACE_API_KEY,
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                token=HUGGING_FACE_API_KEY,
            )
        else:
            self.clip_model = None
            self.clip_processor = None
        
        # BLIP2 모델 로드 (이미지 캡셔닝 및 주제 유사도 측정)
        if self.enable_blip2:
            # 환경 변수 또는 기본 모델 후보군 리스트
            candidates = [BLIP2_MODEL_NAME, "Salesforce/blip2-flan-t5-xl", "Salesforce/blip2-opt-2.7b"]
            loaded = False
            last_error = None
            for model_id in sorted(list(set(candidates))): # 중복 제거 및 정렬
                try:
                    self.blip_processor = Blip2Processor.from_pretrained(model_id, token=HUGGING_FACE_API_KEY)
                    # GPU 사용 가능 시 float16으로 양자화하여 메모리 사용량 감소
                    blip_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
                    device_map = "auto" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
                    
                    self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=blip_dtype,
                        device_map=device_map,
                        token=HUGGING_FACE_API_KEY,
                    )
                    print(f"BLIP2 모델 로드 성공: {model_id}")
                    loaded = True
                    break
                except Exception as e:
                    last_error = e
                    print(f"BLIP2 모델 '{model_id}' 로드 실패: {e}")

            if not loaded:
                print("BLIP2 모델 로드 실패로 해당 지표를 비활성화합니다.")
                if last_error: print(f"실패 원인: {last_error}")
                self.enable_blip2 = False
        
        # Sentence Transformer 모델 로드 (문장 임베딩 생성)
        self.st_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=self.device)
        
        # LPIPS 모델 로드 (이미지 간 지각적 유사도 측정)
        if self.enable_lpips:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_model = None
        
        print("모든 모델이 성공적으로 로드되었습니다.")

    def calculate_clip_score(self, image_path: str, prompt: str) -> float:
        """이미지와 프롬프트 간의 CLIP 코사인 유사도를 계산합니다. [-1, 1] 범위의 값을 반환합니다."""
        if not self.enable_clip: return 0.0
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = self.clip_model.get_image_features(inputs['pixel_values'])
            text_features = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'))
            # 피처 정규화 후 코사인 유사도 계산
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().item()
        return similarity

    def calculate_blip2_similarity(self, image_path: str, topic: str) -> float:
        """BLIP2로 이미지 캡션을 생성하고, 주제와의 문장 유사도를 계산합니다."""
        if not self.enable_blip2: return 0.0
        image = Image.open(image_path).convert("RGB")

        # 이미지 캡션 생성 (캐시 확인 후 없으면 생성)
        if image_path in self._caption_cache:
            caption = self._caption_cache[image_path]
        else:
            inputs = self.blip_processor(images=image, return_tensors="pt")
            blip_device = next(self.blip_model.parameters()).device
            inputs = {k: v.to(blip_device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.blip_model.generate(**inputs, max_new_tokens=30)
            caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            self._caption_cache[image_path] = caption

        # 주제와 캡션의 임베딩 유사도 계산 (캐시 확인 후 없으면 생성)
        embedding1 = self._topic_embedding_cache.setdefault(topic, self.st_model.encode(topic, convert_to_tensor=True))
        embedding2 = self.st_model.encode(caption, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def calculate_lpips_score(self, image_path1: str, image_path2: str) -> float:
        """두 이미지 간의 LPIPS(지각적 유사도) 점수를 계산합니다. 점수가 낮을수록 유사합니다."""
        if not self.enable_lpips: return 0.0
        img_a = Image.open(image_path1).convert("RGB")
        img_b = Image.open(image_path2).convert("RGB")
        # 두 이미지의 크기를 동일하게 맞춤
        if img_b.size != img_a.size:
            img_b = img_b.resize(img_a.size, Image.Resampling.BILINEAR)

        def pil_to_lpips_tensor(img: Image.Image) -> torch.Tensor:
            """PIL 이미지를 LPIPS 모델 입력 형식의 텐서로 변환합니다."""
            np_img = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_img).permute(2, 0, 1)
            tensor = tensor * 2.0 - 1.0
            return tensor.unsqueeze(0)

        img1 = pil_to_lpips_tensor(img_a).to(self.device)
        img2 = pil_to_lpips_tensor(img_b).to(self.device)
        return self.lpips_model(img1, img2).item()

    def evaluate(self, image_path: str, ref_image_path: str, prompt: str, topic: str) -> dict:
        """모든 평가 지표를 계산하고 정규화하여 최종 점수를 반환합니다."""
        clip_score = self.calculate_clip_score(image_path, prompt) if self.enable_clip else 0.0
        blip2_similarity = self.calculate_blip2_similarity(image_path, topic) if self.enable_blip2 else 0.0
        lpips_score = self.calculate_lpips_score(image_path, ref_image_path) if self.enable_lpips else 0.0

        # 각 점수를 [0, 1] 범위로 정규화
        clip_norm = self.normalize_clip(clip_score)
        blip2_norm = self.normalize_blip2(blip2_similarity)
        lpips_norm = self.normalize_lpips(lpips_score)

        # 가중치를 적용하여 최종 점수 계산
        final_score = self.calculate_final_score(clip_norm, lpips_norm, blip2_norm)

        return {
            "clip_score": clip_score, "clip_norm": clip_norm,
            "blip2_similarity": blip2_similarity, "blip2_norm": blip2_norm,
            "lpips_score": lpips_score, "lpips_norm": lpips_norm,
            "final_score": final_score
        }

    def normalize_clip(self, score: float) -> float:
        """CLIP 점수를 [0, 1] 범위로 정규화합니다."""
        if score >= 0.4: return 1.0
        return max(0.0, min(1.0, (score + 1.0) / 1.4))

    def normalize_blip2(self, score: float) -> float:
        """BLIP2 유사도 점수를 [0, 1] 범위로 정규화합니다."""
        if score >= 0.7: return 1.0
        return max(0.0, score / 0.7)

    def normalize_lpips(self, score: float) -> float:
        """LPIPS 점수를 [0, 1] 범위로 정규화합니다. (낮을수록 좋음)"""
        if score <= 0.6: return 1.0
        if score <= 0.75: return 0.9
        if score <= 0.95: return 0.89 - ((score - 0.75) / 0.2) * (0.89 - 0.1)
        return 0.0

    def calculate_final_score(self, clip_norm: float, lpips_norm: float, blip2_norm: float) -> float:
        """정규화된 점수들에 가중치를 적용하여 100점 만점의 최종 점수를 계산합니다."""
        weights = {
            'clip': 0.2 if self.enable_clip else 0.0,
            'lpips': 0.3 if self.enable_lpips else 0.0,
            'blip2': 0.5 if self.enable_blip2 else 0.0,
        }
        weight_sum = sum(weights.values()) or 1.0
        score = (clip_norm * weights['clip'] + lpips_norm * weights['lpips'] + blip2_norm * weights['blip2'])
        return (score / weight_sum) * 100

# 전역 평가기 인스턴스 (싱글톤 패턴)
evaluator = None

def get_evaluator() -> ImageEvaluator:
    """전역 평가기 인스턴스를 반환합니다. 인스턴스가 없으면 새로 생성합니다."""
    global evaluator
    if evaluator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = ImageEvaluator(device=device)
    return evaluator

    def calculate_final_score(self, clip_norm: float, lpips_norm: float, blip2_norm: float) -> float:
        # Dynamic weighting in case some metrics are disabled
        weights = {
            'clip': 0.2 if self.enable_clip else 0.0,
            'lpips': 0.3 if self.enable_lpips else 0.0,
            'blip2': 0.5 if self.enable_blip2 else 0.0,
        }
        weight_sum = sum(weights.values()) or 1.0
        score = (
            clip_norm * weights['clip'] +
            lpips_norm * weights['lpips'] +
            blip2_norm * weights['blip2']
        )
        return (score / weight_sum) * 100

# 싱글톤처럼 사용하기 위한 평가기 인스턴스
# device = "cuda" if torch.cuda.is_available() else "cpu"
# evaluator = ImageEvaluator(device=device)
evaluator = None

def get_evaluator():
    global evaluator
    if evaluator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = ImageEvaluator(device=device)
    return evaluator