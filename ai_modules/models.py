"""
실제 AI 모듈 구현
- 친구의 interfaces.py 인터페이스에 맞춰 구현
- DummyXXX → 실제 모델로 교체
"""

import torch
import numpy as np
from typing import Any

from .schemas import (
    VADInput, VADOutput, STTInput, STTOutput,
    EmotionResult, LLMContext, LLMResponse, FaceInput
)
from .interfaces import (
    BaseVADModel, BaseSTTModel, BaseEmotionModel, BaseLLMModel
)


# ============================================================
# 1. VAD - Silero VAD
# ============================================================

class SileroVADModel(BaseVADModel):
    """Silero VAD - 음성 활동 감지"""

    def __init__(self):
        self.model = None

    def load_model(self):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        self.get_speech_timestamps = utils[0]
        print("[VAD] Silero VAD 로드 완료")

    def process(self, input_data: VADInput) -> VADOutput:
        # audio_chunk를 torch tensor로 변환
        audio = np.frombuffer(input_data.audio_chunk, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio)

        # VAD 판단
        confidence = self.model(audio_tensor, 16000).item()
        is_speech = confidence > 0.5

        return VADOutput(is_speech=is_speech, confidence=confidence)


# ============================================================
# 2. STT - Whisper small
# ============================================================

class WhisperSTTModel(BaseSTTModel):
    """Whisper small - 음성 → 텍스트 변환"""

    def __init__(self, model_size="small"):
        self.model = None
        self.model_size = model_size

    def load_model(self):
        import whisper
        self.model = whisper.load_model(self.model_size)
        print(f"[STT] Whisper {self.model_size} 로드 완료")

    def transcribe(self, input_data: STTInput) -> STTOutput:
        import tempfile
        import os

        # bytes → 임시 wav 파일로 저장 후 transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(input_data.audio_data)
            temp_path = f.name

        try:
            result = self.model.transcribe(
                temp_path,
                language=input_data.language,
            )
            text = result["text"].strip()
        finally:
            os.unlink(temp_path)

        return STTOutput(text=text, language=input_data.language)


# ============================================================
# 3-a. 음성 감정 분석 - wav2vec2-xlsr
# ============================================================

class Wav2VecEmotionModel(BaseEmotionModel):
    """wav2vec2-large-xlsr-53 파인튜닝 - 음성 감정 분류 (86%)"""

    EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

    def __init__(self, model_path="models/base/voice-emotion"):
        self.model = None
        self.processor = None
        self.model_path = model_path

    def load_model(self):
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        print("[Audio Emo] wav2vec2-xlsr 감정 모델 로드 완료 (86%)")

    def analyze(self, input_data: STTInput) -> EmotionResult:
        import soundfile as sf
        import io

        # bytes → numpy array
        audio, sr = sf.read(io.BytesIO(input_data.audio_data))

        # 16kHz로 리샘플링 (필요 시)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # 모델 추론
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # 감정 확률 딕셔너리
        prob_dict = {emo: round(p, 4) for emo, p in zip(self.EMOTIONS, probs)}
        primary = max(prob_dict, key=prob_dict.get)

        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)


# ============================================================
# 3-b. 텍스트 감정 분석 - klue/bert
# ============================================================

class TextEmotionModel(BaseEmotionModel):
    """klue/bert-base 파인튜닝 - 텍스트 감정 분류 (93%)"""

    EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

    def __init__(self, model_path="models/base/text-emotion"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path

    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        print("[Text Emo] klue/bert 감정 모델 로드 완료 (93%)")

    def analyze(self, input_data: Any) -> EmotionResult:
        """input_data: str (텍스트)"""
        text = input_data if isinstance(input_data, str) else str(input_data)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        prob_dict = {emo: round(p, 4) for emo, p in zip(self.EMOTIONS, probs)}
        primary = max(prob_dict, key=prob_dict.get)

        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)


# ============================================================
# 3-c. 얼굴 감정 분석 - DeepFace
# ============================================================

class DeepFaceEmotionModel(BaseEmotionModel):
    """DeepFace - 얼굴 감정 분류 (84%)"""

    EMOTION_MAP = {
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'surprise': 'surprise',
        'fear': 'fear',
        'disgust': 'disgust',
        'neutral': 'neutral',
    }

    def __init__(self):
        pass

    def load_model(self):
        # DeepFace는 첫 호출 시 자동 로드
        from deepface import DeepFace
        self.DeepFace = DeepFace
        print("[Face] DeepFace 로드 완료 (84%)")

    def analyze(self, input_data: FaceInput) -> EmotionResult:
        """input_data.video_frame: OpenCV 이미지 배열 (numpy)"""
        result = self.DeepFace.analyze(
            input_data.video_frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,
        )

        raw_scores = result[0]['emotion']  # {'happy': 80.5, 'sad': 5.2, ...}

        # 0~100 → 0~1로 정규화
        total = sum(raw_scores.values())
        prob_dict = {
            self.EMOTION_MAP.get(k, k): round(v / total, 4)
            for k, v in raw_scores.items()
            if k in self.EMOTION_MAP
        }

        primary = max(prob_dict, key=prob_dict.get)

        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)


# ============================================================
# 4. LLM - Qwen 2.5 3B + CBT LoRA + 감정별 LoRA
# ============================================================

class CBTLLMModel(BaseLLMModel):
    """Qwen 2.5 3B + CBT LoRA + 감정별 LoRA 스위칭"""

    def __init__(
        self,
        base_model_name="Qwen/Qwen2.5-3B-Instruct",
        cbt_adapter_path="models/base/cbt-counselor",
        lora_dir="models/lora",
    ):
        self.switcher = None
        self.base_model_name = base_model_name
        self.cbt_adapter_path = cbt_adapter_path
        self.lora_dir = lora_dir

    def load_model(self):
        from pipeline.lora_switcher import LoRASwitcher

        self.switcher = LoRASwitcher(
            base_model_name=self.base_model_name,
            cbt_adapter_path=self.cbt_adapter_path,
            lora_dir=self.lora_dir,
        )
        self.switcher.load_base_model()
        print("[LLM] Qwen 2.5 3B + CBT LoRA 로드 완료")

    def generate_response(self, context: LLMContext) -> LLMResponse:
        # 감정 융합으로 최종 감정 결정
        final_emotion = self._determine_emotion(context)

        # LoRA 스위칭
        self.switcher.switch_lora(final_emotion)

        # 메시지 구성
        messages = []

        # 시스템 프롬프트 (StepManager에서 제공받을 수도 있음)
        system_prompt = (
            f"당신은 CBT 기반 심리 상담사입니다. "
            f"내담자의 현재 감정: {final_emotion}."
        )
        messages.append({"role": "system", "content": system_prompt})

        # 대화 히스토리
        for msg in context.history:
            messages.append(msg)

        # 현재 사용자 입력
        messages.append({"role": "user", "content": context.user_text})

        # 응답 생성
        reply = self.switcher.generate(messages)

        return LLMResponse(
            reply_text=reply,
            suggested_action=None,
        )

    def _determine_emotion(self, context: LLMContext) -> str:
        """3개 모달리티 감정 중 최종 감정 결정 (간단 버전)"""
        emotions = []
        if context.face_emotion:
            emotions.append(context.face_emotion)
        if context.voice_emotion:
            emotions.append(context.voice_emotion)
        if context.text_emotion:
            emotions.append(context.text_emotion)

        if not emotions:
            return "neutral"

        # 가장 많이 나온 감정 (동일하면 text 우선)
        from collections import Counter
        counter = Counter(emotions)
        return counter.most_common(1)[0][0]


# ============================================================
# 5. 감정 융합 모듈 (추가 - schemas에 없지만 필요)
# ============================================================

class EmotionFusionModel:
    """멀티모달 감정 융합 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25)"""

    WEIGHTS = {'text': 0.40, 'voice': 0.35, 'face': 0.25}
    EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

    def fuse(
        self,
        text_result: EmotionResult = None,
        voice_result: EmotionResult = None,
        face_result: EmotionResult = None,
    ) -> EmotionResult:
        """3개 모달리티 EmotionResult → 융합된 EmotionResult"""
        fused = {emo: 0.0 for emo in self.EMOTIONS}

        if text_result:
            for emo in self.EMOTIONS:
                fused[emo] += text_result.probabilities.get(emo, 0) * self.WEIGHTS['text']

        if voice_result:
            for emo in self.EMOTIONS:
                fused[emo] += voice_result.probabilities.get(emo, 0) * self.WEIGHTS['voice']

        if face_result:
            for emo in self.EMOTIONS:
                fused[emo] += face_result.probabilities.get(emo, 0) * self.WEIGHTS['face']

        # 정규화
        total = sum(fused.values())
        if total > 0:
            fused = {k: round(v / total, 4) for k, v in fused.items()}

        primary = max(fused, key=fused.get)

        return EmotionResult(primary_emotion=primary, probabilities=fused)
