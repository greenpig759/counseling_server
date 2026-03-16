import abc
import io
import random
import numpy as np
from typing import Any
from .schemas import (
    VADInput, VADOutput, STTInput, STTOutput,
    EmotionResult, LLMContext, LLMResponse, FaceInput
)
# 인터페이스 클래스(현재는 더미 데이터로 구성)

# VAD(음성감지)
class BaseVADModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def process(self, input_data: VADInput) -> VADOutput: pass

class DummyVADModel(BaseVADModel):
    def load_model(self):
        print("[VAD] Dummy Model Loaded (Silero VAD 흉내)")
        
    def process(self, input_data: VADInput) -> VADOutput:
        # 가짜 로직
        is_talking = random.choice([True, False])
        return VADOutput(is_speech=is_talking, confidence=0.9 if is_talking else 0.1)


# STT(받아쓰기)
class BaseSTTModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def transcribe(self, input_data: STTInput) -> STTOutput: pass

class DummySTTModel(BaseSTTModel):
    def load_model(self):
        print("[STT] Dummy Whisper Model Loaded")
        
    def transcribe(self, input_data: STTInput) -> STTOutput:
        return STTOutput(text="아, 정말 힘드셨겠어요. (테스트 텍스트)", language="ko")
    
    
class BaseEmotionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def analyze(self, input_data: Any) -> EmotionResult: pass

    
# Emotion Analysis (음성 감정 분석)
class DummyAudioEmotionModel(BaseEmotionModel):
    def load_model(self):
        print("[Audio Emo] Dummy SpeechBrain Loaded")
        
    def analyze(self, input_data: STTInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="fear",  # 목소리가 떨린다고 가정
            probabilities={"fear": 0.7, "sad": 0.3}
        )


# Emotion Analysis (안면 감정 분석)
class DummyFaceEmotionModel(BaseEmotionModel):
    def load_model(self):
        print("[Face] Dummy DeepFace Loaded")
        
    def analyze(self, input_data: FaceInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="sad",
            probabilities={"sad": 0.8, "neutral": 0.2}
        )


# VAD (실제 Silero VAD)
class SileroVADModel(BaseVADModel):
    """
    Silero VAD 실제 구현체.
    입력: float32 PCM 16kHz, 512샘플(32ms) 단위 chunk
    """
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512  # Silero VAD 요구 chunk 크기 (32ms @ 16kHz)

    def load_model(self):
        import torch
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.eval()
        print("[VAD] Silero VAD Loaded")

    def process(self, input_data: VADInput) -> VADOutput:
        import torch
        audio_array = np.frombuffer(input_data.audio_chunk, dtype=np.float32)
        # chunk가 512샘플보다 짧으면 패딩
        if len(audio_array) < self.CHUNK_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.CHUNK_SAMPLES - len(audio_array)))
        tensor = torch.from_numpy(audio_array[:self.CHUNK_SAMPLES])
        confidence = float(self.model(tensor, self.SAMPLE_RATE).item())
        return VADOutput(is_speech=confidence >= 0.5, confidence=confidence)


# STT (실제 faster-whisper)
class FasterWhisperSTTModel(BaseSTTModel):
    """
    faster-whisper 실제 구현체.
    입력: STTInput.audio_data = float32 PCM bytes (16kHz, mono)
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None

    def load_model(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        print(f"[STT] faster-whisper ({self.model_size}) Loaded")

    def transcribe(self, input_data: STTInput) -> STTOutput:
        audio_array = np.frombuffer(input_data.audio_data, dtype=np.float32)
        segments, info = self.model.transcribe(audio_array, language=input_data.language)
        text = " ".join(seg.text.strip() for seg in segments)
        return STTOutput(text=text, language=info.language)


# ──────────────────────────────────────────────────────────────
# [FFmpeg 변환 유틸 - 실제 브라우저(webm/opus) 연결 시 사용]
#
# 브라우저 MediaRecorder가 webm/opus로 보내면
# faster-whisper/VAD가 요구하는 float32 PCM 16kHz로 변환 필요.
#
# 사용 위치: session_manager.py audio 처리 블록 또는 pipeline.append_audio_chunk()
#
# def webm_to_float32_pcm(webm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
#     import ffmpeg
#     out, _ = (
#         ffmpeg
#         .input("pipe:0")
#         .output("pipe:1", format="f32le", ac=1, ar=sample_rate)
#         .run(input=webm_bytes, capture_stdout=True, capture_stderr=True)
#     )
#     return np.frombuffer(out, dtype=np.float32)
# ──────────────────────────────────────────────────────────────


# 4. LLM
class BaseLLMModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def generate_response(self, context: LLMContext) -> LLMResponse: pass

class DummyLLMModel(BaseLLMModel):
    """테스트용 가짜 구현. 실제 모델 연결 시 이 클래스를 교체하면 됩니다."""

    def load_model(self):
        print("[LLM] Dummy Qwen2 Model Loaded")

    def generate_response(self, context: LLMContext) -> LLMResponse:
        # context에서 사용 가능한 데이터:
        #   context.user_text      : 이번 발화 전체 텍스트 (STT 결과 합산)
        #   context.face_emotions  : List[EmotionResult] - 프레임별 얼굴 감정
        #   context.voice_emotions : List[EmotionResult] - 발화별 음성 감정
        face_summary = context.face_emotions[0].primary_emotion if context.face_emotions else "N/A"
        voice_summary = context.voice_emotions[0].primary_emotion if context.voice_emotions else "N/A"
        return LLMResponse(
            reply_text=(
                f"사용자님, '{context.user_text}'라고 하셨군요. "
                f"(얼굴: {face_summary}, 목소리: {voice_summary}) 많이 속상하셨겠습니다."
            ),
            suggested_action="심호흡 하기"
        )