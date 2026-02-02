from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# 데이터 포맷 정리 코드(입력값, 출력값 둘 다 정의한 것)
# 보고 어떤 포맷이 더 올바른지 정리해서 알려주기 바람

# --- 공통 결과 포맷 ---
class EmotionResult(BaseModel):
    primary_emotion: str = Field(..., description="가장 확률이 높은 감정 (예: happy)")
    probabilities: Dict[str, float] = Field(..., description="전체 감정 확률 분포") # ex) "sad": 0.9

# --- 1. VAD (음성 감지) ---
class VADInput(BaseModel):
    audio_chunk: bytes = Field(..., description="실시간 오디오 바이너리 청크")

class VADOutput(BaseModel):
    is_speech: bool
    confidence: float

# --- 2. STT (Whisper) ---
class STTInput(BaseModel):
    audio_data: bytes = Field(..., description="발화가 완료된 오디오 전체")
    language: str = "ko"

class STTOutput(BaseModel):
    text: str
    language: str

# --- 3. Vision (Face Emotion) ---
class FaceInput(BaseModel):
    video_frame: Any = Field(..., description="OpenCV 이미지 배열 혹은 바이너리")

# --- 4. LLM (상담 생성) ---
class LLMContext(BaseModel):
    user_text: str
    face_emotion: Optional[str] = None
    voice_emotion: Optional[str] = None
    text_emotion: Optional[str] = None
    history: List[Dict[str, str]] = [] # [ {"content": "..."}, ...]

class LLMResponse(BaseModel):
    reply_text: str
    suggested_action: Optional[str] = None