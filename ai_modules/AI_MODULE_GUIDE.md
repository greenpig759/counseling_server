# AI 모듈 통합 가이드

## 개요

`ai_modules/models.py`에 실제 AI 모델 구현이 들어있습니다.
기존 `interfaces.py`의 Dummy 클래스를 실제 모델로 교체하는 방식입니다.

| 기존 (Dummy) | 실제 구현 | 모델 | 정확도 |
|---|---|---|---|
| `DummyVADModel` | `SileroVADModel` | Silero VAD | - |
| `DummySTTModel` | `WhisperSTTModel` | Whisper small | - |
| `DummyAudioEmotionModel` | `Wav2VecEmotionModel` | wav2vec2-xlsr-53 | 86% |
| (없음) | `TextEmotionModel` | klue/bert-base | 93% |
| `DummyFaceEmotionModel` | `DeepFaceEmotionModel` | DeepFace | 84% |
| `DummyLLMModel` | `CBTLLMModel` | Qwen 2.5 3B + LoRA | - |
| (없음) | `EmotionFusionModel` | 가중치 융합 | - |

---

## 1. 모델 파일 다운로드

링크: https://drive.google.com/drive/folders/1jOTsFb7hXxFuWo79wtiPO6_e8cvmf9BE?usp=drive_link

Google Drive 공유 링크에서 `models/` 폴더를 다운로드하세요.

다운로드 후 서버 루트에 `models/` 폴더를 배치합니다:

```
counseling_server/
├── ai_modules/
├── app/
├── models/                    ← 여기에 배치
│   ├── cbt-counselor-final/   # CBT 상담 LoRA 어댑터
│   ├── text-emotion-final/    # 텍스트 감정 분류 (klue/bert, 93%)
│   ├── voice-emotion-final/   # 음성 감정 분류 (wav2vec2, 86%)
│   └── lora/                  # 감정별 LoRA (7개)
│       ├── happy/
│       ├── sad/
│       ├── angry/
│       ├── surprise/
│       ├── fear/
│       ├── disgust/
│       └── neutral/
└── requirements.txt
```

---

## 2. 패키지 설치

`requirements.txt`에 아래 패키지를 추가하세요:

```
torch
transformers
peft
bitsandbytes
deepface
openai-whisper
soundfile
librosa
```

설치:

```bash
pip install -r requirements.txt
```

---

## 3. container.py 수정 예시

`app/core/container.py`에서 Dummy 모델을 실제 모델로 교체합니다:

```python
from ai_modules.models import (
    SileroVADModel,
    WhisperSTTModel,
    Wav2VecEmotionModel,
    TextEmotionModel,
    DeepFaceEmotionModel,
    CBTLLMModel,
    EmotionFusionModel,
)

class Container:
    """AI 모델 관리 (Singleton)"""

    def __init__(self):
        self.vad = SileroVADModel()
        self.stt = WhisperSTTModel(model_size="small")
        self.voice_emotion = Wav2VecEmotionModel(model_path="models/voice-emotion-final")
        self.text_emotion = TextEmotionModel(model_path="models/text-emotion-final")
        self.face_emotion = DeepFaceEmotionModel()
        self.llm = CBTLLMModel(
            cbt_adapter_path="models/cbt-counselor-final",
            lora_dir="models/lora",
        )
        self.fusion = EmotionFusionModel()

    def load_all(self):
        """서버 시작 시 모든 모델 로드 (약 30초~1분)"""
        self.vad.load_model()          # ~2초
        self.stt.load_model()          # ~5초
        self.voice_emotion.load_model() # ~3초
        self.text_emotion.load_model()  # ~2초
        self.face_emotion.load_model()  # ~3초
        self.llm.load_model()          # ~15초
        print("모든 AI 모델 로드 완료!")
```

---

## 4. 각 모듈 사용법

### VAD (음성 활동 감지)

```python
from ai_modules.schemas import VADInput

result = container.vad.process(VADInput(audio_chunk=audio_bytes))
# result.is_speech: True/False
# result.confidence: 0.0~1.0
```

### STT (음성 → 텍스트)

```python
from ai_modules.schemas import STTInput

result = container.stt.transcribe(STTInput(audio_data=full_audio_bytes))
# result.text: "사용자가 말한 텍스트"
```

### 감정 분석 (3개 모달리티)

```python
# 텍스트 감정
text_emo = container.text_emotion.analyze("진짜 웃기네")
# text_emo.primary_emotion: "angry"
# text_emo.probabilities: {"angry": 0.7, "sad": 0.2, ...}

# 음성 감정
voice_emo = container.voice_emotion.analyze(STTInput(audio_data=audio_bytes))
# voice_emo.primary_emotion: "sad"

# 얼굴 감정
from ai_modules.schemas import FaceInput
face_emo = container.face_emotion.analyze(FaceInput(video_frame=opencv_image))
# face_emo.primary_emotion: "angry"
```

### 감정 융합

```python
fused = container.fusion.fuse(
    text_result=text_emo,       # 가중치 0.40
    voice_result=voice_emo,     # 가중치 0.35
    face_result=face_emo,       # 가중치 0.25
)
# fused.primary_emotion: "angry"  (최종 감정)
# fused.probabilities: {...}
```

### LLM 응답 생성

```python
from ai_modules.schemas import LLMContext

response = container.llm.generate_response(LLMContext(
    user_text="진짜 웃기네. 이런 일이 나한테 일어나다니.",
    face_emotion="angry",
    voice_emotion="angry",
    text_emotion="happy",       # 텍스트만 보면 happy지만
    history=[
        {"role": "assistant", "content": "안녕하세요. 편하게 이야기해주세요."},
        {"role": "user", "content": "네..."},
    ],
))
# response.reply_text: "정말 화가 나셨겠어요..."
# → angry LoRA가 적용되어 감정 맥락에 맞는 응답 생성
```

---

## 5. 턴별 처리 흐름 (WebSocket 연동)

```
[사용자 발화 중]
├─ 1초마다 웹캠 프레임 → container.face_emotion.analyze()
│  → 얼굴 감정 결과를 리스트에 저장 (버퍼)
├─ 음성 청크 → 오디오 버퍼에 누적

[사용자 발화 완료] (2초 침묵 or 버튼)
├─ 얼굴 버퍼 → 평균 계산 → face_result
├─ 전체 오디오 → container.stt.transcribe() → 텍스트
├─ 텍스트 → container.text_emotion.analyze() → text_result    ← 병렬
├─ 전체 오디오 → container.voice_emotion.analyze() → voice_result ← 병렬
├─ container.fusion.fuse(text, voice, face) → 최종 감정
├─ container.llm.generate_response(context) → 상담사 응답
└─ WebSocket으로 응답 전송
```

---

## 6. 주의사항

- **GPU 필수**: LLM, wav2vec2, Whisper는 GPU가 필요합니다 (최소 8GB VRAM)
- **모델 로드 시간**: 서버 시작 시 약 30초~1분 소요
- **모델 파일 용량**: 전체 약 5~8GB (Git에 올리면 안 됩니다)
- **LoRA 스위칭**: 감정이 바뀔 때만 발생 (같은 감정이면 스킵, ~1~3초)
- **병렬 처리**: STT→텍스트 감정과 음성 감정은 `asyncio.gather`로 병렬 처리 권장

---

## 7. 처리 시간 예상 (GPU 기준)

| 처리 | 소요 시간 |
|------|----------|
| 얼굴 버퍼 평균 | ~0.01초 |
| Whisper small STT | ~1초 |
| klue/bert 텍스트 감정 | ~0.1초 |
| wav2vec2 음성 감정 | ~0.5초 |
| 감정 융합 | ~0.01초 |
| LoRA 스위칭 (변경 시) | ~1~3초 |
| LLM 응답 생성 (vLLM) | ~1.5~2.5초 |
| **총 평균** | **~4초** |
