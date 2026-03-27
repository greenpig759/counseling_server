from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # VAD 설정
    vad_silence_threshold: float = 1.5   # 침묵 감지 임계값 (초)
    vad_speech_threshold: float = 0.5    # 음성 감지 신뢰도 임계값
    vad_sample_rate: int = 16000          # pipeline.py의 VAD_SAMPLE_RATE
    vad_chunk_samples: int = 512          # pipeline.py의 VAD_CHUNK_SAMPLES

    # STT 설정
    whisper_model_size: str = "small"     # small (한국어 정확도 향상)

    # 서버 설정
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

settings = Settings()