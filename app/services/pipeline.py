import asyncio
import logging
from typing import Dict, List, Optional

from ai_modules.schemas import (
    CounselingSetup, EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer
from app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class CounselingPipeline:
    """
    WebSocket에서 수신한 데이터를 버퍼링하고 AI 모델을 순서대로 호출하는 오케스트레이터.
    오디오(VAD/STT) 처리는 AudioProcessor에 위임한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.audio = AudioProcessor(container)
        # 세션별 비오디오 버퍼
        self._counseling_setup: Dict[str, Optional[CounselingSetup]] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}

    # 세션 수명 주기

    def init_session(self, session_id: str) -> None:
        self.audio.init_session(session_id)
        self._counseling_setup[session_id] = None
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        self.audio.cleanup_session(session_id)
        for buf in (
            self._counseling_setup,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")

    async def start_transcription_worker(self, session_id: str) -> None:
        await self.audio.start_worker(session_id)

    # 초기 상담 설정 저장
    def setup_counseling(self, session_id: str, topic: str, mood: str, content: str, style: str = None) -> None:
        self._counseling_setup[session_id] = CounselingSetup(
            topic=topic,
            mood=mood,
            content=content,
            style=style
        )
        logger.info(f"[Setup] {session_id}: topic={topic} / mood={mood}")

    # 초기 CBT 질문 생성 (setup 데이터 → LLM → 초기 질문 반환)
    # TODO: AI 개발자 - LLMContext.user_text 대신 CounselingSetup을 직접 활용하는 방식으로 교체 예정
    def generate_initial_questions(self, session_id: str) -> Optional[LLMResponse]:
        setup = self._counseling_setup.get(session_id)
        if not setup:
            logger.warning(f"[InitialQ] {session_id}: 초기 설정 없음, 건너뜀")
            return None
        setup_text = f"주제: {setup.topic}, 기분: {setup.mood}, 내용: {setup.content}"
        llm_context = LLMContext(user_text=setup_text)
        response = self.container.llm.generate_response(llm_context)
        logger.info(f"[InitialQ] {session_id}: '{response.reply_text}'")
        return response

    # 오디오 청크 → AudioProcessor에 위임
    def append_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        return self.audio.append_chunk(session_id, chunk)

    # 이미지 프레임 → 얼굴 감정 분석
    def process_face_frame(self, session_id: str, image_bytes: bytes) -> EmotionResult:
        face_input = FaceInput(video_frame=image_bytes)
        result = self.container.face_emotion.analyze(face_input)
        self._face_emotion_buffer[session_id].append(result)
        logger.info(f"[Face] {session_id}: {result.primary_emotion} {result.probabilities}")
        return result

    # 발화 종료 → 증분 STT 완료 대기 → 음성 감정 → 결과 반환
    async def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        accumulated = await self.audio.wait_and_get_text(session_id)
        if not accumulated or session_id not in self._stt_text_buffer:
            logger.warning(f"[SpeechEnd] {session_id}: 텍스트 없음, 건너뜀")
            return None

        self._stt_text_buffer[session_id].append(accumulated)
        logger.info(f"[SpeechEnd] {session_id}: 최종 텍스트 = '{accumulated}'")

        # 음성 감정 - 마지막 오디오 스냅샷으로 분석
        audio_for_emotion = self.audio.get_last_audio_snapshot(session_id)
        if audio_for_emotion:
            loop = asyncio.get_event_loop()
            try:
                voice_emotion = await loop.run_in_executor(
                    None, self.container.audio_emotion.analyze, STTInput(audio_data=audio_for_emotion)
                )
                if session_id in self._voice_emotion_buffer:
                    self._voice_emotion_buffer[session_id].append(voice_emotion)
                logger.info(f"[VoiceEmotion] {session_id}: {voice_emotion.primary_emotion} {voice_emotion.probabilities}")
            except Exception as e:
                logger.error(f"[VoiceEmotion] {session_id}: 오류: {e}")

        return STTOutput(text=accumulated, language="ko")

    # STT 누적 텍스트 + 감정 → LLM 응답 생성
    def generate_response(self, session_id: str) -> Optional[LLMResponse]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer[session_id]
        voice_emotions = self._voice_emotion_buffer[session_id]

        logger.info(f"[Generate] {session_id}: face {len(face_emotions)}건, voice {len(voice_emotions)}건 → AI 전달")
        logger.info(f"[Generate] {session_id}: 누적 텍스트='{accumulated_text}'")

        llm_context = LLMContext(
            user_text=accumulated_text,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
        )
        response = self.container.llm.generate_response(llm_context)
        logger.info(f"[LLM] {session_id}: '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"[LLM] 추천 행동: '{response.suggested_action}'")

        return response


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
