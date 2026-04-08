import asyncio
import logging
from typing import Dict, Optional

from ai_modules.schemas import STTInput, VADInput
from app.core.container import AIContainer
from app.core.config import settings

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD_SEC = settings.vad_silence_threshold
VAD_SAMPLE_RATE       = settings.vad_sample_rate
VAD_CHUNK_SAMPLES     = settings.vad_chunk_samples
VAD_CHUNK_BYTES       = VAD_CHUNK_SAMPLES * 4  # float32 = 4bytes
MIN_SPEECH_BYTES      = VAD_SAMPLE_RATE * 4 // 2  # 최소 발화 길이: 0.5초 (float32)
PRE_ROLL_BYTES        = VAD_SAMPLE_RATE * 4 // 2  # pre-roll 버퍼: 0.5초 (float32)


class AudioProcessor:
    """
    VAD(음성 감지) + 증분 STT 처리를 담당하는 클래스.
    CounselingPipeline에서 오디오 관련 버퍼와 로직을 위임받아 처리한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        # VAD를 통과한 음성 구간만 누적하는 STT 입력 버퍼 (침묵 감지 시 STT 큐로 전달 후 비워짐)
        self._audio_buffers: Dict[str, bytearray] = {}
        # 입력 청크를 VAD_CHUNK_BYTES 단위로 정렬하기 위한 정렬 버퍼 (잔여 바이트 임시 보관)
        self._vad_chunk_buffer: Dict[str, bytearray] = {}
        # 발화 시작 직전 0.5초를 보관하는 pre-roll 버퍼 (발화 첫 음절 잘림 방지)
        self._pre_roll: Dict[str, bytearray] = {}
        # 현재 발화 중 여부 (True: 음성 구간, False: 침묵 구간)
        self._is_speaking: Dict[str, bool] = {}
        # 발화 종료 후 연속 침묵 샘플 수 카운터 (SILENCE_THRESHOLD_SEC 초과 시 STT 큐 등록)
        self._silence_samples: Dict[str, int] = {}
        # STT 워커가 현재 추론 중인지 여부 (wait_and_get_text의 완료 대기 판단에 사용)
        self._stt_running: Dict[str, bool] = {}
        # 워커가 STT 결과를 누적하는 텍스트 버퍼 (wait_and_get_text 호출 시 반환 후 초기화)
        self._accumulated_text: Dict[str, str] = {}
        # VAD가 추출한 음성 세그먼트를 STT 워커에 전달하는 asyncio 큐
        self._transcription_queue: Dict[str, asyncio.Queue] = {}
        # 세션별 STT 워커 asyncio.Task 핸들 (세션 종료 시 cancel에 사용)
        self._transcription_tasks: Dict[str, asyncio.Task] = {}
        # VAD가 마지막으로 추출한 음성 세그먼트 (음성 감정 분석 시 pipeline에서 참조)
        self._last_audio_snapshot: Dict[str, bytes] = {}

    def init_session(self, session_id: str) -> None:
        self._audio_buffers[session_id] = bytearray()
        self._vad_chunk_buffer[session_id] = bytearray()
        self._pre_roll[session_id] = bytearray()
        self._is_speaking[session_id] = False
        self._silence_samples[session_id] = 0
        self._stt_running[session_id] = False
        self._accumulated_text[session_id] = ""
        self._transcription_queue[session_id] = asyncio.Queue(maxsize=64)
        self._last_audio_snapshot[session_id] = b""

    def cleanup_session(self, session_id: str) -> None:
        task = self._transcription_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        for buf in (
            self._audio_buffers,
            self._vad_chunk_buffer,
            self._pre_roll,
            self._is_speaking,
            self._silence_samples,
            self._stt_running,
            self._accumulated_text,
            self._transcription_queue,
            self._last_audio_snapshot,
        ):
            buf.pop(session_id, None)

    # 증분 STT 워커 시작 (세션 연결 시 호출)
    async def start_worker(self, session_id: str) -> None:
        task = asyncio.create_task(self._worker(session_id))
        self._transcription_tasks[session_id] = task

    # 증분 STT 워커 - 큐에서 오디오를 꺼내 STT 실행 후 텍스트 누적
    async def _worker(self, session_id: str) -> None:
        queue = self._transcription_queue[session_id]
        loop = asyncio.get_event_loop()
        try:
            while True:
                audio_bytes = await queue.get()
                # queue.get() 직후 즉시 설정 — wait_and_get_text의 race condition 방지
                if session_id in self._stt_running:
                    self._stt_running[session_id] = True
                try:
                    if session_id not in self._accumulated_text:
                        continue
                    stt_input = STTInput(audio_data=audio_bytes)
                    stt_result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
                    if session_id in self._accumulated_text and stt_result.text.strip():
                        prev = self._accumulated_text[session_id]
                        self._accumulated_text[session_id] = (prev + " " + stt_result.text.strip()).strip()
                        logger.info(f"[IncrSTT] {session_id}: +'{stt_result.text}' → 누적: '{self._accumulated_text[session_id]}'")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[IncrSTT] {session_id}: STT 실패: {e}")
                finally:
                    queue.task_done()
                    if session_id in self._stt_running:
                        self._stt_running[session_id] = False
        except asyncio.CancelledError:
            logger.info(f"[IncrSTT] {session_id}: 워커 종료")

    # 오디오 청크 누적 + VAD 음성/침묵 감지 → 음성 구간만 STT 버퍼에 누적, 침묵 시 STT 큐 등록
    def append_chunk(self, session_id: str, chunk: bytes) -> bool:
        self._vad_chunk_buffer[session_id].extend(chunk)

        while len(self._vad_chunk_buffer[session_id]) >= VAD_CHUNK_BYTES:
            vad_chunk = bytes(self._vad_chunk_buffer[session_id][:VAD_CHUNK_BYTES])
            self._vad_chunk_buffer[session_id] = self._vad_chunk_buffer[session_id][VAD_CHUNK_BYTES:]

            vad_result = self.container.vad.process(VADInput(audio_chunk=vad_chunk))

            if vad_result.is_speech:
                self._silence_samples[session_id] = 0
                if not self._is_speaking[session_id]:
                    # 발화 시작: pre-roll 버퍼를 먼저 붙여서 발화 첫 음절 손실 방지
                    self._is_speaking[session_id] = True
                    self._audio_buffers[session_id].extend(self._pre_roll[session_id])
                    self._pre_roll[session_id].clear()
                    logger.debug(f"[VAD] {session_id}: 발화 시작")
                # 음성 구간 → STT 버퍼에 누적
                self._audio_buffers[session_id].extend(vad_chunk)
            else:
                # 침묵 구간 → pre-roll 버퍼에 유지 (최대 PRE_ROLL_BYTES)
                self._pre_roll[session_id].extend(vad_chunk)
                if len(self._pre_roll[session_id]) > PRE_ROLL_BYTES:
                    self._pre_roll[session_id] = self._pre_roll[session_id][-PRE_ROLL_BYTES:]

                if self._is_speaking[session_id]:
                    self._silence_samples[session_id] += VAD_CHUNK_SAMPLES
                    silence_sec = self._silence_samples[session_id] / VAD_SAMPLE_RATE
                    if silence_sec >= SILENCE_THRESHOLD_SEC:
                        self._is_speaking[session_id] = False
                        self._silence_samples[session_id] = 0
                        audio_snapshot = bytes(self._audio_buffers[session_id])
                        self._audio_buffers[session_id].clear()
                        if audio_snapshot and session_id in self._transcription_queue:
                            if len(audio_snapshot) < MIN_SPEECH_BYTES:
                                logger.debug(f"[VAD] {session_id}: 세그먼트 너무 짧음 ({len(audio_snapshot)}B < {MIN_SPEECH_BYTES}B), 스킵")
                            else:
                                self._last_audio_snapshot[session_id] = audio_snapshot
                                try:
                                    self._transcription_queue[session_id].put_nowait(audio_snapshot)
                                    logger.info(f"[VAD] {session_id}: {silence_sec:.1f}초 침묵 → STT 큐 등록 ({len(audio_snapshot)}B)")
                                except asyncio.QueueFull:
                                    logger.warning(f"[VAD] {session_id}: STT 큐 가득 참, 세그먼트 버림")
                        return True  # 발화 종료 신호

        return False  # 아직 발화 중 또는 침묵 대기

    # STT 큐 완료 대기 후 누적 텍스트 반환
    async def wait_and_get_text(self, session_id: str) -> Optional[str]:
        if session_id not in self._transcription_queue:
            return None

        queue = self._transcription_queue[session_id]

        # END_OF_SPEECH 도착 시 VAD가 아직 큐에 안 넣은 잔여 음성 강제 플러시
        # (발화 직후 END_OF_SPEECH가 와서 VAD 침묵 임계값이 아직 안 찼을 경우)
        remaining = bytes(self._audio_buffers.get(session_id, b""))
        if remaining and len(remaining) >= MIN_SPEECH_BYTES:
            self._audio_buffers[session_id].clear()
            self._is_speaking[session_id] = False
            self._silence_samples[session_id] = 0
            self._last_audio_snapshot[session_id] = remaining
            try:
                queue.put_nowait(remaining)
                logger.info(f"[SpeechEnd] {session_id}: 미플러시 음성 강제 STT 큐 등록 ({len(remaining)}B)")
            except asyncio.QueueFull:
                logger.warning(f"[SpeechEnd] {session_id}: STT 큐 가득 참, 잔여 음성 버림")

        if not queue.empty() or self._stt_running.get(session_id, False):
            logger.info(f"[SpeechEnd] {session_id}: 증분 STT 완료 대기 중...")
            await queue.join()

        accumulated = self._accumulated_text.get(session_id, "").strip()

        # 폴백: 큐 처리가 안 됐고 버퍼에 오디오가 남아있는 경우
        if not accumulated:
            audio_data = bytes(self._audio_buffers[session_id])
            if not audio_data:
                return None
            logger.info(f"[SpeechEnd] {session_id}: 폴백 STT ({len(audio_data)}B)")
            stt_input = STTInput(audio_data=audio_data)
            self._audio_buffers[session_id].clear()
            loop = asyncio.get_event_loop()
            self._stt_running[session_id] = True
            try:
                stt_result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
                accumulated = stt_result.text.strip()
            finally:
                if session_id in self._stt_running:
                    self._stt_running[session_id] = False

        # 다음 발화를 위해 초기화
        if session_id in self._accumulated_text:
            self._accumulated_text[session_id] = ""

        return accumulated or None

    def get_last_audio_snapshot(self, session_id: str) -> bytes:
        """음성 감정 분석용 마지막 오디오 스냅샷 반환"""
        return self._last_audio_snapshot.get(session_id, b"")
