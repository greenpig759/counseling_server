"""
WebSocket 테스트 클라이언트 - test_video1.mp4 사용
mp4에서 오디오/영상을 추출해 실제 WebSocket으로 서버에 전송.

실행 순서:
    1. 서버 먼저 실행: uvicorn app.main:app --reload
    2. 이 스크립트 실행: python -m tests.test_ws_client

전송 방식:
    - 오디오: float32 PCM 16kHz → base64 → { type: "audio", data: ... }
    - 영상: JPEG bytes → base64 → { type: "video", data: ... }
    - 30초 시점: { type: "control", data: "END_OF_SPEECH" }
"""

import asyncio
import base64
import json
import logging
import os
import cv2
import ffmpeg
import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_files", "test_video1.mp4")
WS_URL = "ws://localhost:8000/ws/counseling/test-ws-client"

AUDIO_CHUNK_SEC = 0.032      # 512샘플 @ 16kHz = 32ms (Silero VAD 요구 크기)
FRAME_INTERVAL = 90           # 90프레임마다 1장 전송 (30fps 기준 3초마다 1장)
END_OF_SPEECH_INTERVAL = 15.0 # 15초마다 END_OF_SPEECH 전송 (15, 30, 45, 60초)
TEMP_FRAME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_frames")


def load_audio_chunks(video_path: str):
    """mp4에서 오디오를 읽어 float32 PCM 16kHz 청크 리스트로 반환. ffmpeg 사용."""
    out, _ = (
        ffmpeg
        .input(video_path)
        .output("pipe:1", format="f32le", ac=1, ar=16000)
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio_np = np.frombuffer(out, dtype=np.float32)
    total_sec = len(audio_np) / 16000
    logger.info(f"[Audio] {total_sec:.1f}초, 16kHz mono float32 변환 완료")

    chunk_samples = int(16000 * AUDIO_CHUNK_SEC)  # 512
    chunks = []
    for start in range(0, len(audio_np), chunk_samples):
        segment = audio_np[start:start + chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        chunks.append(segment.tobytes())

    logger.info(f"[Audio] {AUDIO_CHUNK_SEC*1000:.0f}ms 단위 → {len(chunks)}개 청크")
    return chunks, total_sec


def load_video_frames(video_path: str):
    """mp4에서 프레임을 추출해 (frame_idx, jpeg_bytes, timestamp_sec) 리스트로 반환.
    추출한 프레임은 temp_frames/ 폴더에 저장해 육안 확인 가능.
    """
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                timestamp = frame_idx / fps
                jpeg_bytes = buf.tobytes()
                frames.append((frame_idx, jpeg_bytes, timestamp))

                # temp_frames/ 에 저장 (육안 확인용)
                save_path = os.path.join(TEMP_FRAME_DIR, f"frame_{saved_count:03d}_{timestamp:.1f}s.jpg")
                with open(save_path, "wb") as f:
                    f.write(jpeg_bytes)
                saved_count += 1
        frame_idx += 1

    cap.release()
    logger.info(f"[Video] 총 {frame_idx}프레임 중 {len(frames)}장 추출 (매 {FRAME_INTERVAL}프레임마다 1장, {fps:.0f}fps 기준 3초마다 1장)")
    logger.info(f"[Video] temp_frames/ 에 {saved_count}장 저장 완료 → 육안 확인 가능")
    return frames, fps


async def recv_until(ws, target_status: str, timeout: float = 120.0) -> str:
    """target_status가 올 때까지 다른 메시지는 로그만 찍고 넘김."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        try:
            data = json.loads(msg)
            if data.get("status") == target_status:
                return msg
            else:
                logger.info(f"[서버 수신 (대기 중)] {msg}")
        except Exception:
            logger.info(f"[서버 수신 (파싱 불가)] {msg}")


async def run():
    logger.info(f"서버 연결 중: {WS_URL}")

    audio_chunks, total_sec = load_audio_chunks(VIDEO_PATH)
    video_frames, fps = load_video_frames(VIDEO_PATH)

    async with websockets.connect(WS_URL) as ws:
        logger.info("WebSocket 연결 완료")

        # 서버 연결 응답 수신
        msg = await ws.recv()
        logger.info(f"[서버] {msg}")

        # 오디오/영상을 시간 순서대로 전송
        video_frame_ptr = 0
        next_eos_sec = END_OF_SPEECH_INTERVAL  # 다음 END_OF_SPEECH 전송 시점

        for audio_idx, chunk_bytes in enumerate(audio_chunks):
            current_sec = audio_idx * AUDIO_CHUNK_SEC

            # 해당 시점의 영상 프레임 전송 (fire and forget - Ack 기다리지 않음)
            while video_frame_ptr < len(video_frames):
                _, jpeg_bytes, frame_ts = video_frames[video_frame_ptr]
                if frame_ts > current_sec:
                    break
                b64 = base64.b64encode(jpeg_bytes).decode()
                await ws.send(json.dumps({"type": "video", "data": b64}))
                logger.info(f"[Video] {frame_ts:.1f}s 프레임 전송 ({len(jpeg_bytes)}B)")
                video_frame_ptr += 1

            # 15초마다 END_OF_SPEECH 전송 (15, 30, 45, 60초)
            if current_sec >= next_eos_sec:
                await ws.send(json.dumps({"type": "control", "data": "END_OF_SPEECH"}))
                logger.info(f"[Control] {current_sec:.1f}s → END_OF_SPEECH 전송 (구간 {next_eos_sec:.0f}s)")
                try:
                    resp = await recv_until(ws, "processing", timeout=5.0)
                    logger.info(f"[서버] {resp}")
                    stt_resp = await recv_until(ws, "stt_done", timeout=120.0)
                    logger.info(f"[STT 결과] {stt_resp}")
                    llm_resp = await recv_until(ws, "response", timeout=10.0)
                    logger.info(f"[LLM 응답] {llm_resp}")
                except asyncio.TimeoutError:
                    logger.warning(f"[Control] {next_eos_sec:.0f}s 구간 응답 타임아웃")
                next_eos_sec += END_OF_SPEECH_INTERVAL  # 다음 구간으로

            # 오디오 청크 전송 (fire and forget - Ack 기다리지 않음)
            b64 = base64.b64encode(chunk_bytes).decode()
            await ws.send(json.dumps({"type": "audio", "data": b64}))

            # 실시간 흉내 (32ms마다 청크 전송)
            await asyncio.sleep(AUDIO_CHUNK_SEC)

        # 영상 종료 후 마지막 구간이 남아있으면 전송
        if next_eos_sec - END_OF_SPEECH_INTERVAL < total_sec:
            await ws.send(json.dumps({"type": "control", "data": "END_OF_SPEECH"}))
            logger.info("[Control] END_OF_SPEECH 전송 (영상 종료)")
            try:
                resp = await recv_until(ws, "processing", timeout=5.0)
                logger.info(f"[서버] {resp}")
                stt_resp = await recv_until(ws, "stt_done", timeout=120.0)
                logger.info(f"[STT 결과] {stt_resp}")
                llm_resp = await recv_until(ws, "response", timeout=10.0)
                logger.info(f"[LLM 응답] {llm_resp}")
            except asyncio.TimeoutError:
                logger.warning("[Control] 응답 타임아웃")

        logger.info("테스트 완료")


if __name__ == "__main__":
    asyncio.run(run())
