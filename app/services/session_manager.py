from fastapi import WebSocket
from typing import Dict
from app.schemas import InputTest, ServerResponse
import json
import base64
import random
import os

# 접속자를 관리하고 데이터를 전달하는 역할, 비동기 처리

class ConnectionManager:
    def __init__(self):
        # 활성화된 상담 세션들을 저장하는 장부
        self.active_connections: Dict[str, WebSocket] = {}

        # 유저별 오디오 버퍼(bytearray, 바이트 배열)
        self.audio_buffers: Dict[str, bytearray] = {}

    # [초기 상담 생성] 웹소캣 연결 수락
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = bytearray()
        print(f"--- [Session] Client {client_id} 연결(현재 접속자: {len(self.active_connections)})명")

        # 연결 성공 메시지 전송
        await self.send_personal_message(
            ServerResponse(status =  "connected", message = "상담실에 입장하였습니다.").model_dump(),
            client_id
        )

    # 연결 해제 처리
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"--- [Session] Client {client_id} 연결 해제")

    
    # 특정 사용자에게 JSON 변환 메시지 전송
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False)) # 파이썬 딕셔너리를 문자열로 변환


    # [데이터 처리 파이프라인] 들어온 데이터 분류 및 처리
    async def process_data(self, client_id: str, raw_data: str):
        try:
            # JSON 데이터 파싱
            data_dict = json.loads(raw_data)
            input_obj = InputTest(**data_dict)

            # 타입별 처리 분기

            # [음성 데이터 처리]
            if input_obj.type == "audio":
                print(f"[Audio] {client_id}의 음성 데이터 수신 중 ... 데이터 내용: {input_obj.data}")

                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split[1]

                    audio_chunk = base64.b64decode(base64_data)

                    self.audio_buffers[client_id].extend(audio_chunk)

                except Exception as e:
                    print(f"[Error] 오디오 처리 실해: {e}")
                # VAD 등 음성 처리 로직 추가



            # [이미지 데이터 처리]
            elif input_obj.type == "video":
                # 표정기반 감정추출 로직 추가 필요
                print(f"[Video] {client_id}의 얼굴 데이터 수신 중 ...")

                try:
                    # 1. 클라이언트가 보낸 데이터 꺼내기
                    base64_data = str(input_obj.data)

                    # 2. data:image/jpeg:base64, 부분을 자르고 뒤쪽만 가져오기
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]

                    # 3. Base64 텍스트를 이미지 바이너리로 해독
                    image_bytes = base64.b64decode(base64_data)

                    # 4. 해독한 이미지를 저장(확인을 위한 검증용, 추후 제거 예정)
                    file_name = f"test_file/face_{client_id}{random.randint(1, 10000)}.jpg"
                    with open(file_name, "wb") as f:
                        f.write(image_bytes)

                    print(f"이미지 변환 완료, 프로젝트 폴더에 {file_name} 저장")

                except Exception as e:
                    print(f"[Error] 이미지 변환 실패 {str(e)}")
                


            # [발화 신호 처리]
            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH":
                    print(f"[Control] {client_id}의 발화 종료, 처리 시작 ...")

                    if len(self.audio_buffers[client_id]) > 0: # 오디오가 존재한다면
                        file_name = f"test_file/voice.{client_id}{random.randint(1, 10000)}.webm"

                        # [테스트용]
                        with open(file_name, "wb") as f:
                            f.write(self.audio_buffers[client_id])

                        print(f"오디오 변환 완료, 프로젝트 폴더에 {file_name} 저장")

                        self.audio_buffers[client_id].clear()
                    # 말하기 종료 및 STT, LLM 로직 추가
                    
                    await self.send_personal_message(
                        {"status": "processing", "message": "답변 생성중"}
                        , client_id
                    )
            
            else:
                print(f"[Error] 알 수 없는 신호: {input_obj.data}")
            
            # 응답(Ack) 전송
            response = ServerResponse(
                status = "received",
                message=f"{input_obj.type} 데이터 수신 완료"
            )
            await self.send_personal_message(response.model_dump(), client_id)
        
        except json.JSONDecodeError:
            print(f"[Error] {client_id}가 올바르지 않은 데이터 전송: {raw_data}")
        except Exception as e:
            print(f"[Error] 처리 중 문제 발생: {str(e)}")

# 전역 매니저 인스턴스 생성
manager = ConnectionManager()