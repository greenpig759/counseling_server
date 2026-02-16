from pydantic import BaseModel # 자동 형변환, 테이터 형식 검사 역할
from typing import Optional, Any

# WebSocket에서는 수동으로 매핑 -> session_manager.py

# -- [테스트용] 입력 데이터 확인 객체 --
class InputTest(BaseModel):
    type: str
    data: Any
    session_id: str
    timestamp: Optional[float] = None

# -- [응답용] 잘 받았다고 서버가 보내줄 객체 --
class ServerResponse(BaseModel):
    status: str
    message: str
    next_action: Optional[str] = None