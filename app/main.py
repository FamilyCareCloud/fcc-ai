import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from app.config import settings
from app.stt import transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    transcriber.load()
    yield


app = FastAPI(title="fcc-ai", description="Family Care Cloud - GPU 음성 인식(STT) 서버", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # settings.api_key가 비어있으면(로컬 단독 실행 가정) 검증을 건너뜁니다.
    # 공인 터널로 노출할 때는 반드시 설정하세요 - 그렇지 않으면 누구나 GPU를 호출할 수 있습니다.
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


@app.get("/health")
def health():
    return {"status": "ok", "device": transcriber.device}


@app.post("/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("예상치 못한 content-type: %s (계속 진행)", file.content_type)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        text = transcriber.transcribe(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("음성 인식 실패")
        raise HTTPException(status_code=500, detail=f"음성 인식 실패: {exc}") from exc

    return {"text": text}
