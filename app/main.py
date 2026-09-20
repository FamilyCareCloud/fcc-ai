import logging
import secrets
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


@app.get("/health")
def health():
    return {"status": "ok", "device": transcriber.device}


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """FCC_AI_API_KEY를 설정한 경우에만 검사한다.

    공인 터널(ngrok 등)로 노출할 때 GPU 엔드포인트가 열리는 것을 막기 위한 공유 비밀이다.
    백엔드(fcc-backend)는 STT_SERVICE_API_KEY를 같은 헤더로 보낸다. 로컬 개발처럼
    키를 비워두면 검사하지 않는다.
    """
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="유효한 API 키가 필요합니다.")


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
