import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

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


@app.post("/transcribe")
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
