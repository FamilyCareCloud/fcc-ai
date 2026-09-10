import io
import logging

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from app.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Transcriber:
    """노인 명령어 음성에 파인튜닝된 Whisper-small LoRA 어댑터를 얹은 STT 래퍼.

    GPU가 없으면 CPU로 대체 로드하되, 응답 속도가 크게 느려질 수 있습니다.
    """

    def __init__(self) -> None:
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None
        self._device = settings.device if torch.cuda.is_available() else "cpu"
        if settings.device == "cuda" and self._device == "cpu":
            logger.warning("CUDA를 사용할 수 없어 CPU로 대체합니다. 응답이 느릴 수 있습니다.")

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("모델 로드 중: base=%s adapter=%s device=%s", settings.base_model, settings.adapter_path, self._device)
        processor = WhisperProcessor.from_pretrained(settings.adapter_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(settings.base_model)
        model = PeftModel.from_pretrained(base_model, settings.adapter_path)
        model = model.to(self._device).eval()

        self._processor = processor
        self._model = model
        logger.info("모델 로드 완료")

    @property
    def device(self) -> str:
        return self._device

    def transcribe(self, audio_bytes: bytes) -> str:
        if self._model is None or self._processor is None:
            self.load()

        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        inputs = self._processor.feature_extractor(
            audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs, language=settings.language, task=settings.task
            )
        text = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


transcriber = Transcriber()
