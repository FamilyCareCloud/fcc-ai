# fcc-ai

Family Care Cloud는 가족이 함께 돌봄 기록을 남기고 다음 보호자에게 전달하는 서비스입니다.

이 저장소는 **GPU가 필요한 AI 기능만** 담당합니다. 현재는 노인 명령어 음성 인식(STT)만 구현되어 있으며, 백엔드(fcc-backend)에서 이 서버의 `/transcribe` API를 호출해 사용하는 구조를 전제로 합니다. 프론트/백엔드는 별도 저장소에서 운영되며, 이 서버는 로컬(GPU) 환경에서 단독으로 실행합니다.

## 현재 상태

- **음성 인식 (STT)**: `openai/whisper-small` + LoRA 파인튜닝 어댑터 (AI Hub "명령어 음성(노인남녀)" 데이터 12,000건 기반, held-out WER 14.54%)
- Care Handoff 요약 등 LLM 기반 기능은 GPU가 필수는 아니라 이 저장소 범위에서 제외했습니다 (백엔드에서 별도 LLM API로 처리 예정).

## 실행 방법

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU가 Blackwell(RTX 50 시리즈) 등 최신 세대라면 PyTorch를 cu128 빌드로 별도 설치해야 합니다:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. 모델 배치

`models/whisper-senior-lora/` 폴더에 파인튜닝된 LoRA 어댑터 파일(`adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json` 등)을 넣습니다. 용량 문제로 git에는 포함하지 않으며(.gitignore 처리됨), 담당자에게 직접 전달받아 배치합니다.

### 3. 서버 실행

```bash
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. API

- `GET /health` — 서버 및 device(cuda/cpu) 상태 확인
- `POST /transcribe` — multipart/form-data로 오디오 파일(`file`)을 보내면 `{"text": "..."}` 반환

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@sample.wav"
```

## 관련 저장소

- 프론트: https://github.com/FamilyCareCloud/fcc-frontend
- 백엔드: https://github.com/FamilyCareCloud/fcc-backend
- AI: https://github.com/FamilyCareCloud/fcc-ai

## 협업

main은 확인된 결과, develop은 개발 통합 브랜치입니다. develop에서 작업 브랜치를 만들고 PR로 변경을 검토합니다. 요청·응답 형식과 환경 설정은 관련 담당자와 먼저 맞춥니다.
