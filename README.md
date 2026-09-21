# fcc-ai - Family Care Cloud AI 서버

노인 명령어 음성 인식(STT) GPU 마이크로서비스

## 프로젝트 개요

Family Care Cloud는 가족이 함께 돌봄 기록을 남기고 다음 보호자에게 전달하는 서비스입니다.
이 저장소는 그중 **GPU가 필요한 AI 기능만** 담당합니다. 현재는 고령자 음성 명령을 텍스트로 바꾸는 음성 인식(STT)이 구현되어 있으며,
`fcc-backend`가 이 서버의 `/transcribe` API를 호출해 음성 질의 기능(`/groups/{g}/assistant/voice`)에 사용합니다.

- 프론트/백엔드는 별도 저장소에서 운영되며, 이 서버는 GPU가 있는 로컬 환경에서 단독으로 실행합니다.
- Care Handoff 요약 등 LLM 기반 기능은 GPU가 필수가 아니므로 이 저장소 범위에서 제외했습니다 (백엔드에서 별도 LLM API로 처리).

## 팀 정보

- **팀명**: fcc (FamilyCareCloud)
- **AI 레포 담당**: 1인 개발
- **개발 기간**: 2026년 9월 7일 ~

## 기술 스택

| 분류 | 기술 |
|------|------|
| 음성 인식 모델 | `openai/whisper-small` (244M) + LoRA 어댑터 |
| 파인튜닝 | transformers, PEFT (LoRA r=16), PyTorch (cu128) |
| 학습 데이터 | AI Hub "명령어 음성(노인남녀)" |
| AI 서비스 | FastAPI, uvicorn |
| 오디오 처리 | librosa, soundfile |
| 외부 노출 | Cloudflare Tunnel + `X-API-Key` 인증 |

## 레포지토리 구조

```
fcc-ai/
├── README.md
├── requirements.txt
├── .env.example                   # 환경 변수 예시
├── .gitignore
│
├── app/
│   ├── main.py                    # FastAPI 진입점 (/health, /transcribe, API 키 검증)
│   ├── config.py                  # 환경 설정 (FCC_AI_* 환경 변수)
│   └── stt.py                     # Whisper + LoRA 로딩 및 추론 (GPU/CPU 자동 선택)
│
└── models/                        # gitignore 처리 - 로컬에 직접 배치
    └── whisper-senior-lora/       # 파인튜닝된 LoRA 어댑터
```

## AI 서비스 API

| 엔드포인트 | 설명 | 인증 | 상태 |
|---|---|---|---|
| `GET /health` | 서버 및 device(cuda/cpu) 상태 확인 | 불필요 | 구현 완료 |
| `POST /transcribe` | 오디오 파일 → 한국어 텍스트 | `X-API-Key` (설정 시) | 구현 완료 |

### POST /transcribe

`multipart/form-data`로 오디오 파일을 `file` 필드에 담아 보냅니다. 서버가 16kHz mono로 변환해 추론합니다.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.wav" \
  -H "X-API-Key: $FCC_AI_API_KEY"
```

**Response Body**

```json
{ "text": "알람 그만 울려 줘." }
```

| 상태 코드 | 의미 |
|---|---|
| 200 | 성공 |
| 400 | 빈 파일 |
| 401 | `X-API-Key` 불일치 (`FCC_AI_API_KEY`가 설정된 경우) |
| 500 | 음성 인식 실패 |

지원 형식: wav, flac, ogg, webm 등 (librosa가 읽을 수 있는 형식). 그 외 content-type은 경고만 남기고 처리합니다.

### GET /health

```json
{ "status": "ok", "device": "cuda" }
```

---

## 모델 학습 결과

### 학습 데이터

| 항목 | 값 |
|---|---|
| 원본 | AI Hub "명령어 음성(노인남녀)" — AI비서 / AI로봇 / 키오스크 / 비정형 각 4,000건, 총 16,000건 |
| 무음·손상 파일 제거 | 184건 제거 (RMS 임계값 0.0005) → 15,816건 |
| **최종 학습 데이터** | **11,996건** (비정형 카테고리 제외) |
| Train / Eval | 약 10,796 / 1,200건 (9:1) |

비정형(자유 발화) 카테고리는 다른 카테고리(WER 12~13%)보다 오류율이 훨씬 높았고(약 68%),
일부 구간에서 무음·손상 파일이 집중되어 있어 최종 모델에서는 제외했습니다.

### 학습 설정

| 항목 | 값 |
|---|---|
| 베이스 모델 | `openai/whisper-small` |
| LoRA | r=16, alpha=32, dropout=0.05, target: `q_proj`, `v_proj` |
| Epoch / Batch | 5 epoch, batch 8 × gradient accumulation 2 (effective 16) |
| Learning rate | 1e-4 (linear scheduler), fp16 |
| 학습 환경 | NVIDIA RTX 5070 Ti (16GB), PyTorch 2.11.0+cu128, PEFT 0.19.1 |

### 성능 (WER, 낮을수록 좋음)

| 모델 | WER |
|---|---|
| 파인튜닝 전 (base whisper-small, zero-shot) | 49.30% |
| **파인튜닝 후 (최종, 5 epoch)** | **14.54%** |

Epoch별 검증 WER: 20.86% → 17.23% → 15.40% → 14.71% → **14.54%**

> 참고: 파인튜닝 전 수치는 별도 400건 평가셋 기준이고, 파인튜닝 후 수치는 학습에 사용하지 않은 검증셋(약 1,200건, 비정형 제외) 기준입니다.
> 평가셋 구성이 완전히 같지 않으므로 절대 비교보다는 개선 경향으로 참고해 주세요.

### 추론 속도 (샘플 1건, 약 2.4초 오디오)

| 환경 | 추론 시간 |
|---|---|
| GPU (RTX 5070 Ti) | 약 0.13초 |
| CPU | 약 1.1초 |

GPU가 없으면 자동으로 CPU로 대체됩니다. 학습에는 GPU가 사실상 필수이지만, 추론은 CPU로도 실사용이 가능합니다.

---

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

`models/whisper-senior-lora/` 폴더에 파인튜닝된 LoRA 어댑터 파일(`adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json` 등)을 넣습니다.
용량 문제로 git에는 포함하지 않으며(.gitignore 처리됨), 담당자에게 직접 전달받아 배치합니다.

### 3. 서버 실행

```bash
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `FCC_AI_BASE_MODEL` | `openai/whisper-small` | 베이스 모델 |
| `FCC_AI_ADAPTER_PATH` | `models/whisper-senior-lora` | LoRA 어댑터 경로 |
| `FCC_AI_DEVICE` | `cuda` | `cuda` 또는 `cpu` (GPU 없으면 자동 CPU) |
| `FCC_AI_LANGUAGE` / `FCC_AI_TASK` | `korean` / `transcribe` | Whisper 디코딩 옵션 |
| `FCC_AI_HOST` / `FCC_AI_PORT` | `0.0.0.0` / `8000` | 서버 주소 |
| `FCC_AI_API_KEY` | (비어 있음) | 설정 시 `/transcribe`에 `X-API-Key` 헤더 필수 |

## 배포된 백엔드(AWS Lambda)와 연결하기

이 서버는 GPU가 필요해 로컬에서만 실행합니다. AWS의 fcc-backend가 호출하려면 공인 HTTPS 주소로 터널링해야 합니다.
**`FCC_AI_API_KEY`를 반드시 설정**하세요 — 그렇지 않으면 터널 주소를 아는 누구나 이 GPU를 호출할 수 있습니다.

```bash
# .env에 FCC_AI_API_KEY=<임의의 긴 비밀 문자열> 설정 후 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 다른 터미널에서 Cloudflare Tunnel로 공인 URL 발급 (설치: winget install --id Cloudflare.cloudflared)
cloudflared tunnel --url http://localhost:8000
```

출력된 `https://xxxx.trycloudflare.com` 형태의 URL을 fcc-backend 저장소의 GitHub 저장소 변수/시크릿에 등록하면(Settings → Secrets and variables → Actions) 다음 배포부터 반영됩니다:

- Variables: `STT_SERVICE_URL` = 위 터널 URL
- Secrets: `STT_SERVICE_API_KEY` = 위에서 설정한 `FCC_AI_API_KEY`와 동일한 값

Quick Tunnel은 서버를 재시작할 때마다 URL이 바뀝니다. 고정 주소가 필요하면 Cloudflare 계정에 도메인을 연결한 Named Tunnel을 사용하세요.

## 관련 저장소

- 프론트: https://github.com/FamilyCareCloud/fcc-frontend
- 백엔드: https://github.com/FamilyCareCloud/fcc-backend
- AI: https://github.com/FamilyCareCloud/fcc-ai

## 협업

main은 확인된 결과, develop은 개발 통합 브랜치입니다. develop에서 작업 브랜치를 만들고 PR로 변경을 검토합니다. 요청·응답 형식과 환경 설정은 관련 담당자와 먼저 맞춥니다.
