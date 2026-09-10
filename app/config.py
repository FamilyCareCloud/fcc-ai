from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_model: str = "openai/whisper-small"
    adapter_path: str = "models/whisper-senior-lora"
    device: str = "cuda"
    language: str = "korean"
    task: str = "transcribe"
    host: str = "0.0.0.0"
    port: int = 8000
    # 공인 터널(Cloudflare Tunnel/ngrok 등)로 노출할 때 설정. 비워두면 인증 없이 허용(로컬 전용 실행 가정).
    api_key: str = ""

    class Config:
        env_prefix = "FCC_AI_"
        env_file = ".env"


settings = Settings()
