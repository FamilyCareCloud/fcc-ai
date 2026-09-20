from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    base_model: str = "openai/whisper-small"
    api_key: str = ""
    adapter_path: str = "models/whisper-senior-lora"
    device: str = "cuda"
    language: str = "korean"
    task: str = "transcribe"
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = "FCC_AI_"
        env_file = ".env"


settings = Settings()
