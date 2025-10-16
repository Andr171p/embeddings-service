from typing import Final

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# Модель для векторизации
MODEL_NAME = "deepvk/USER-bge-m3"
# Валидация входных данных
MIN_BATCH_SIZE = 1
DEFAULT_BATCH_SIZE = 64
TIMEOUT = 30
MAX_RETRIES = 7
BACKOFF_FACTOR = 0.1
STATUS_FORCELIST: list[int] = [429, 500, 502, 503, 504]


class Settings(BaseSettings):
    model_name: str = MODEL_NAME
    instance_number: int = 1

    @property
    def instance_id(self) -> str:
        return f"embeddings-service-{self.instance_number}"


settings: Final[Settings] = Settings()
