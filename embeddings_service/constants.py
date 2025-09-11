from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# Модель для векторизации
MODEL_NAME = "deepvk/USER-bge-m3"
# Валидация входных данных
MIN_BATCH_SIZE = 1
DEFAULT_BATCH_SIZE = 32
TIMEOUT = 30
MAX_RETRIES = 7
BACKOFF_FACTOR = 0.1
STATUS_FORCELIST: list[int] = [429, 500, 502, 503, 504]
