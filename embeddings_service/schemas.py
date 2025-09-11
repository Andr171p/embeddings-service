from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_settings import BaseSettings

from .constants import DEFAULT_BATCH_SIZE, ENV_FILE, MIN_BATCH_SIZE, MODEL_NAME

load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    model_name: str = MODEL_NAME
    instance_number: int = 1

    @property
    def instance_id(self) -> str:
        return f"embeddings-service-{self.instance_number}"


class HealthCheck(BaseModel):
    instance_id: str
    hostname: str
    status: Literal["healthy", "failed"] = "healthy"
    model: str
    device: str = "cpu"
    uptime: int | None = None


class EmbeddingRequest(BaseModel):
    texts: list[str]
    normalize: bool = True
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=MIN_BATCH_SIZE)

    @field_validator("texts")
    def validate_texts(cls, texts: list[str]) -> list[str]:
        for text in texts:
            if len(text) == 0:
                raise ValueError("Empty text")
        return texts


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]

    @computed_field
    def dimensions(self) -> int:
        return len(self.embeddings)
