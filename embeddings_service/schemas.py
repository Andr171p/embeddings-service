from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from .settings import DEFAULT_BATCH_SIZE, MIN_BATCH_SIZE


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
            if len(text.strip()) == 0:
                raise ValueError("Empty text")
        return texts


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]

    @computed_field
    def dimensions(self) -> int:
        return len(self.embeddings[0])
