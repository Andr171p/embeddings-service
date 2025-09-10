from pydantic import BaseModel, Field, computed_field, field_validator

from .constants import DEFAULT_BATCH_SIZE, MIN_BATCH_SIZE, MODEL_NAME


class HealthCheck(BaseModel):
    status: str = "healthy"
    model: str = MODEL_NAME
    device: str = "cpu"


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
    embeddings: list[float]

    @computed_field
    def dimensions(self) -> int:
        return len(self.embeddings)
