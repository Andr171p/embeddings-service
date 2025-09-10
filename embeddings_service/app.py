from typing import Annotated

from fastapi import FastAPI, Depends, Request, status
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from .depends import get_embeddings
from .schemas import HealthCheck, EmbeddingRequest, EmbeddingResponse

app = FastAPI(
    title="API Сервис ембеддингов",
    description="Предоставляет HTTP методы для векторизации текста",
    version="1.0.0",
)


@app.get(
    path="/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    summary="Проверяет доступность сервера"
)
def health_check() -> HealthCheck:
    return HealthCheck()


@app.post(
    path="/api/v1/embeddings/vectorize",
    status_code=status.HTTP_200_OK,
    response_model=EmbeddingResponse,
    summary="Векторизует текст"
)
def vectorize(
        request: EmbeddingRequest,
        embeddings: Annotated[SentenceTransformer, Depends(get_embeddings)]
) -> EmbeddingResponse:
    vectors = embeddings.encode(
        request.texts,
        batch_size=request.batch_size,
        normalize_embeddings=request.normalize,
        convert_to_tensor=False,
        convert_to_numpy=True
    ).tolist()
    return EmbeddingResponse(embeddings=vectors)


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
