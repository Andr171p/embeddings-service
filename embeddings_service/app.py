from typing import Annotated

import logging
import time

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from .depends import get_device, get_model, settings
from .schemas import EmbeddingRequest, EmbeddingResponse, HealthCheck

START_TIME = time.time()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Сервис ембеддингов",
    description="Предоставляет HTTP методы для векторизации текста",
    version="1.0.0",
)


def is_model_ready(model: SentenceTransformer) -> bool:
    test_text = "healthcheck"
    test_embeddings = model.encode(
        [test_text], convert_to_tensor=False, convert_to_numpy=True
    )
    if test_embeddings is None or len(test_embeddings) == 0:
        logger.error("Model encoding test failed")
        return False
    logger.info("Model encoding test succeeded!")
    return True


@app.get(
    path="/health",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    summary="Проверяет доступность и работоспособность сервера"
)
def healthcheck(
        model: Annotated[SentenceTransformer, Depends(get_model)],
        device: Annotated[str, Depends(get_device)],
) -> HealthCheck:
    if not is_model_ready(model):
        return HealthCheck(
            status="failed",
            model=settings.model_name,
            device=device,
            model_status="MODEL_TEST_FAILED"
        )
    return HealthCheck(
        status="healthy",
        model=settings.model_name,
        device=device,
        model_status="WORKING_AND_LOADING",
        uptime=time.time() - START_TIME,
    )


@app.post(
    path="/api/v1/embeddings/vectorize",
    status_code=status.HTTP_200_OK,
    response_model=EmbeddingResponse,
    summary="Векторизует текст"
)
def vectorize(
        request: EmbeddingRequest,
        model: Annotated[SentenceTransformer, Depends(get_model)]
) -> EmbeddingResponse:
    embeddings = model.encode(
        request.texts,
        batch_size=request.batch_size,
        normalize_embeddings=request.normalize,
        convert_to_tensor=False,
        convert_to_numpy=True
    ).tolist()
    return EmbeddingResponse(embeddings=embeddings)


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
