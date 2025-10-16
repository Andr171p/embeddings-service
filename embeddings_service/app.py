from typing import Annotated, ParamSpec, TypeVar

import logging
import time
from collections.abc import Callable
from functools import wraps

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sentence_transformers import SentenceTransformer

from .depends import get_device, get_hostname, get_model
from .schemas import EmbeddingRequest, EmbeddingResponse, HealthCheck
from .settings import settings

T = TypeVar("T")  # Тип возвращаемого значения
P = ParamSpec("P")  # Параметры функции


START_TIME = time.time()

logger = logging.getLogger(__name__)


def timer(func: Callable[P, T]) -> Callable[P, T]:
    """Замер времени выполнения функции"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            execution_time = time.time() - start_time
            logger.info(
                "%s execution time: %s seconds",
                func.__name__, round(execution_time, 2)
            )
    return wrapper


app = FastAPI(
    title="API Сервис ембеддингов",
    description="Предоставляет HTTP методы для векторизации текста",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)


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
        hostname: Annotated[str, Depends(get_hostname)],
) -> HealthCheck:
    payload: dict[str, str | int] = {
        "instance_id": settings.instance_id,
        "hostname": hostname,
        "model": settings.model_name,
        "device": device,
    }
    if not is_model_ready(model):
        payload.update({"status": "failed", "model_status": "MODEL_TEST_FAILED"})
    else:
        payload.update({
            "status": "healthy",
            "model_status": "WORKING_AND_LOADING",
            "uptime": int(time.time() - START_TIME),
        })
    return HealthCheck.model_validate(payload)


@app.post(
    path="/api/v1/embeddings/vectorize",
    status_code=status.HTTP_200_OK,
    response_model=EmbeddingResponse,
    summary="Векторизует текст"
)
@timer
def vectorize(
        request: EmbeddingRequest,
        model: Annotated[SentenceTransformer, Depends(get_model)]
) -> EmbeddingResponse:
    embeddings = model.encode(
        request.texts,
        batch_size=request.batch_size,
        normalize_embeddings=request.normalize,
        convert_to_tensor=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    return EmbeddingResponse(embeddings=embeddings)


@app.exception_handler(ValueError)
def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
