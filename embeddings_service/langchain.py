import logging

import aiohttp
import requests
from langchain_core.embeddings import Embeddings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import BACKOFF_FACTOR, DEFAULT_BATCH_SIZE, MAX_RETRIES, STATUS_FORCELIST, TIMEOUT
from .schemas import EmbeddingRequest, EmbeddingResponse, HealthCheck

logger = logging.getLogger("RemoteHTTPEmbeddings")


class RemoteHTTPEmbeddings(Embeddings):
    """
    Совместимый с LangChain клиент для взаимодействия с развёрнутой моделью ембедингов
    на удалённом HTTP сервере.

    Класс предоставляет синхронный и асинхронный интерфейс для векторизации текста.
    Он реализует стандартный функционал LangChain Embeddings
    с логикой healthcheck для проверки работоспособности сервера.

    Attributes:
        base_url (str): Базовый URL сервера, пример: http://localhost:8000
        normalize_embeddings (bool): Нужно ли нормализовать векторы на выходе.
        batch_size (int): Количество текста обрабатываемого за один запрос.
        timeout (int): Таймаут для ожидания ответа от сервера.
        max_retries (int): Максимальное количество попыток для healthcheck.

    Examples:
        >>> from embeddings_service.langchain import RemoteHTTPEmbeddings
        >>>
        >>> # Синхронный вариант
        >>> embeddings = RemoteHTTPEmbeddings(base_url="http://localhost:8000")
        >>> if embeddings.wait_for_healthy():
        ...     vector = embeddings.embed_query("Hello world")
        ...     vectors = embeddings.embed_documents(["Text 1", "Text 2"])
        >>>
        >>> # Асинхронный вариант
        >>> async def main():
        ...     embeddings = RemoteHTTPEmbeddings(base_url="http://localhost:8000")
        ...     vector = await embeddings.aembed_query("Hello world")
        ...     vectors = await embeddings.aembed_documents(["Text 1", "Text 2"])
    """
    def __init__(
            self,
            base_url: str,
            normalize_embeddings: bool = True,
            batch_size: int = DEFAULT_BATCH_SIZE,
            timeout: int = TIMEOUT,
            max_retries: int = MAX_RETRIES,
            backoff_factor: float = BACKOFF_FACTOR,
    ) -> None:
        self.base_url = base_url
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST
        )

    def wait_for_healthy(self) -> bool:
        """Ожидает и проверяет доступность сервера.

        :return True если сервер готов к работе,
        False предыдущие попытки были неудачны.
        """
        url = f"{self.base_url}/health"
        try:
            with requests.Session() as session:
                session.mount(url, HTTPAdapter(max_retries=self.retries))
                response = session.get(url=url, timeout=self.timeout)
                data = response.json()
            healthcheck = HealthCheck.model_validate(data)
            if healthcheck.status != "healthy":
                logger.info(
                    "Server not healthy! Status: %s, model_status: %s",
                    healthcheck.status, healthcheck.model_status
                )
                return False
        except TimeoutError:
            logger.exception("Service still not healthy! Error: {e}")
            return False
        else:
            logger.info("Server healthy!", extra=healthcheck.model_dump())
            return True

    def _vectorize(self, texts: list[str]) -> list[list[float]]:
        with requests.Session() as session:
            response = session.post(
                url=f"{self.base_url}/api/v1/embeddings/vectorize",
                headers={"Content-Type": "application/json"},
                json=EmbeddingRequest(
                    texts=texts,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=self.batch_size
                ).model_dump(),
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        embeddings_response = EmbeddingResponse.model_validate(data)
        return embeddings_response.embeddings

    async def _avectorize(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(
            url=f"{self.base_url}/api/v1/embeddings/vectorize",
            headers={"Content-Type": "application/json"},
                json=EmbeddingRequest(
                    texts=texts,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=self.batch_size
                ).model_dump(),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        embeddings_response = EmbeddingResponse.model_validate(data)
        return embeddings_response.embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectorize(texts)

    async def aembed_query(self, text: str) -> list[float]:
        embeddings = await self._avectorize([text])
        return embeddings[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._avectorize(texts)
