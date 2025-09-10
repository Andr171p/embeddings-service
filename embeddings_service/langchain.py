import aiohttp
import requests
from langchain_core.embeddings import Embeddings

from .constants import DEFAULT_BATCH_SIZE


class RemoteEmbeddings(Embeddings):
    def __init__(
            self, base_url: str,
            normalize_embeddings: bool = True,
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        self.base_url = base_url
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            url=f"{self.base_url}/embeddings/vectorize",
            headers={"Content-Type": "application/json"},
            json={
                "texts": [text],
                "normalized": self.normalize_embeddings,
                "batch_size": self.batch_size
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        response = requests.post(
            url=f"{self.base_url}/embeddings/vectorize",
            headers={"Content-Type": "application/json"},
            json={
                "texts": texts,
                "normalized": self.normalize_embeddings,
                "batch_size": self.batch_size
            }
        )
        response.raise_for_status()
        data = response.json()
        vectors.append(data["embeddings"])
        return vectors
