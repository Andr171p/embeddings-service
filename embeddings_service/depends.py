from typing import Final

import logging
from functools import cache

import torch
from sentence_transformers import SentenceTransformer

from .constants import MODEL_NAME
from .schemas import Settings

logger = logging.getLogger(__name__)

settings: Final[Settings] = Settings()


@cache
def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    return device


@cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.model_name, device=get_device())
