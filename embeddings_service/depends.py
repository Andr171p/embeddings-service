import logging
from functools import cache

import torch
from sentence_transformers import SentenceTransformer

from .constants import MODEL_NAME

logger = logging.getLogger(__name__)


@cache
def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    return device


@cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=get_device())
