"""Process-wide singleton for the local sentence-transformers embedding model."""

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings() -> Embeddings:
    """Load the embedding model once. Runs on CPU; vectors are L2-normalised so cosine == dot product."""
    settings = get_settings()
    logger.info("Loading embedding model %s", settings.EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
