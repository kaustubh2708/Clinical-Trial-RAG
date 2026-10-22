"""Application configuration, loaded from environment variables or a `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "clinical_protocols"
    UPLOAD_DIR: str = "./data/raw_protocols"
    MAX_UPLOAD_MB: int = Field(default=20, ge=1, le=200)

    # Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "openai" needs OPENAI_API_KEY; "ollama" runs a local model (e.g. LLM_MODEL_NAME=qwen2.5:7b) and needs no key.
    LLM_PROVIDER: Literal["openai", "ollama"] = "openai"
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_SEED: int = 42
    LLM_TIMEOUT_SECONDS: float = 60.0
    # Hard cap on generated tokens: a runaway generation fails fast instead of holding a worker until timeout.
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=256)
    OPENAI_API_KEY: SecretStr | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Set explicitly: Ollama truncates prompts longer than its context window, silently losing part of them.
    OLLAMA_NUM_CTX: int = Field(default=8192, ge=2048)

    # Chunking
    CHUNK_SIZE: int = Field(default=700, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)

    # Retrieval
    RETRIEVAL_TOP_K: int = Field(default=4, ge=1, le=10)
    # "hybrid" = dense + BM25 fused with Reciprocal Rank Fusion; "dense" = embeddings only.
    RETRIEVAL_MODE: Literal["hybrid", "dense"] = "hybrid"
    HYBRID_CANDIDATES: int = Field(default=20, ge=1, le=200)  # candidates per ranker before fusion
    RRF_K: int = Field(default=60, ge=1)
    # Relevance gate (cosine relevance = 1 - cosine distance). If the best dense match scores below this,
    # the service abstains without calling the LLM. In dense mode, weaker chunks are also dropped.
    MIN_RELEVANCE_SCORE: float = Field(default=0.30, ge=-1.0, le=1.0)
    # Upper bound on criteria sent to the LLM in one screening call.
    MAX_SCREENING_CRITERIA: int = Field(default=80, ge=1)

    LOG_LEVEL: str = "INFO"

    def ensure_directories(self) -> None:
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
