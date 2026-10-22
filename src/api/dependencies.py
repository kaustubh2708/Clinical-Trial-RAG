"""FastAPI dependency providers. Tests replace these via `app.dependency_overrides`."""

from functools import lru_cache

from fastapi import Depends
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from src.config import Settings, get_settings
from src.ingestion.indexer import ProtocolIndexer, build_vector_store
from src.rag.chain import EligibilityRAGChain
from src.rag.embeddings import get_embeddings


@lru_cache
def _cached_vector_store(persist_directory: str, collection_name: str) -> Chroma:
    return build_vector_store(get_embeddings(), persist_directory, collection_name)


def get_vector_store(settings: Settings = Depends(get_settings)) -> Chroma:
    return _cached_vector_store(settings.CHROMA_PERSIST_DIR, settings.CHROMA_COLLECTION_NAME)


@lru_cache
def _cached_llm(model: str, temperature: float, seed: int, timeout: float, max_tokens: int,
                api_key: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI  # imported lazily so the service starts without the OpenAI stack

    return ChatOpenAI(model=model, temperature=temperature, seed=seed, timeout=timeout, max_retries=2,
                      max_completion_tokens=max_tokens, api_key=SecretStr(api_key))


@lru_cache
def _cached_ollama(model: str, temperature: float, seed: int, timeout: float, max_tokens: int, base_url: str,
                   num_ctx: int) -> BaseChatModel:
    from langchain_ollama import ChatOllama  # imported lazily, like ChatOpenAI

    return ChatOllama(model=model, temperature=temperature, seed=seed, base_url=base_url, num_ctx=num_ctx,
                      num_predict=max_tokens, client_kwargs={"timeout": timeout})


def get_llm(settings: Settings = Depends(get_settings)) -> BaseChatModel | None:
    """The only place the LLM provider is chosen.

    Returns None for OpenAI without a key, so indexing and abstentions still work without an LLM.
    """
    if settings.LLM_PROVIDER == "ollama":
        return _cached_ollama(
            settings.LLM_MODEL_NAME,
            settings.LLM_TEMPERATURE,
            settings.LLM_SEED,
            settings.LLM_TIMEOUT_SECONDS,
            settings.LLM_MAX_OUTPUT_TOKENS,
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_NUM_CTX,
        )
    if settings.OPENAI_API_KEY is None or not settings.OPENAI_API_KEY.get_secret_value():
        return None
    return _cached_llm(
        settings.LLM_MODEL_NAME,
        settings.LLM_TEMPERATURE,
        settings.LLM_SEED,
        settings.LLM_TIMEOUT_SECONDS,
        settings.LLM_MAX_OUTPUT_TOKENS,
        settings.OPENAI_API_KEY.get_secret_value(),
    )


def get_indexer(vector_store: Chroma = Depends(get_vector_store)) -> ProtocolIndexer:
    return ProtocolIndexer(vector_store)


def get_rag_chain(
    vector_store: Chroma = Depends(get_vector_store),
    indexer: ProtocolIndexer = Depends(get_indexer),
    llm: BaseChatModel | None = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> EligibilityRAGChain:
    return EligibilityRAGChain(vector_store=vector_store, indexer=indexer, llm=llm, settings=settings)
