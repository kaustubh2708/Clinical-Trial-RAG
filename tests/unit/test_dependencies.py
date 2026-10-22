"""Provider selection in get_llm(): building a client never makes a network call."""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.api.dependencies import get_llm
from src.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_openai_without_key_means_no_llm():
    assert get_llm(_settings(LLM_PROVIDER="openai", OPENAI_API_KEY=None)) is None


def test_openai_with_key():
    llm = get_llm(_settings(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-test", LLM_MAX_OUTPUT_TOKENS=1024))
    assert isinstance(llm, ChatOpenAI)
    assert (llm.temperature, llm.seed, llm.max_tokens) == (0.0, 42, 1024)


def test_ollama_needs_no_key_and_bounds_context_and_output():
    llm = get_llm(_settings(LLM_PROVIDER="ollama", LLM_MODEL_NAME="qwen2.5:7b", OPENAI_API_KEY=None,
                            OLLAMA_BASE_URL="http://ollama:11434", OLLAMA_NUM_CTX=16384))
    assert isinstance(llm, ChatOllama)
    assert (llm.model, llm.base_url, llm.num_ctx, llm.num_predict) == ("qwen2.5:7b", "http://ollama:11434", 16384, 2048)
    assert (llm.temperature, llm.seed) == (0.0, 42)
