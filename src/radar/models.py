"""The only place chat and embedding models are built, so tests can swap in fakes."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from radar.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, *, reasoning: bool = False) -> BaseChatModel:
    """Local Qwen through Ollama. `reasoning` turns Qwen's thinking mode on."""
    s = settings or get_settings()
    return ChatOllama(
        model=s.chat_model, base_url=s.ollama_host, temperature=0, reasoning=reasoning
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_host)
