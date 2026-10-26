from langchain_ollama import ChatOllama, OllamaEmbeddings

from radar.models import get_chat_model, get_embeddings


def test_chat_model_uses_settings(settings):
    model = get_chat_model(settings)
    assert isinstance(model, ChatOllama)
    assert model.model == settings.chat_model
    assert model.base_url == settings.ollama_host


def test_embeddings_use_settings(settings):
    emb = get_embeddings(settings)
    assert isinstance(emb, OllamaEmbeddings)
    assert emb.model == settings.embed_model
