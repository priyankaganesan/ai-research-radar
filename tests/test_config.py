from radar.config import get_settings


def test_defaults_point_at_local_qwen(settings):
    assert settings.chat_model == "qwen3.5:9b"
    assert settings.embed_model == "qwen3-embedding:0.6b"
    assert settings.ollama_host.startswith("http://127.0.0.1")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    monkeypatch.setenv("CHAT_MODEL", "qwen3.5:4b")
    get_settings.cache_clear()
    s = get_settings()
    assert s.ollama_host == "http://127.0.0.1:11435"
    assert s.chat_model == "qwen3.5:4b"


def test_tracing_off_in_tests(settings):
    assert settings.langsmith_tracing is False
