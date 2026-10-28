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


def test_tracing_turned_off_without_key(monkeypatch, capsys):
    import os

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()
    s = get_settings()
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert s.langsmith_tracing is False
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def test_tracing_kept_with_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    get_settings.cache_clear()
    assert get_settings().langsmith_tracing is True
