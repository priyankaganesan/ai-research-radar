import pytest

from radar.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Tests never read the real .env, never trace to LangSmith, and use a temp store."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "radar.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()
