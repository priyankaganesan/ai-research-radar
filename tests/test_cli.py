from typer.testing import CliRunner

from radar.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["config", "init-db", "ingest", "ask", "daily", "rate", "watchlist", "eval"]:
        assert cmd in result.output


def test_config_hides_secrets(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-value")
    from radar.config import get_settings

    get_settings.cache_clear()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "secret-value" not in result.output
    assert "semantic_scholar_api_key = set" in result.output


def test_init_db(tmp_path):
    result = runner.invoke(app, ["init-db"])
    assert result.exit_code == 0
    assert (tmp_path / "radar.db").exists()


def test_unbuilt_commands_say_so():
    result = runner.invoke(app, ["daily"])
    assert result.exit_code == 1
    assert "M2" in result.output
