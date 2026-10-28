from typer.testing import CliRunner

from radar.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in [
        "config",
        "init-db",
        "ingest",
        "search",
        "ask",
        "daily",
        "rate",
        "watchlist",
        "eval",
        "reindex",
        "compare-backends",
    ]:
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


def test_search_on_empty_store(monkeypatch):
    from conftest import FakeEmbeddings

    monkeypatch.setattr("radar.models.get_embeddings", lambda settings=None: FakeEmbeddings())
    result = runner.invoke(app, ["search", "Medusa"])
    assert result.exit_code == 0
    assert "radar ingest" in result.output


def test_eval_without_file():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "questions.example.jsonl" in result.output


def test_reindex_and_compare_on_lance(monkeypatch):
    from conftest import SAMPLE_ITEMS, FakeEmbeddings

    from radar.config import get_settings
    from radar.context import build_context
    from radar.rag.index import index_items

    monkeypatch.setattr("radar.models.get_embeddings", lambda settings=None: FakeEmbeddings())
    monkeypatch.setenv("VECTOR_BACKEND", "lance")
    ctx = build_context(get_settings())
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, SAMPLE_ITEMS)

    result = runner.invoke(app, ["compare-backends"])
    assert result.exit_code == 0, result.output
    assert "lancedb (flat, exact) | 1.000" in result.output
    assert "Saved to evals/results/backends-" in result.output

    result = runner.invoke(app, ["reindex", "--backend", "lance", "--ann"])
    assert result.exit_code == 1
    assert "at least 256" in result.output
