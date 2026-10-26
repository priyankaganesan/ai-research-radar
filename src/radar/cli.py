"""`radar` command line. Commands beyond `config` and `init-db` arrive with later milestones."""

import typer

from radar.config import get_settings

app = typer.Typer(help="AI Research Radar: track, summarize and explain new AI research.")


def _todo(milestone: str) -> None:
    typer.echo(f"Not built yet (planned for {milestone}, see PLAN.md).")
    raise typer.Exit(code=1)


@app.command()
def config() -> None:
    """Show the active settings (secrets hidden)."""
    s = get_settings()
    for key, value in s.model_dump().items():
        if "key" in key:
            value = "set" if value else "not set"
        typer.echo(f"{key} = {value}")


@app.command("init-db")
def init_db() -> None:
    """Create the store and apply migrations."""
    from radar.store.db import connect

    s = get_settings()
    conn = connect(s.db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    typer.echo(f"Store ready at {s.db_path} ({len(tables)} tables).")


@app.command()
def ingest(days: int = 7, categories: str = "cs.AI,cs.CL,cs.LG") -> None:
    """Fetch recent papers into the store."""
    _todo("M1")


@app.command()
def ask(
    question: str,
    no_retrieval: bool = typer.Option(False, "--no-retrieval", help="Answer without tools."),
    thread: str | None = typer.Option(None, help="Conversation ID for follow-ups."),
) -> None:
    """Ask a question, answered from the store with citations."""
    _todo("M1")


@app.command()
def daily() -> None:
    """Run the daily pipeline and write today's digest."""
    _todo("M2")


@app.command()
def rate(paper_id: str, value: str) -> None:
    """Rate a digest item: useful, not_relevant or skipped."""
    _todo("M4")


@app.command()
def watchlist() -> None:
    """Show the early-buzz watchlist."""
    _todo("M4")


@app.command("eval")
def run_eval() -> None:
    """Run the LangSmith eval suite."""
    _todo("M6")


if __name__ == "__main__":
    app()
