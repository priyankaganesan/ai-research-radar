"""M6: the launchd job (plist, launchctl calls) and the scheduled daily run. launchctl and
osascript are never really called: a fake runner records the commands."""

import plistlib
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx

from radar import schedule, services


def test_plist_spells_everything_out(tmp_path):
    p = schedule.build_plist(
        Path("/Users/me/radar"),
        Path("/Users/me/.local/bin/uv"),
        6,
        5,
        "multi",
        home=Path("/Users/me"),
    )
    assert p["ProgramArguments"] == [
        "/Users/me/.local/bin/uv", "run", "--directory", "/Users/me/radar",
        "radar", "daily", "--scheduled", "--mode", "multi",
    ]  # fmt: skip
    assert p["StartCalendarInterval"] == {"Hour": 6, "Minute": 5}
    assert p["WorkingDirectory"] == "/Users/me/radar"
    assert p["StandardOutPath"] == "/Users/me/radar/data/logs/daily.log"
    assert p["EnvironmentVariables"]["PATH"].startswith("/Users/me/.local/bin:")
    assert plistlib.loads(plistlib.dumps(p)) == p  # valid plist XML


def test_weekdays_and_bad_times():
    p = schedule.build_plist(Path("/r"), Path("/uv"), 7, 30, weekdays_only=True, home=Path("/h"))
    assert [d["Weekday"] for d in p["StartCalendarInterval"]] == [1, 2, 3, 4, 5]
    try:
        schedule.build_plist(Path("/r"), Path("/uv"), 25, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("hour 25 accepted")


class FakeLaunchctl:
    def __init__(self, fail=()):
        self.commands = []
        self.fail = fail

    def __call__(self, cmd, **kw):
        self.commands.append(cmd)
        code = 1 if cmd[1] in self.fail else 0
        out = (
            "\tstate = not running\n\truns = 3\n\tlast exit code = 0\n" if cmd[1] == "print" else ""
        )
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="boom" if code else "")


def test_install_status_remove(tmp_path):
    plist = schedule.build_plist(tmp_path, Path("/uv"), 7, 30, home=tmp_path)
    path = tmp_path / "LaunchAgents" / "x.plist"
    run = FakeLaunchctl(fail=("bootout",))  # not loaded yet: bootout fails, which is fine
    schedule.install(plist, path, run)
    assert [c[1] for c in run.commands] == ["bootout", "bootstrap"]
    assert run.commands[1][-1] == str(path)
    assert (tmp_path / "data" / "logs").is_dir()

    st = schedule.status(path, run)
    assert st["installed"] and st["loaded"]
    assert (st["state"], st["runs"], st["last_exit"]) == ("not running", "3", "0")

    assert schedule.remove(path, run) is True
    assert not path.exists()


def test_install_reports_a_failed_bootstrap(tmp_path):
    plist = schedule.build_plist(tmp_path, Path("/uv"), home=tmp_path)
    try:
        schedule.install(plist, tmp_path / "x.plist", FakeLaunchctl(fail=("bootstrap",)))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("no error")


# ---- health and the scheduled run -------------------------------------------------------------


def ollama(models):
    def handler(req):
        return httpx.Response(200, json={"models": [{"name": m} for m in models]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_health(ctx):
    s = ctx.settings
    ok = services.health(ctx, ollama([s.chat_model, s.embed_model]))
    assert ok["ok"] and ok["db"] and ok["ollama"]
    missing = services.health(ctx, ollama([s.embed_model]))
    assert not missing["ok"] and missing["models"][s.chat_model] is False

    def down(req):
        raise httpx.ConnectError("refused")

    dead = services.health(ctx, httpx.Client(transport=httpx.MockTransport(down)))
    assert not dead["ollama"] and "Ollama" in dead["error"]


def test_scheduled_run_waits_for_ollama_then_runs(ctx):
    checks = iter([{"ok": False, "models": {}, "error": "x"}, {"ok": True}])
    slept = []
    fake = SimpleNamespace(picks=[1, 2, 3], date="2026-11-14", path="p", candidates=9)
    out = services.scheduled_daily(
        ctx, check=lambda: next(checks), sleep=slept.append, run_daily=lambda c, mode, f=fake: f
    )
    assert out["status"] == "done" and out["message"] == "3 picks for 2026-11-14"
    assert slept == [30]


def test_scheduled_run_gives_up_and_never_raises(ctx):
    down = {"ok": False, "models": {"qwen3.5:9b": False}, "error": ""}
    out = services.scheduled_daily(ctx, wait_s=60, check=lambda: down, sleep=lambda s: None)
    assert out["status"] == "failed" and "qwen3.5:9b" in out["message"]

    def boom(c, mode):
        raise RuntimeError("summarize failed")

    out = services.scheduled_daily(ctx, check=lambda: {"ok": True}, run_daily=boom)
    assert out == {"status": "failed", "message": "RuntimeError: summarize failed"}


def test_scheduled_run_skips_a_day_already_done(filled_ctx):
    with filled_ctx.conn:
        filled_ctx.conn.execute(
            "INSERT INTO digests (date, item_id, rank, slot, score, reason)"
            " VALUES (?, '2401.10774', 1, 'main', 1.0, 'r')",
            (date.today().isoformat(),),
        )
    out = services.scheduled_daily(filled_ctx, check=lambda: {"ok": True})
    assert out["status"] == "skipped"


def test_scheduled_run_reports_an_empty_digest(ctx):
    """A run that picks nothing is "empty", not "done", so the banner doesn't say it's ready."""
    for candidates, why in [(0, "nothing new was fetched"), (40, "every candidate was filtered")]:
        fake = SimpleNamespace(picks=[], date="2026-11-20", path="p", candidates=candidates)
        out = services.scheduled_daily(
            ctx, check=lambda: {"ok": True}, run_daily=lambda c, mode, f=fake: f
        )
        assert out["status"] == "empty"
        assert f"{candidates} candidates" in out["message"] and why in out["message"]
