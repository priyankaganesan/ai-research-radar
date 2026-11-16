"""The morning digest job (M6): a macOS launchd agent that runs `radar daily` every day.

**launchd** is macOS's service manager (the job of cron and systemd elsewhere). A *launch agent*
is a job that belongs to one user: its description is a small XML file (a *property list*, or
plist) in `~/Library/LaunchAgents/`, and it runs while that user is logged in. No admin rights,
no Homebrew, nothing system-wide, which fits this Mac (see CLAUDE.md).

Why launchd and not cron: with `StartCalendarInterval`, if the Mac is asleep at 7:30, launchd
runs the job when it wakes up (cron would just skip that day). If the Mac is shut down, the
run is skipped. GitHub Actions can't do it either: it can't reach the local Ollama.

`launchctl` loads and controls the job. Modern syntax, used here:
    launchctl bootstrap gui/<uid> <plist>   load (and enable) the job
    launchctl bootout gui/<uid>/<label>     unload it
    launchctl kickstart gui/<uid>/<label>   run it now
    launchctl print gui/<uid>/<label>       state, last exit code, number of runs

This module builds the plist (a pure function, tested) and wraps those four commands. The job
itself is `radar daily --scheduled` (see services.scheduled_daily): it waits for Ollama, skips
a day that already has a digest, and logs to data/logs/daily.log.
"""

import os
import plistlib
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

LABEL = "com.ai-research-radar.daily"
LOG_NAME = "daily.log"


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(
    repo: Path,
    uv: Path,
    hour: int = 7,
    minute: int = 30,
    mode: str | None = None,
    weekdays_only: bool = False,
    home: Path | None = None,
) -> dict:
    """The launch agent, as a dict that plistlib writes as XML.

    launchd starts jobs with an almost empty environment (PATH is just /usr/bin:/bin:...), so
    everything is spelled out: the absolute path to `uv`, the working directory (where `.env`
    and `data/` live), and a PATH that includes ~/.local/bin.
    """
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("hour must be 0-23 and minute 0-59")
    home = home or Path.home()
    args = [str(uv), "run", "--directory", str(repo), "radar", "daily", "--scheduled"]
    if mode:
        args += ["--mode", mode]
    when: dict | list[dict] = {"Hour": hour, "Minute": minute}
    if weekdays_only:  # launchd counts weekdays 0 (Sunday) to 6; one entry per day
        when = [{"Weekday": d, "Hour": hour, "Minute": minute} for d in range(1, 6)]
    log = str(repo / "data" / "logs" / LOG_NAME)
    return {
        "Label": LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": str(repo),
        "StartCalendarInterval": when,
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {
            "PATH": f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",  # write log lines as they happen, not at exit
        },
        "RunAtLoad": False,  # don't run just because it was loaded (e.g. at login)
        "ProcessType": "Standard",  # "Background" would throttle Qwen's CPU/GPU time
    }


def find_uv() -> Path:
    found = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(found).exists():
        raise FileNotFoundError("Couldn't find `uv` on PATH or in ~/.local/bin.")
    return Path(found).resolve()


Runner = Callable[..., subprocess.CompletedProcess]


def _launchctl(run: Runner, *args: str) -> subprocess.CompletedProcess:
    return run(["launchctl", *args], capture_output=True, text=True, check=False)


def _domain() -> str:
    return f"gui/{os.getuid()}"  # the logged-in user's GUI session


def install(plist: dict, path: Path | None = None, run: Runner = subprocess.run) -> Path:
    """Write the plist and (re)load it. Loading an already-loaded job fails, so unload first."""
    path = path or plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(plist["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(plist))
    _launchctl(run, "bootout", f"{_domain()}/{LABEL}")  # fine if it wasn't loaded
    result = _launchctl(run, "bootstrap", _domain(), str(path))
    if result.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {result.stderr.strip()}")
    return path


def remove(path: Path | None = None, run: Runner = subprocess.run) -> bool:
    """Unload the job and delete its plist. Returns whether there was one."""
    path = path or plist_path()
    _launchctl(run, "bootout", f"{_domain()}/{LABEL}")
    if path.exists():
        path.unlink()
        return True
    return False


def run_now(run: Runner = subprocess.run) -> None:
    result = _launchctl(run, "kickstart", f"{_domain()}/{LABEL}")
    if result.returncode != 0:
        raise RuntimeError(f"launchctl kickstart failed (is it installed?): {result.stderr}")


def status(path: Path | None = None, run: Runner = subprocess.run) -> dict:
    """What launchd knows about the job, plus what the plist says."""
    path = path or plist_path()
    out: dict = {"installed": path.exists(), "plist": str(path), "loaded": False}
    if path.exists():
        p = plistlib.loads(path.read_bytes())
        out["when"] = p["StartCalendarInterval"]
        out["command"] = " ".join(p["ProgramArguments"])
        out["log"] = p["StandardOutPath"]
    result = _launchctl(run, "print", f"{_domain()}/{LABEL}")
    if result.returncode == 0:
        out["loaded"] = True
        out.update(parse_print(result.stdout))
    return out


def parse_print(text: str) -> dict:
    """Pick the useful lines out of `launchctl print` (its format is meant for people)."""
    want = {"state": "state", "runs": "runs", "last exit code": "last_exit"}
    out = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(" = ")
        if sep and key in want and want[key] not in out:
            out[want[key]] = value.strip()
    return out


def notify(title: str, message: str, run: Runner = subprocess.run) -> None:
    """A macOS notification banner, via AppleScript. Does nothing on other systems."""
    if sys.platform != "darwin":
        return
    script = f"display notification {_quote(message)} with title {_quote(title)}"
    run(["osascript", "-e", script], capture_output=True, check=False)


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
