"""Launching a supervised run, and saying why a run is where it is.

TMUX FIRST. A run is started in a tmux session `useml-<experience>` when tmux exists:
closing the laptop, the SSH connection or the notebook leaves it running, and
`useml exp attach <name>` shows its terminal from any new connection. Without tmux, the
supervisor starts in its own session (setsid, SIGHUP ignored): it survives too, there is
just no terminal to reattach to (`useml exp follow` reads the log instead).

THE ENVIRONMENT goes through a file (`.launch-env.json`, mode 0600, read then deleted by
the supervisor), NOT through tmux: a tmux session receives the environment of the tmux
SERVER, frozen when it first started — a `gpu=1` would silently run on the card of an
older launch. And not on the command line, which `ps` shows to every user.

WHAT DOES NOT SURVIVE, checked before launching: a machine whose systemd kills a user's
processes at logout (`KillUserProcesses=yes`) while lingering is off. tmux does not help
there; `loginctl enable-linger $USER` does.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ._runtime import read_state, write_state


def tmux_available() -> bool:
    if os.environ.get("USEML_NO_TMUX"):
        return False
    tmux = shutil.which("tmux")
    if tmux is None:
        return False
    try:
        return subprocess.run([tmux, "-V"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def session_name(name: str) -> str:
    return "useml-" + re.sub(r"[^A-Za-z0-9_-]", "_", name)[:60]


def logout_kills_processes() -> str | None:
    """A warning if this machine kills the user's processes at logout, else None."""
    if not sys.platform.startswith("linux") or shutil.which("loginctl") is None:
        return None
    kill = False
    for path in [Path("/etc/systemd/logind.conf"),
                 *sorted(Path("/etc/systemd/logind.conf.d").glob("*.conf"))]:
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith("KillUserProcesses="):
                    kill = line.split("=", 1)[1].strip().lower() in ("yes", "true", "1")
        except OSError:
            continue
    if not kill:
        return None
    user = os.environ.get("USER", "")
    try:
        linger = subprocess.run(["loginctl", "show-user", user, "--property=Linger"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        linger = ""
    if linger.endswith("=yes"):
        return None
    return ("this machine kills a user's processes at logout (systemd KillUserProcesses=yes) "
            "and lingering is off: the run would die when you disconnect, tmux included. "
            f"Run once: loginctl enable-linger {user}")


def launch(folder: Path, name: str, env: dict) -> dict:
    """Start the supervisor (in tmux if possible); return {backend, session, pid}."""
    from .. import _progress

    warning = logout_kills_processes()
    if warning:
        _progress.note(f"⚠ {warning}")
    env = dict(env, PYTHONUNBUFFERED="1", USEML_LAUNCH_CWD=os.getcwd())
    env_file = folder / ".launch-env.json"
    descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(env, file)
    from ._supervise import SUPERVISOR_ENTRY

    command = [sys.executable, "-c", SUPERVISOR_ENTRY, str(folder)]
    write_state(folder, status="running", launched=time.time(), host=platform.node(),
                error=None, exit=None, stop_requested=False, supervisor_pid=None,
                runtime_pid=None, step=None, step_progress=None, name=name)
    if tmux_available():
        session = session_name(name)
        subprocess.run(["tmux", "kill-session", "-t", f"={session}"], capture_output=True)
        result = subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", os.getcwd(),
                                 " ".join(shlex.quote(part) for part in command)],
                                capture_output=True, text=True, env=env)
        if result.returncode == 0:
            write_state(folder, backend="tmux", session=session)
            return {"backend": "tmux", "session": session}
        _progress.note(f"⚠ tmux refused session {session!r} "
                       f"({(result.stderr or result.stdout).strip() or result.returncode}); "
                       f"starting detached without it")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True, env=env,
                               cwd=os.getcwd())
    # REAP it when it ends: otherwise a dead supervisor stays a zombie of this notebook
    # kernel, and a zombie still "exists" — the run would look alive forever.
    import threading

    threading.Thread(target=process.wait, daemon=True).start()
    write_state(folder, backend="detached", session=None, supervisor_pid=process.pid)
    return {"backend": "detached", "pid": process.pid}


# ── liveness and diagnosis ────────────────────────────────────────────────────

def alive(pid) -> bool:
    """The process exists AND is not a zombie (dead, waiting for its parent to reap it)."""
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    try:
        with open(f"/proc/{pid}/stat") as file:                 # Linux
            return file.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        pass
    try:                                                        # macOS, BSD
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return bool(state) and not state.startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def tmux_session_exists(session: str | None) -> bool:
    if not session or shutil.which("tmux") is None:
        return False
    return subprocess.run(["tmux", "has-session", "-t", f"={session}"],
                          capture_output=True).returncode == 0


def live_status(folder: Path, state: dict) -> str:
    """`running` only if a process of the run is really alive (on this machine)."""
    status = state.get("status")
    if status != "running":
        return status or "new"
    host = state.get("host")
    if host and host != platform.node():
        return "running"                         # another machine: cannot check, trust it
    pids = [state.get(key) for key in ("supervisor_pid", "runtime_pid", "pid")]
    if any(pid and alive(pid) for pid in pids):
        return "running"
    if tmux_session_exists(state.get("session")):
        return "running"
    launched = state.get("launched") or 0
    if not any(pids) and time.time() - launched < 30:
        return "running"                         # starting: no pid written yet
    return "died"


def _ago(timestamp) -> str:
    if not timestamp:
        return "never"
    seconds = time.time() - float(timestamp)
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago ({time.strftime('%Y-%m-%d %H:%M', time.localtime(float(timestamp)))})"


def why(folder: Path, name: str, log_tail: str) -> str:
    """Where the run is, and if it ended badly, why — in words."""
    state = read_state(folder)
    status = live_status(folder, state)
    lines = [f"{name}: {status}" + (f" on {state['host']}" if state.get("host") else "")]
    if status == "running":
        progress = state.get("step_progress")
        lines.append(f"  step       {state.get('step') or '(between steps)'}"
                     + (f"  {progress[0]}/{progress[1]} {progress[2]}" if progress else ""))
        lines.append(f"  log        last written {_ago(_mtime(folder / 'run.log'))}")
        lines.append(f"  supervisor heartbeat {_ago(state.get('supervisor_heartbeat'))}")
        if state.get("session"):
            lines.append(f"  terminal   useml exp attach {name}   (tmux {state['session']})")
        lines.append(f"  stuck?     useml exp stack {name}   (prints where it is)")
    elif status == "died":
        lines += [
            "  The run AND its supervisor disappeared without recording anything.",
            f"  Last sign of life: {_ago(state.get('supervisor_heartbeat') or _mtime(folder / 'run.log'))}.",
            "  This happens when the machine rebooted, when the session was killed at logout "
            "(systemd KillUserProcesses; see `loginctl enable-linger`), or when an admin or "
            "a scheduler killed every process of the job.",
        ]
    elif status in ("failed", "stopped"):
        crash = folder / "crash.txt"
        if crash.exists():
            lines.append(crash.read_text().rstrip())
            return "\n".join(lines)
        if state.get("error"):
            lines += ["", state["error"].rstrip()]
    elif status == "done":
        exit_info = state.get("exit") or {}
        if exit_info.get("returncode") not in (None, 0):
            lines.append(f"  every step is done, but the process then {exit_info['meaning']}")
        return "\n".join(lines)
    lines += ["", "last lines of run.log:", log_tail]
    return "\n".join(lines)


def _mtime(path: Path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None
