"""The AGENT: one process per machine, talking to the dashboard through OUTGOING requests.

    useml monitor start        # in tmux when available, else detached; survives logout
    useml monitor status
    useml monitor stop

    loop:
        GET  <dashboard>/agent/wait?machine=<alias>    held up to 25 s by the dashboard
             → "refresh": push a FULL snapshot now
             → "none":    push a LIGHT one if something changed (at most every
                          `min_interval` s) or if `heartbeat` s passed since the last push
        POST <dashboard>/agent/push

The agent ACCEPTS NO COMMAND but "refresh": nothing the dashboard says can make it run,
stop or change anything. It never opens a port. Errors back off (5 s → 5 min) and are
written to `<home>/logs/monitor.log`; the agent never dies of a network problem.

It runs in its own process, outside every experience: its single connection never
competes with a run's downloads, and the one-thread download policy holds.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ._share import settings as load_settings
from ._snapshot import experience_roots, snapshot

MIN_INTERVAL = 10.0


def space_url(space: str) -> str:
    """`owner/name` → `https://owner-name.hf.space` (Hugging Face's rule)."""
    host = space.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    return f"https://{host}.hf.space"


def dashboard_url(config: dict) -> str:
    if config.get("url"):
        return str(config["url"]).rstrip("/")
    if config.get("space"):
        return space_url(config["space"])
    raise RuntimeError("no dashboard: run `useml monitor deploy`, or set [monitor] url")


def hub_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None
    if not token:
        raise RuntimeError("no Hugging Face token on this machine: run `hf auth login`")
    return token


def _log(message: str) -> None:
    from .. import paths

    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    try:
        with open(paths.directory("logs") / "monitor.log", "a") as file:
            file.write(line + "\n")
    except OSError:
        pass


class Agent:
    def __init__(self, *, url: str | None = None, token: str | None = None,
                 config: dict | None = None) -> None:
        self.config = config or load_settings()
        self.url = (url or dashboard_url(self.config)).rstrip("/")
        self.token = token or hub_token()
        self.alias = self.config["machine"]
        self.last_push = 0.0
        self.last_change = None
        self.running = True

    # ── http ─────────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: bytes | None = None,
                 timeout: float = 40.0) -> dict:
        headers = {"X-Useml-Token": self.token, "User-Agent": "useml-agent"}
        if not self.url.startswith(("http://127.0.0.1", "http://localhost")):
            headers["Authorization"] = f"Bearer {self.token}"   # private Space access
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.url}{path}", data=body, method=method,
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")

    def push(self, *, full: bool) -> None:
        body = json.dumps({"machine": self.alias, "full": full,
                           "snapshot": snapshot(full=full, config=self.config)}).encode()
        self._request("POST", "/agent/push", body, timeout=60)
        self.last_push = time.time()

    # ── change detection ─────────────────────────────────────────────────────

    def _signature(self):
        latest = 0.0
        for root in experience_roots(self.config):
            if not root.is_dir():
                continue
            for name in ("state.json", "run.log"):
                for path in root.rglob(name):
                    try:
                        latest = max(latest, path.stat().st_mtime)
                    except OSError:
                        continue
        return latest

    def step(self) -> str:
        """One round: wait for the dashboard, then push if needed. Returns what it did."""
        answer = self._request("GET", f"/agent/wait?machine={urllib.request.quote(self.alias)}")
        if answer.get("command") == "refresh":
            self.push(full=True)
            return "refresh"
        signature = self._signature()
        changed = signature != self.last_change
        now = time.time()
        if (changed and now - self.last_push >= self.config.get("min_interval", MIN_INTERVAL)) \
                or now - self.last_push >= self.config["heartbeat"]:
            self.push(full=self.last_push == 0.0)
            self.last_change = signature
            return "push"
        return "idle"

    def run(self) -> None:
        import threading

        if threading.current_thread() is threading.main_thread():
            # Only the main thread may set a handler; inside `serve` the agent is a thread
            # and stops with the process.
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "running", False))
        _log(f"agent {self.alias} → {self.url} (sharing: {', '.join(self.config['share'])})")
        delay = 5.0
        while self.running:
            try:
                if self.last_push == 0.0:
                    self.push(full=True)
                    self.last_change = self._signature()
                self.step()
                delay = 5.0
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    _log(f"refused by the dashboard ({error.code}): is the token the Space "
                         f"owner's? retrying in {delay:.0f} s")
                else:
                    _log(f"dashboard error {error.code}; retrying in {delay:.0f} s")
                self._sleep(delay)
                delay = min(delay * 2, 300.0)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                _log(f"cannot reach the dashboard ({error}); retrying in {delay:.0f} s")
                self._sleep(delay)
                delay = min(delay * 2, 300.0)
        _log("agent stopped")

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))


# ── running it detached ───────────────────────────────────────────────────────

def _pid_file() -> Path:
    from .. import paths

    return paths.directory("monitor") / "agent.pid"


def agent_pid() -> int | None:
    from ..exp._launch import alive

    try:
        pid = int(_pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


def start() -> dict:
    """Start the agent in tmux (session `useml-monitor`) or detached; once per machine."""
    import shlex
    import subprocess

    from ..exp._launch import tmux_available

    running = agent_pid()
    if running:
        return {"already": True, "pid": running}
    load_settings()                                     # fail now on a bad config
    entry = ("import sys; from useml.monitor._agent import main; sys.exit(main())")
    command = [sys.executable, "-c", entry]
    if tmux_available():
        subprocess.run(["tmux", "kill-session", "-t", "=useml-monitor"], capture_output=True)
        result = subprocess.run(["tmux", "new-session", "-d", "-s", "useml-monitor", "-c",
                                 os.getcwd(), " ".join(shlex.quote(p) for p in command)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            for _ in range(50):
                if agent_pid():
                    break
                time.sleep(0.1)
            return {"backend": "tmux", "session": "useml-monitor", "pid": agent_pid()}
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               cwd=os.getcwd())
    import threading

    threading.Thread(target=process.wait, daemon=True).start()
    return {"backend": "detached", "pid": process.pid}


def stop() -> bool:
    pid = agent_pid()
    if not pid:
        return False
    os.kill(pid, signal.SIGTERM)
    return True


def main() -> int:
    import useml  # noqa: F401  - paths and download policy

    _pid_file().write_text(str(os.getpid()))
    try:
        Agent().run()
    finally:
        try:
            if _pid_file().read_text().strip() == str(os.getpid()):
                _pid_file().unlink()
        except OSError:
            pass
    return 0
