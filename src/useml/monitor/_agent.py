"""The AGENT: one process per machine, talking to the dashboard through OUTGOING requests.

    useml monitor start        # in tmux when available, else detached; survives logout
    useml monitor status
    useml monitor stop

    loop:
        GET  <dashboard>/agent/wait?machine=<alias>    held up to 25 s by the dashboard
             → "refresh": push a snapshot now (full, or light when asked)
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
from ._snapshot import experience_roots, pending_records, snapshot

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
        self.sent = self._load_sent()          # ids of run records the dashboard has kept
        self.history_retry = 0.0               # when to offer records again to a dashboard
                                               # that keeps no history

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
        payload = {"machine": self.alias, "full": full,
                   "snapshot": snapshot(full=full, config=self.config)}
        records = []
        if time.time() >= self.history_retry:
            try:
                records = pending_records(self.config, self.sent)
            except Exception as error:                       # noqa: BLE001 - never block a push
                _log(f"cannot describe a finished run: {error}")
        if records:
            payload["records"] = records
        answer = self._request("POST", "/agent/push", json.dumps(payload, default=str).encode(),
                               timeout=120 if records else 60)
        self.last_push = time.time()
        if records:
            stored = set(answer.get("stored") or [])
            if stored:
                self.sent |= stored
                self._save_sent()
            if not answer.get("history"):
                self.history_retry = time.time() + 3600      # ask again in an hour

    # ── the run records already kept ─────────────────────────────────────────

    @staticmethod
    def _sent_file() -> Path:
        from .. import paths

        return paths.directory("monitor") / "recorded.json"

    def _load_sent(self) -> set:
        try:
            return set(json.loads(self._sent_file().read_text()).get("ids", []))
        except (OSError, ValueError, AttributeError):
            return set()

    def _save_sent(self) -> None:
        path = self._sent_file()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"ids": sorted(self.sent)}))
        os.replace(temporary, path)

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
            self.push(full=answer.get("full", True) is not False)
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
            def stop_now(*_):
                # Leave at once, even in the middle of a long poll (up to 25 s).
                self.running = False
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, stop_now)
        _log(f"agent {self.alias} → {self.url} (sharing: {', '.join(self.config['share'])})")
        try:
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
        finally:
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
    for _ in range(100):                     # until it says it runs (its pid file), ≤ 10 s
        if agent_pid() or process.poll() is not None:
            break
        time.sleep(0.1)
    return {"backend": "detached", "pid": process.pid}


def stop(timeout: float = 15.0) -> bool:
    """Stop the agent and wait until it is gone (so that `start` right after works)."""
    pid = agent_pid()
    if not pid:
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while agent_pid() and time.time() < deadline:
        time.sleep(0.2)
    if agent_pid():
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
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
