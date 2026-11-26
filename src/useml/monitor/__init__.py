"""useml.monitor — see your experiments from anywhere, through a private dashboard.

    useml monitor snapshot     # EXACTLY what would leave this machine, with your settings
    useml monitor serve        # the dashboard on this machine only (127.0.0.1)
    useml monitor deploy       # create the private Hugging Face Space (once)
    useml monitor start        # the agent of this machine, in tmux / detached
    useml monitor status | stop

Machines open no port: the agent keeps an outgoing HTTPS connection to the dashboard,
which can only ask it for a fresh snapshot. What is sent is chosen by `[monitor] share`
in the machine config (`_share`), masked of paths, hostname and user name, and never
contains code. See the guide: `useml docs monitor`.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from pathlib import Path

from ._agent import Agent, agent_pid, dashboard_url, space_url, start, stop
from ._share import DEFAULT_SHARE, GROUPS, Masker, default_alias, settings
from ._snapshot import snapshot

SPACE_FILES = ("app.py", "index.html", "Dockerfile", "README.md")


def space_folder() -> Path:
    return Path(__file__).resolve().parent / "space"


def serve(*, port: int = 8765, open_agent: bool = True) -> None:
    """Run the dashboard on 127.0.0.1 with this machine's agent, until Ctrl-C."""
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError("useml monitor serve needs FastAPI: pip install 'useml[monitor]'") \
            from error
    token = secrets.token_urlsafe(24)
    os.environ["USEML_MONITOR_TOKEN"] = token
    os.environ.pop("SPACE_ID", None)
    import importlib.util

    spec = importlib.util.spec_from_file_location("useml_monitor_app", space_folder() / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = uvicorn.Server(uvicorn.Config(module.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    if open_agent:
        agent = Agent(url=f"http://127.0.0.1:{port}", token=token)
        threading.Thread(target=agent.run, daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{port}/?token={token}\n"
          f"(this machine only; from elsewhere: ssh -L {port}:127.0.0.1:{port} <server>)\n"
          f"Ctrl-C to stop")
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True


def deploy(space: str | None = None) -> dict:
    """Create (or update) the PRIVATE Space holding the dashboard, then record it in the
    machine config. Refuses to upload into a Space that is not private."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError("deploying needs huggingface_hub: pip install 'useml[monitor]'") \
            from error
    from ._agent import hub_token

    api = HfApi(token=hub_token())
    owner = api.whoami()["name"]
    repo_id = space or f"{owner}/useml-monitor"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=True,
                    exist_ok=True)
    info = api.space_info(repo_id)
    if not getattr(info, "private", False):
        raise RuntimeError(f"{repo_id} exists and is PUBLIC: refusing to deploy a dashboard "
                           f"there. Make it private, or choose another name.")
    api.upload_folder(folder_path=str(space_folder()), repo_id=repo_id, repo_type="space",
                      allow_patterns=list(SPACE_FILES),
                      commit_message="useml monitor dashboard")
    recorded = _record_space(repo_id)
    return {"space": repo_id, "page": f"https://huggingface.co/spaces/{repo_id}",
            "url": space_url(repo_id), "recorded": recorded}


def _record_space(repo_id: str) -> bool:
    """Add `space = …` to `[monitor]` of the machine config if it has none."""
    from .. import paths
    from .._setup import _quote

    path = paths.config_path()
    current = paths.config()
    table = current.get("monitor", {})
    if table.get("space"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    if "monitor" in current:
        return False                     # a [monitor] table exists: never edit inside it
    text += ("" if not text or text.endswith("\n") else "\n") + \
        f"\n[monitor]\nspace = {_quote(repo_id)}\n"
    path.write_text(text)
    return True


def status() -> dict:
    config = settings()
    try:
        url = dashboard_url(config)
    except RuntimeError:
        url = None
    return {"agent_pid": agent_pid(), "machine": config["machine"], "dashboard": url,
            "share": config["share"], "experiences": config["experiences"]}


__all__ = ["snapshot", "settings", "serve", "deploy", "start", "stop", "status", "Agent",
           "GROUPS", "DEFAULT_SHARE", "Masker", "default_alias", "space_url", "space_folder"]
