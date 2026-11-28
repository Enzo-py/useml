"""useml monitor — the relay and the dashboard.

Runs in a PRIVATE Hugging Face Space (Docker), or locally with `useml monitor serve`. It
depends on FastAPI only: the Space never installs useml.

    browser ── HTTPS ──▶ Space ◀── HTTPS (outgoing) ── useml agent on each machine

THE MACHINES NEVER ACCEPT A CONNECTION. Each agent keeps asking the Space "anything for
me?" (`/agent/wait`, held up to 25 s). A refresh answers that question with "refresh"
(full: everything shared; light: without results, figures, configs and setup — what the
page's auto-refresh asks for); the agent reads its experiences and pushes a snapshot
(`/agent/push`); the refresh returns with it.

THE ONLY COMMAND IS "refresh". Even with full control of this app, nobody can make an
agent run anything: agents only send data. "Forget" removes a machine from this app's
memory only.

WHO MAY CALL WHAT
  • in a Space: the Space is private, so Hugging Face only lets its owner (and the
    owner's organisation) reach it. Agents also send their HF token in `X-Useml-Token`;
    it is checked once against huggingface.co (`whoami`) and must belong to the Space's
    owner. Only a hash of it is kept, in memory, for 10 minutes.
  • locally: every request needs the token printed by `useml monitor serve` (query
    `?token=` once, then a cookie), and the server listens on 127.0.0.1 only.

THE LIVE STATE IS NOT WRITTEN: it lives in memory; after a restart the agents push it
again within a minute.

THE HISTORY OF PROJECTS IS. When a run of a project ends, its agent sends a record
(factors, numbers, curves, configs — never code). The dashboard keeps it:
  • in a Space: in a PRIVATE dataset repository (`USEML_HISTORY_REPO`, a variable set by
    `useml monitor deploy`), written with `USEML_HISTORY_TOKEN`, a secret the owner adds
    to the Space: a fine-grained token that can write to that dataset only. Without the
    token, records live in memory until the next restart;
  • locally: in a folder (`USEML_HISTORY_DIR`, set by `useml monitor serve`).
An agent marks a record as kept only when the dashboard answers that it stored it.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).resolve().parent
OWNER = os.environ.get("SPACE_AUTHOR_NAME", "")
LOCAL_TOKEN = os.environ.get("USEML_MONITOR_TOKEN", "")
IN_SPACE = bool(os.environ.get("SPACE_ID"))
MAX_PUSH_BYTES = 16 * 1024 * 1024
WAIT_SECONDS = 25.0
REFRESH_SECONDS = 12.0          # a full refresh waits this long at most
LIGHT_REFRESH_SECONDS = 5.0     # an automatic one, this long
LISTENING_SECONDS = 60.0        # an agent that polled within this is waited for
HEAVY = ("results", "figures", "configs")
HISTORY_REPO = os.environ.get("USEML_HISTORY_REPO", "")
HISTORY_TOKEN = os.environ.get("USEML_HISTORY_TOKEN", "")
HISTORY_DIR = os.environ.get("USEML_HISTORY_DIR", "")
RECORD_FORMAT = "useml-run/1"
RECORD_ID = re.compile(r"[0-9a-f]{16}")

app = FastAPI(title="useml monitor", docs_url=None, redoc_url=None, openapi_url=None)

_machines: dict = {}        # alias -> {"snapshot", "received", "polled"}
_pending: dict = {}         # alias -> "full" or "light" when a refresh is asked
_wake: dict = {}            # alias -> asyncio.Event: a waiting agent has something to do
_pushed: dict = {}          # alias -> asyncio.Event: a snapshot arrived
_tokens: dict = {}          # sha256(token) -> expiry of a successful check
_records: dict = {}         # run id -> record (the history of projects)
_projects: dict = {}        # project name -> its declaration, as last sent
_history = {"where": ("dataset" if HISTORY_REPO and HISTORY_TOKEN else
                      "folder" if HISTORY_DIR else None),
            "repo": HISTORY_REPO or None, "loaded": False, "error": None}
_history_lock = threading.Lock()


def _event(table: dict, alias: str) -> asyncio.Event:
    if alias not in table:
        table[alias] = asyncio.Event()
    return table[alias]


# ── authentication ────────────────────────────────────────────────────────────

def _whoami(token: str) -> dict:
    request = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _agent_allowed(request: Request) -> bool:
    token = request.headers.get("x-useml-token", "")
    if not token:
        return False
    if not IN_SPACE:
        return bool(LOCAL_TOKEN) and hmac.compare_digest(token, LOCAL_TOKEN)
    digest = hashlib.sha256(token.encode()).hexdigest()
    if _tokens.get(digest, 0) > time.time():
        return True
    try:
        identity = _whoami(token)
    except Exception:                                              # noqa: BLE001
        return False
    names = {identity.get("name")} | {org.get("name") for org in identity.get("orgs", [])}
    if OWNER and OWNER in names:
        _tokens[digest] = time.time() + 600
        return True
    return False


def _browser_allowed(request: Request) -> bool:
    if IN_SPACE:
        return True                     # private Space: Hugging Face already checked
    given = request.query_params.get("token") or request.cookies.get("useml_monitor", "")
    return bool(LOCAL_TOKEN) and hmac.compare_digest(given, LOCAL_TOKEN)


def _require(allowed: bool) -> None:
    if not allowed:
        raise HTTPException(status_code=403, detail="forbidden")


# ── the history of projects ──────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.") or "project"


def _read_folder(folder: Path) -> None:
    for path in folder.glob("projects/*/project.json"):
        try:
            declaration = json.loads(path.read_text())
            _projects.setdefault(declaration["name"], declaration)
        except (OSError, ValueError, KeyError):
            continue
    for path in folder.glob("projects/*/runs/*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("id"):
            _records.setdefault(record["id"], record)


def _load_history() -> None:
    """Read the kept records once (in a thread: it may download)."""
    with _history_lock:
        if _history["loaded"] or not _history["where"]:
            return
        try:
            if _history["where"] == "folder":
                _read_folder(Path(HISTORY_DIR))
            else:
                from huggingface_hub import snapshot_download

                local = snapshot_download(HISTORY_REPO, repo_type="dataset", token=HISTORY_TOKEN,
                                          allow_patterns=["projects/**"],
                                          local_dir="/tmp/useml-history")
                _read_folder(Path(local))
            _history["loaded"], _history["error"] = True, None
        except Exception as error:                               # noqa: BLE001
            _history["error"] = f"cannot read the history: {error}"
            _history["retry_at"] = time.time() + 60          # not at every refresh


def _write_history(files: dict) -> None:
    """{path in the history: JSON-able} → written (a folder) or one commit (a dataset)."""
    if not files:
        return
    if _history["where"] == "folder":
        for relative, value in files.items():
            path = Path(HISTORY_DIR) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, indent=1))
            os.replace(temporary, path)
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [CommitOperationAdd(path_in_repo=relative,
                                     path_or_fileobj=json.dumps(value, indent=1).encode())
                  for relative, value in files.items()]
    HfApi(token=HISTORY_TOKEN).create_commit(
        HISTORY_REPO, repo_type="dataset", operations=operations,
        commit_message=f"useml: {len(operations)} file(s)")


def _valid_record(record) -> bool:
    return isinstance(record, dict) and record.get("format") == RECORD_FORMAT and \
        isinstance(record.get("id"), str) and bool(RECORD_ID.fullmatch(record["id"])) and \
        isinstance(record.get("project"), str) and bool(record["project"])


async def _keep(records: list, declarations: list) -> list:
    """Store new records (and changed declarations); return the ids now kept."""
    if _history["where"] and not _history["loaded"]:
        await asyncio.to_thread(_load_history)
    valid = [r for r in records if _valid_record(r)]
    files = {}
    for declaration in declarations:
        if isinstance(declaration, dict) and isinstance(declaration.get("name"), str):
            if _projects.get(declaration["name"]) != declaration:
                _projects[declaration["name"]] = declaration
                files[f"projects/{_slug(declaration['name'])}/project.json"] = declaration
    for record in valid:
        previous = _records.get(record["id"])
        same = previous is not None and \
            {k: v for k, v in previous.items() if k != "recorded"} == \
            {k: v for k, v in record.items() if k != "recorded"}
        if not same:                       # new, or sent again with more (`resend`)
            files[f"projects/{_slug(record['project'])}/runs/{record['id']}.json"] = record
        _records[record["id"]] = record
    if not _history["where"]:
        return []                       # in memory only: the agent will offer them again
    try:
        await asyncio.to_thread(_write_history, files)
        _history["error"] = None
    except Exception as error:                                   # noqa: BLE001
        _history["error"] = f"cannot write the history: {error}"
        return []
    return [record["id"] for record in valid]


def _history_status() -> dict:
    return {"kept": bool(_history["where"]), "where": _history["where"], "repo": _history["repo"],
            "loaded": _history["loaded"], "error": _history["error"], "runs": len(_records)}


# ── the browser side ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    _require(_browser_allowed(request))
    response = HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))
    if not IN_SPACE and request.query_params.get("token"):
        response.set_cookie("useml_monitor", request.query_params["token"], httponly=True,
                            samesite="strict")
    response.headers["Cache-Control"] = "no-store"
    return response


def _lighten(snapshot: dict) -> dict:
    """The snapshot without its heavy parts: what the page needs every few seconds. A
    phone then downloads kilobytes, not the figures of every experience."""
    out = {key: value for key, value in snapshot.items() if key != "setup"}
    out["has_setup"] = "setup" in snapshot
    checks = (snapshot.get("setup") or {}).get("checks") or []
    out["health"] = {level: sum(1 for c in checks if c.get("level") == level)
                     for level in ("warn", "fail")}
    out["experiences"] = [{key: value for key, value in experience.items() if key not in HEAVY}
                          for experience in snapshot.get("experiences", [])]
    return out


def _state(light: bool = False) -> dict:
    now = time.time()
    machines = []
    for alias, entry in sorted(_machines.items()):
        snapshot = entry["snapshot"]
        machines.append({"alias": alias, "received": entry["received"],
                         "age": now - entry["received"],
                         "polled_age": now - entry.get("polled", entry["received"]),
                         "snapshot": _lighten(snapshot) if light else snapshot})
    return {"now": now, "machines": machines, "space": IN_SPACE, "history": _history_status()}


def _light(request: Request) -> bool:
    return request.query_params.get("light") in ("1", "true")


async def _history_ready() -> None:
    """Read the kept history once, before the first answer that counts it (a restarted
    dashboard would otherwise say "0 runs" until a project is opened)."""
    if _history["where"] and not _history["loaded"] and time.time() >= _history.get("retry_at", 0):
        await asyncio.to_thread(_load_history)


@app.get("/api/state")
async def state(request: Request):
    """Everything (`?light=1`: without results, figures, configs and setup)."""
    _require(_browser_allowed(request))
    await _history_ready()
    return JSONResponse(_state(_light(request)), headers={"Cache-Control": "no-store"})


def _machine_or_404(alias: str) -> dict:
    if alias not in _machines:
        raise HTTPException(status_code=404, detail="unknown machine")
    return _machines[alias]["snapshot"]


@app.get("/api/experience")
async def experience(request: Request, machine: str, name: str):
    """One experience with its heavy parts: what the detail panel shows."""
    _require(_browser_allowed(request))
    for entry in _machine_or_404(machine).get("experiences", []):
        if entry.get("name") == name:
            return JSONResponse(entry, headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="unknown experience")


@app.get("/api/setup")
async def setup(request: Request, machine: str):
    _require(_browser_allowed(request))
    return JSONResponse(_machine_or_404(machine).get("setup") or {},
                        headers={"Cache-Control": "no-store"})


def _live_by_project() -> dict:
    """{project: [experience as the machines report it now, with its machine]}."""
    out: dict = {}
    for alias, entry in _machines.items():
        snapshot = entry["snapshot"]
        label = (snapshot.get("machine") or {}).get("label")
        for experience in snapshot.get("experiences", []):
            if experience.get("project"):
                light = {k: v for k, v in experience.items() if k not in HEAVY and k != "log"}
                out.setdefault(experience["project"], []).append(
                    {**light, "machine": alias, "machine_label": label,
                     "polled_age": time.time() - entry.get("polled", entry["received"])})
        for declaration in snapshot.get("projects", []):
            if isinstance(declaration, dict) and declaration.get("name"):
                _projects.setdefault(declaration["name"], declaration)
    return out


@app.get("/api/projects")
async def projects(request: Request):
    _require(_browser_allowed(request))
    await _history_ready()
    live = _live_by_project()
    names = set(_projects) | set(live) | {r["project"] for r in _records.values()}
    out = []
    for name in sorted(names):
        declaration = _projects.get(name) or {"name": name}
        runs = [r for r in _records.values() if r.get("project") == name]
        out.append({"name": name, "description": declaration.get("description", ""),
                    "focus": declaration.get("focus"),
                    "methods": len(declaration.get("methods") or {}),
                    "runs": len(runs),
                    "running": sum(1 for e in live.get(name, []) if e.get("status") == "running"),
                    "last": max((r.get("ended") or 0 for r in runs), default=None)})
    return JSONResponse({"projects": out, "history": _history_status()},
                        headers={"Cache-Control": "no-store"})


@app.get("/api/project")
async def project(request: Request, name: str):
    _require(_browser_allowed(request))
    await _history_ready()
    live = _live_by_project().get(name, [])
    runs = sorted((r for r in _records.values() if r.get("project") == name),
                  key=lambda r: r.get("ended") or 0, reverse=True)
    if not runs and not live and name not in _projects:
        raise HTTPException(status_code=404, detail="unknown project")
    return JSONResponse({"project": _projects.get(name) or {"name": name, "methods": {}},
                         "runs": runs, "live": live, "history": _history_status()},
                        headers={"Cache-Control": "no-store"})


@app.post("/api/forget")
async def forget(request: Request):
    """Remove a machine from this dashboard's memory (a machine that was renamed or shut
    down). Nothing is sent to the machine; a running agent simply reappears."""
    _require(_browser_allowed(request))
    try:
        alias = str((await request.json())["machine"])
    except Exception:                                              # noqa: BLE001
        raise HTTPException(status_code=400, detail="machine expected") from None
    for table in (_machines, _pending, _wake, _pushed):
        table.pop(alias, None)
    return {"ok": True}


@app.post("/api/refresh")
async def refresh(request: Request):
    """Ask agents for a fresh snapshot and wait for their answers (a few seconds at most).

    Body: {"machine": alias (default: all), "full": true (default) or false}. Only agents
    that are listening are waited for: a machine that went silent never slows the page."""
    _require(_browser_allowed(request))
    await _history_ready()
    try:
        body = await request.json()
    except Exception:                                              # noqa: BLE001
        body = {}
    full = body.get("full", True) is not False
    wanted = [body["machine"]] if body.get("machine") else list(_machines)
    now = time.time()
    waits = []
    for alias in wanted:
        entry = _machines.get(alias)
        if entry is None or now - entry.get("polled", 0) > LISTENING_SECONDS:
            continue
        _pending[alias] = "full" if full or _pending.get(alias) == "full" else "light"
        _event(_pushed, alias).clear()
        _event(_wake, alias).set()
        waits.append(_event(_pushed, alias).wait())
    if waits:
        try:
            await asyncio.wait_for(asyncio.gather(*waits),
                                   timeout=REFRESH_SECONDS if full else LIGHT_REFRESH_SECONDS)
        except asyncio.TimeoutError:
            pass                       # some machine did not answer: shown by its age
    return JSONResponse(_state(_light(request)), headers={"Cache-Control": "no-store"})


# ── the agent side ────────────────────────────────────────────────────────────

@app.get("/agent/wait")
async def agent_wait(request: Request, machine: str):
    _require(_agent_allowed(request))
    if machine in _machines:
        _machines[machine]["polled"] = time.time()
    wake = _event(_wake, machine)
    if not _pending.get(machine):
        wake.clear()
        try:
            await asyncio.wait_for(wake.wait(), timeout=WAIT_SECONDS)
        except asyncio.TimeoutError:
            pass
    kind = _pending.pop(machine, None)
    if kind:
        return {"command": "refresh", "full": kind == "full"}
    return {"command": "none"}


@app.post("/agent/push")
async def agent_push(request: Request):
    _require(_agent_allowed(request))
    raw = await request.body()
    if len(raw) > MAX_PUSH_BYTES:
        raise HTTPException(status_code=413, detail="snapshot too large")
    try:
        body = json.loads(raw)
        alias = str(body["machine"])[:64]
        snapshot = body["snapshot"]
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=400, detail="bad snapshot") from None
    previous = _machines.get(alias, {}).get("snapshot") or {}
    if not body.get("full"):
        # A light snapshot (heartbeat): keep the heavy fields of the last full one.
        former = {e.get("name"): e for e in previous.get("experiences", [])}
        for experience in snapshot.get("experiences", []):
            old = former.get(experience.get("name"), {})
            for key in HEAVY:
                if key not in experience and key in old:
                    experience[key] = old[key]
        if "setup" not in snapshot and "setup" in previous:
            snapshot["setup"] = previous["setup"]
    _machines[alias] = {"snapshot": snapshot, "received": time.time(),
                        "polled": time.time()}
    _event(_pushed, alias).set()
    stored = []
    records = body.get("records") or []
    declarations = snapshot.get("projects") or []
    if records or declarations:
        stored = await _keep(records if isinstance(records, list) else [],
                             declarations if isinstance(declarations, list) else [])
    return {"ok": True, "history": bool(_history["where"]), "stored": stored}


@app.get("/healthz")
async def health():
    return {"ok": True, "machines": len(_machines)}
