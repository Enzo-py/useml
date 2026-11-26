"""useml monitor — the relay and the dashboard.

Runs in a PRIVATE Hugging Face Space (Docker), or locally with `useml monitor serve`. It
depends on FastAPI only: the Space never installs useml.

    browser ── HTTPS ──▶ Space ◀── HTTPS (outgoing) ── useml agent on each machine

THE MACHINES NEVER ACCEPT A CONNECTION. Each agent keeps asking the Space "anything for
me?" (`/agent/wait`, held up to 25 s). A click on Refresh answers that question with
"refresh"; the agent reads its experiences and pushes a snapshot (`/agent/push`); the
click returns with it.

THE ONLY COMMAND IS "refresh". Even with full control of this app, nobody can make an
agent run anything: agents only send data.

WHO MAY CALL WHAT
  • in a Space: the Space is private, so Hugging Face only lets its owner (and the
    owner's organisation) reach it. Agents also send their HF token in `X-Useml-Token`;
    it is checked once against huggingface.co (`whoami`) and must belong to the Space's
    owner. Only a hash of it is kept, in memory, for 10 minutes.
  • locally: every request needs the token printed by `useml monitor serve` (query
    `?token=` once, then a cookie), and the server listens on 127.0.0.1 only.

NOTHING IS WRITTEN TO DISK. The state lives in memory; after a restart the agents
push it again within a minute.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
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
REFRESH_SECONDS = 12.0
HEAVY = ("results", "figures", "configs")

app = FastAPI(title="useml monitor", docs_url=None, redoc_url=None, openapi_url=None)

_machines: dict = {}        # alias -> {"snapshot", "received", "polled"}
_pending: dict = {}         # alias -> True when a refresh is asked
_wake: dict = {}            # alias -> asyncio.Event: a waiting agent has something to do
_pushed: dict = {}          # alias -> asyncio.Event: a snapshot arrived
_tokens: dict = {}          # sha256(token) -> expiry of a successful check


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


def _state() -> dict:
    now = time.time()
    machines = []
    for alias, entry in sorted(_machines.items()):
        machines.append({"alias": alias, "received": entry["received"],
                         "age": now - entry["received"],
                         "polled_age": now - entry.get("polled", entry["received"]),
                         "snapshot": entry["snapshot"]})
    return {"now": now, "machines": machines, "space": IN_SPACE}


@app.get("/api/state")
async def state(request: Request):
    _require(_browser_allowed(request))
    return JSONResponse(_state(), headers={"Cache-Control": "no-store"})


@app.post("/api/refresh")
async def refresh(request: Request):
    """Ask agents for a fresh snapshot; wait (up to ~12 s) for their answers."""
    _require(_browser_allowed(request))
    try:
        body = await request.json()
    except Exception:                                              # noqa: BLE001
        body = {}
    wanted = [body["machine"]] if body.get("machine") else list(_machines)
    waits = []
    for alias in wanted:
        if alias not in _machines:
            continue
        _pending[alias] = True
        _event(_pushed, alias).clear()
        _event(_wake, alias).set()
        waits.append(_event(_pushed, alias).wait())
    if waits:
        try:
            await asyncio.wait_for(asyncio.gather(*waits), timeout=REFRESH_SECONDS)
        except asyncio.TimeoutError:
            pass                       # some machine did not answer: shown by its age
    return JSONResponse(_state(), headers={"Cache-Control": "no-store"})


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
    if _pending.pop(machine, False):
        return {"command": "refresh"}
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
    return {"ok": True}


@app.get("/healthz")
async def health():
    return {"ok": True, "machines": len(_machines)}
