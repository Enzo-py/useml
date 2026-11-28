"""`useml mcp` — the `useml info` views as tools of an MCP server, for AI assistants.

    claude mcp add useml -- useml mcp            # Claude Code: once, on your computer

MCP (Model Context Protocol) is how an assistant calls tools on your machine: it starts
this process and talks JSON-RPC over its standard input and output. Each tool returns the
same short text as `useml info`:

    useml_overview      machines, what runs, what failed in the last 24 h, projects
    useml_project       a project: methods, running runs, latest runs; with `method`, its
                        runs table, links between factors and results, versions
    useml_run           one kept run: factors, results, curves, error (+ code, configs)
    useml_experience    an experience as its machine reports it now, with its log
    useml_machine       a machine: resources, the health checks of useml there

READ ONLY: no tool starts, stops or changes anything. The data comes from your private
dashboard with this computer's Hugging Face token (or this machine only), as `useml info`.
No dependency: the protocol is a few JSON messages.
"""
from __future__ import annotations

import json
import sys

PROTOCOL = "2024-11-05"

TOOLS = [
    {"name": "useml_overview",
     "description": "Start here. The state of every machine and experiment tracked by useml: "
                    "machines (GPU, disk, useml health), experiences running now (progress, "
                    "latest metric values), failures of the last 24 hours with their cause, "
                    "projects (focus method, runs kept). Short text.",
     "inputSchema": {"type": "object", "properties": {
         "local": {"type": "boolean", "description": "this machine only, not the dashboard"}}}},
    {"name": "useml_project",
     "description": "One research project. Without `method`: its methods (each with its best "
                    "run), running runs, latest runs. With `method`: that method's runs table "
                    "(results first, then only the factors that differ), the strongest links "
                    "between factors and results, and the code versions.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string"},
         "method": {"type": "string"},
         "result": {"type": "string", "description": "the result key to rank runs by"},
         "limit": {"type": "integer", "description": "rows of the runs table (default 25)"},
         "factors": {"type": "integer", "description": "factor columns (default 10)"},
         "family": {"type": "string", "description": "with method: the family of runs (first part of their names under the method)"},
         "local": {"type": "boolean"}}}},
    {"name": "useml_benchmarks",
     "description": "A project's benchmarks (where scores were measured, read from result keys "
                    "or run names). With `benchmark`: its sheet — one row per model at its best "
                    "run, its metrics (*best, _second), by method.",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": {"type": "string"},
         "benchmark": {"type": "string"},
         "metrics": {"type": "integer", "description": "metric columns (default 6)"},
         "local": {"type": "boolean"}}}},
    {"name": "useml_run",
     "description": "One finished run: its factors, results, curves (first → last, best), "
                    "steps, data, configs, error. `code` adds its declaration as Python (to "
                    "write an alternative); `configs` adds the YAML of its configs.",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {
         "ref": {"type": "string", "description": "a run id (prefix) or an experience name"},
         "project": {"type": "string"},
         "code": {"type": "boolean"},
         "configs": {"type": "boolean"},
         "local": {"type": "boolean"}}}},
    {"name": "useml_experience",
     "description": "An experience as its machine reports it now (running or not): status, "
                    "progress, steps, error, curves, results, and the end of its log.",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {
         "ref": {"type": "string", "description": "<machine>:<experience name>, or the name"},
         "log": {"type": "integer", "description": "log lines (default 15, 0: none)"},
         "local": {"type": "boolean"}}}},
    {"name": "useml_machine",
     "description": "One machine: GPUs, RAM, disks, experiences by status, and the health "
                    "checks of useml there, each with the command that fixes it.",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {
         "ref": {"type": "string", "description": "the machine's identifier or name"},
         "local": {"type": "boolean"}}}},
]


def call(name: str, arguments: dict) -> str:
    """Run one tool; returns its text (raises `InfoError` for a wrong request)."""
    from . import info

    src = info.source(local=bool(arguments.get("local")))
    if name == "useml_overview":
        return info.overview(src)
    if name == "useml_project":
        return info.project(src, arguments["name"], method=arguments.get("method"),
                            result=arguments.get("result"),
                            limit=int(arguments.get("limit") or 25),
                            max_factors=int(arguments.get("factors") or 10),
                            family_name=arguments.get("family"))
    if name == "useml_benchmarks":
        if arguments.get("benchmark"):
            return info.bench(src, arguments["project"], arguments["benchmark"],
                              metrics=int(arguments.get("metrics") or 6))
        return info.benchmarks(src, arguments["project"])
    if name == "useml_run":
        return info.run(src, arguments["ref"], project_name=arguments.get("project"),
                        code=bool(arguments.get("code")), configs=bool(arguments.get("configs")))
    if name == "useml_experience":
        log = arguments.get("log")
        return info.experience(src, arguments["ref"], log=15 if log is None else int(log))
    if name == "useml_machine":
        return info.machine(src, arguments["ref"])
    raise info.InfoError(f"unknown tool {name!r}")


def handle(message: dict) -> dict | None:
    """One JSON-RPC message → its answer (None for a notification)."""
    from . import __version__
    from .info import InfoError

    method, ident = message.get("method"), message.get("id")
    if ident is None:
        return None                                    # notifications need no answer
    if method == "initialize":
        asked = (message.get("params") or {}).get("protocolVersion") or PROTOCOL
        result = {"protocolVersion": asked, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "useml", "version": __version__},
                  "instructions": "Read-only views of the user's machine-learning runs and "
                                  "projects. Call useml_overview first, then drill down."}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        try:
            text, error = call(params.get("name", ""), params.get("arguments") or {}), False
        except (InfoError, KeyError, ValueError) as problem:
            text, error = f"useml: {problem}", True
        except Exception as problem:                                # noqa: BLE001
            text, error = f"useml: {type(problem).__name__}: {problem}", True
        result = {"content": [{"type": "text", "text": text}], "isError": error}
    else:
        return {"jsonrpc": "2.0", "id": ident,
                "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def serve(stdin=None, stdout=None) -> int:
    """Answer JSON-RPC messages, one per line, until the input ends."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    # The output carries the protocol only: anything printed elsewhere (a warning, a
    # progress bar) goes to the error stream, where the assistant's client ignores it.
    sys.stdout = sys.stderr
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            answer = {"jsonrpc": "2.0", "id": None,
                      "error": {"code": -32700, "message": "not JSON"}}
        else:
            answer = handle(message) if isinstance(message, dict) else None
        if answer is not None:
            stdout.write(json.dumps(answer, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0
