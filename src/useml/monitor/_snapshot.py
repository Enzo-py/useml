"""A SNAPSHOT: what a machine reports — its experiences and its resources — filtered by
the `[monitor] share` groups and masked (`_share`).

A LIGHT snapshot (sent every `heartbeat` seconds and when something changes) leaves out
results, figures and configs; a FULL one (sent when someone clicks Refresh) has them. The
dashboard keeps the heavy fields of the last full snapshot.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ._share import Masker, settings as load_settings, wanted

MAX_RESULT_BYTES = 64 * 1024
MAX_FIGURE_BYTES = 400 * 1024
MAX_FIGURES = 6
MAX_METRIC_POINTS = 500


def experience_roots(config: dict) -> list[Path]:
    if config["roots"]:
        return config["roots"]
    from ..exp import root

    return [root()]


def _experiences(roots: list[Path]):
    from ..exp._experience import Experience, _is_ours

    for base in roots:
        if not base.is_dir():
            continue
        for spec in sorted(base.rglob("experience.json")):
            if "steps" in spec.parent.parts or not _is_ours(spec):
                continue
            yield Experience(spec.parent, spec.parent.relative_to(base).as_posix())


def _walk(value, kind: str):
    if isinstance(value, dict):
        if set(value) == {kind}:
            yield value[kind]
            return
        for item in value.values():
            yield from _walk(item, kind)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, kind)


def _tail(path: Path, lines: int) -> list[str]:
    try:
        with open(path, "rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 256 * 1024))
            text = file.read().decode(errors="replace")
    except OSError:
        return []
    # A progress bar redraws itself with \r: keep only what it shows last.
    return [line.rsplit("\r", 1)[-1] for line in text.splitlines()[-lines:]]


def _metrics(step_dir: Path) -> list[dict]:
    points = []
    index = step_dir / "checkpoints" / "index.json"
    if index.exists():
        try:
            for entry in json.loads(index.read_text()):
                values = {k: v for k, v in entry.get("metrics", {}).items()
                          if isinstance(v, (int, float))}
                if values:
                    points.append({"step": entry["step"], **values})
        except (OSError, ValueError, KeyError):
            pass
    live = step_dir / "metrics.jsonl"
    if live.exists():
        try:
            for line in live.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    points.append({k: v for k, v in entry.items()
                                   if isinstance(v, (int, float)) and k != "time"})
        except (OSError, ValueError):
            pass
    points.sort(key=lambda point: point.get("step", 0))
    if len(points) > MAX_METRIC_POINTS:                     # keep the shape, not every point
        stride = len(points) / MAX_METRIC_POINTS
        points = [points[int(i * stride)] for i in range(MAX_METRIC_POINTS)] + [points[-1]]
    return points


def _configs(spec: dict) -> list[dict]:
    out, seen = [], set()
    for recipe in _walk(spec.get("steps", []), "$config"):
        key = (recipe["name"], recipe.get("hash"))
        if key in seen:
            continue
        seen.add(key)
        entry = {"name": recipe["name"], "kind": recipe.get("kind"), "hash": recipe.get("hash"),
                 "overrides": recipe.get("overrides") or {}}
        try:
            from .. import config

            entry["yaml"] = config.load_raw(recipe["name"], recipe.get("kind"),
                                            overrides=recipe.get("overrides")).to_yaml()
        except Exception:                                    # noqa: BLE001 - library elsewhere
            pass
        out.append(entry)
    return out


def _data(spec: dict) -> list[str]:
    names = {recipe["name"] for recipe in _walk(spec.get("steps", []), "$data")}
    for recipe in _walk(spec.get("steps", []), "$dataset"):
        names.add(recipe["name"] + (f"[{recipe['split']}]" if recipe.get("split") else ""))
    return sorted(names)


def describe_experience(experience, config: dict, *, full: bool) -> dict:
    share = set(config["share"])
    spec = experience.spec or {}
    state = experience.state
    steps = experience.steps()
    status = state.get("status", "new")
    try:
        updated = (experience.folder / "state.json").stat().st_mtime
    except OSError:
        updated = None
    out = {"name": experience.name, "status": status, "updated": updated,
           "done": sum(step["done"] for step in steps), "total": len(steps),
           "current_step": state.get("step")}
    if "steps" in share:
        out["steps"] = [{"name": s["name"], "function": s["function"], "done": s["done"],
                         "seconds": s["seconds"], "output": s["output"]} for s in steps]
    if "progress" in share:
        out["progress"] = state.get("step_progress")
    if "errors" in share and status in ("failed", "died", "stopped"):
        meaning = (state.get("exit") or {}).get("meaning")
        if status == "died":
            meaning = "the run and its supervisor disappeared without recording anything"
        message = "\n".join(str(state.get("error") or "").splitlines()[-20:])
        out["error"] = {"meaning": meaning, "message": message}
    if "log" in share:
        out["log"] = _tail(experience.folder / "run.log", config["log_lines"])
    if "metrics" in share:
        metrics = {s["name"]: points for s in steps
                   if (points := _metrics(experience.path(s["name"])))}
        if metrics:
            out["metrics"] = metrics
    if "data" in share:
        out["datasets"] = _data(spec)
    if not full:
        return out
    if "configs" in share:
        out["configs"] = _configs(spec)
    if "results" in share:
        results = {}
        for step in steps:
            path = experience.path(step["name"]) / "output.json"
            if not step["done"] or not path.exists():
                continue
            size = path.stat().st_size
            try:
                if size <= MAX_RESULT_BYTES:
                    results[step["name"]] = json.loads(path.read_text())
                else:
                    # A big report: its numbers, not its long lists (they would be MBs).
                    from ..projects import MAX_REPORT_BYTES, numbers_of

                    if size <= MAX_REPORT_BYTES:
                        results[step["name"]] = {"__numbers_of_a_report_of__": size,
                                                 **numbers_of(json.loads(path.read_text()))}
            except ValueError:
                pass
        out["results"] = results
    if "figures" in share:
        figures = []
        for step in steps:
            if not step["done"]:
                continue
            for png in sorted(experience.path(step["name"]).glob("*.png")):
                if len(figures) >= MAX_FIGURES:
                    break
                if png.stat().st_size <= MAX_FIGURE_BYTES:
                    figures.append({"step": step["name"], "name": png.name,
                                    "png": base64.b64encode(png.read_bytes()).decode()})
        out["figures"] = figures
    return out


# ── the machine ───────────────────────────────────────────────────────────────

def _gpus() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,"
                               "memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            try:
                out.append({"index": int(parts[0]), "name": parts[1], "util": int(parts[2]),
                            "mem_used": int(parts[3]) << 20, "mem_total": int(parts[4]) << 20})
            except ValueError:
                continue
    return out


def _ram() -> dict | None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except OSError:
        try:
            import psutil

            available = psutil.virtual_memory().available
        except ImportError:
            pass
    if available is None:
        return None                     # unknown (macOS without psutil): better than a wrong 0
    return {"total": total, "available": available}


def describe_machine(config: dict, roots: list[Path]) -> dict:
    from .. import __version__, paths

    out = {"alias": config["machine"], "useml": __version__, "time": time.time()}
    if config.get("label"):
        out["label"] = config["label"]
    if "machine" not in config["share"]:
        return out
    out["gpus"] = _gpus()
    out["ram"] = _ram()
    disks, seen = [], set()
    for label, folder in [("useml", paths.home()), *[("experiences", r) for r in roots]]:
        try:
            usage = shutil.disk_usage(folder if Path(folder).exists() else Path(folder).parent)
        except OSError:
            continue
        if (usage.total, usage.free) in seen:
            continue
        seen.add((usage.total, usage.free))
        disks.append({"label": label, "free": usage.free, "total": usage.total})
    out["disks"] = disks
    try:
        out["load"] = list(os.getloadavg())
    except OSError:
        pass
    return out


def describe_setup(roots: list[Path]) -> dict:
    """useml on this machine: what is set, how big, what to do. Only in full snapshots."""
    import platform as _platform

    from .. import __version__, hub, paths
    from .. import _settings

    info: dict = {"useml": __version__, "python": _platform.python_version(),
                  "config_file": str(paths.config_path()), "suggestions": []}
    settings = []
    for entry in _settings.listing():
        value = entry["value"]
        if entry["key"] == "exp.notify" and value:
            value = "<set>"                    # a command may hold a webhook secret
        settings.append({**entry, "value": value})
    info["settings"] = settings
    status = hub.status()
    info["policy"] = {key: status[key] for key in ("parallel_allowed", "xet_disabled",
                                                   "hub_pool_patched", "socket_guard")}
    downloads = hub.ls()
    info["downloads"] = {"count": len(downloads),
                         "bytes": sum(entry["size"] for entry in downloads)}
    stray = hub.scan(measure=False)
    info["home_caches"] = [{"path": entry["path"], "variable": entry["variable"]} for entry in stray]
    if stray and paths.root():
        info["suggestions"].append("useml hub scan --move --apply")
    try:
        from .. import store as stores

        opened = stores.open()
        total, budget = opened.total_bytes(), stores.budget()
        info["store"] = {"bytes": total, "budget": budget, "spaces": len(opened.declared()),
                         "free": stores.free_disk_bytes(opened.root)}
        if budget and total > budget:
            info["suggestions"].append("useml store gc --apply")
    except Exception as error:                               # noqa: BLE001
        info["store"] = {"error": str(error)}
    try:
        from .. import data

        tables = data.tables()
        info["processed"] = {"tables": len(tables), "bytes": sum(t["bytes"] for t in tables),
                             "names": len(data.ls(measure=False)),
                             "unused_tables": sum(1 for t in tables if not t["names"])}
    except Exception as error:                               # noqa: BLE001
        info["processed"] = {"error": str(error)}
    try:
        from .. import datasets

        entries = datasets.ls(measure=False)
        info["datasets"] = [{"name": e["name"], "records": e["records"],
                             "state": "ready" if e["current"] else
                             "outdated" if e["prepared"] else "never prepared"}
                            for e in entries]
        for entry in info["datasets"]:
            if entry["state"] == "outdated":
                info["suggestions"].append(f"useml datasets prepare {entry['name']}")
    except Exception as error:                               # noqa: BLE001
        info["datasets"] = {"error": str(error)}
    counts: dict = {}
    for experience in _experiences(roots):
        status_name = experience.status
        counts[status_name] = counts.get(status_name, 0) + 1
    info["experiences"] = counts
    try:
        from ..exp._launch import logout_kills_processes, tmux_available

        info["tmux"] = tmux_available()
        warning = logout_kills_processes()
        if warning:
            info["warnings"] = [warning]
            info["suggestions"].append("loginctl enable-linger $USER")
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from .._setup import checks

        info["checks"] = checks()
        for entry in info["checks"]:
            if entry["fix"] and entry["level"] != "ok" and entry["fix"] not in info["suggestions"]:
                info["suggestions"].append(entry["fix"])
    except Exception as error:                               # noqa: BLE001
        info["checks"] = [{"level": "warn", "area": "check", "message": f"useml check: {error}",
                           "fix": "useml check"}]
    return info


def _projects_by_root(roots: list[Path]) -> dict:
    """{experiences folder: its project's declaration or None}."""
    from .. import projects

    out = {}
    for root in roots:
        try:
            out[root] = projects.declared(root if root.is_dir() else root.parent)
        except Exception:                                    # noqa: BLE001 - a broken toml
            out[root] = None
    return out


def _reported(config: dict, roots: list[Path]):
    """(experience, its project or None) for every experience this machine reports."""
    declared = _projects_by_root(roots)
    for root in roots:
        for experience in _experiences([root]):
            if wanted(experience.name, config["experiences"]):
                yield experience, declared.get(root)


def _public(project: dict) -> dict:
    return {key: value for key, value in project.items() if key != "folder"}


def snapshot(*, full: bool = True, config: dict | None = None) -> dict:
    """What this machine would send now, filtered and masked."""
    from .. import projects

    config = config or load_settings()
    roots = experience_roots(config)
    share_projects = "projects" in config["share"]
    experiences, declarations = [], {}
    for experience, project in _reported(config, roots):
        out = describe_experience(experience, config, full=full)
        if share_projects and project:
            declarations[project["name"]] = _public(project)
            spec = experience.spec or {}
            out["project"] = project["name"]
            out["method"] = projects.method_of(experience.name, spec, project)
            out["factors"] = projects.factors(spec)
            out["version"] = projects._code_version(spec)
        experiences.append(out)
    raw = {"machine": describe_machine(config, roots), "experiences": experiences,
           "share": config["share"], "full": full}
    if declarations:
        raw["projects"] = list(declarations.values())
    if full and "setup" in config["share"]:
        raw["setup"] = describe_setup(roots)
    return masker(config, roots).apply(raw)


def masker(config: dict, roots: list[Path]) -> Masker:
    from .. import paths

    home = paths.home().expanduser().resolve()
    # An experiences folder inside useml's own folder belongs to no project: it is masked
    # as <useml>, not <project>.
    folders = [root.parent for root in roots
               if not (root.resolve() == home or home in root.resolve().parents)]
    return Masker(projects=folders, extra=config["redact"])


MAX_RECORDS_PER_PUSH = 10


def pending_records(config: dict, sent: set) -> list[dict]:
    """Records of ended runs of a project not sent yet (at most MAX_RECORDS_PER_PUSH),
    masked like the snapshot. Empty unless the group "projects" is shared."""
    from .. import projects

    if "projects" not in config["share"]:
        return []
    roots = experience_roots(config)
    out = []
    for experience, project in _reported(config, roots):
        if project is None or experience.status not in projects.ENDED:
            continue
        if projects.run_id(project["name"], experience) in sent:
            continue
        out.append(projects.record(experience, project,
                                   machine=config.get("label") or config["machine"]))
        if len(out) >= MAX_RECORDS_PER_PUSH:
            break
    return masker(config, roots).apply(out) if out else []
