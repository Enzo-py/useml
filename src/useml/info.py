"""useml.info — runs, projects and machines, in few words: for you, and for an AI assistant.

    useml info                          the overview: machines, what runs, what failed, projects
    useml info project robust-vision    methods, running runs, the runs table, correlations
    useml info project robust-vision -m wide
    useml info run wide/w4-mix          one run: factors, results, curves, error (an id works)
    useml info exp gpu-a:wide/w4-mix    an experience as its machine reports it now, + its log
    useml info machine gpu-a            resources and the health checks of useml there
    useml mcp                           the same, as tools of an MCP server (AI assistants)

WHERE FROM: the private dashboard (`[monitor] space` or `url`, read with this machine's
Hugging Face token, as the agent does), so every machine and the kept history are seen
from anywhere. Without a dashboard, or with `--local`: this machine only.

READ ONLY. Nothing here starts, stops or changes anything.

WRITTEN TO BE SHORT: numbers to 4 digits, relative times, only the factors that differ
between runs (the others once), curves as first → last (best @x), long lists cut with
"+N more" and the command that shows the rest. `--json` gives the data unformatted.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request

LOWER_IS_BETTER = re.compile(r"(loss|error|err\b|perplexity|ppl|mse|mae|rmse|wer|cer|fid|"
                             r"time|seconds|latency|memory|regret|distance)", re.I)
ENDED = ("done", "failed", "died", "stopped")
SILENT_AFTER = 180


class InfoError(RuntimeError):
    pass


# ── where the data comes from ─────────────────────────────────────────────────

class Remote:
    """The private dashboard, read like the browser does (with the Hugging Face token)."""

    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        from .monitor._agent import dashboard_url, hub_token
        from .monitor._share import settings

        try:
            self.url = (url or dashboard_url(settings())).rstrip("/")
        except RuntimeError as error:
            raise InfoError(f"{error} — or read this machine only: --local") from None
        self.token = token or hub_token()
        self.where = self.url.split("//")[-1]

    def get(self, path: str, **params):
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        request = urllib.request.Request(f"{self.url}{path}{query}",
                                         headers={"Authorization": f"Bearer {self.token}",
                                                  "X-Useml-Token": self.token})
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise InfoError(f"the dashboard answered {error.code} for {path}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise InfoError(f"cannot reach the dashboard ({error}); --local reads this "
                            f"machine only") from None

    def state(self) -> dict:
        return self.get("/api/state", light=1)

    def projects(self) -> dict:
        return self.get("/api/projects") or {"projects": [], "history": None}

    def project(self, name: str) -> dict | None:
        return self.get("/api/project", name=name)

    def experience(self, machine: str, name: str) -> dict | None:
        return self.get("/api/experience", machine=machine, name=name)

    def setup(self, machine: str) -> dict | None:
        return self.get("/api/setup", machine=machine)


class Local:
    """This machine only: its experiences, its projects' ended runs."""

    where = "this machine"

    def __init__(self) -> None:
        from .monitor import snapshot

        self._full = snapshot(full=True)
        self.alias = self._full["machine"]["alias"]

    def _light(self) -> dict:
        snap = dict(self._full)
        snap["experiences"] = [{k: v for k, v in e.items() if k not in ("results", "figures", "configs")}
                               for e in snap["experiences"]]
        snap.pop("setup", None)
        snap["has_setup"] = "setup" in self._full
        return snap

    def state(self) -> dict:
        now = time.time()
        return {"now": now, "space": False, "history": None,
                "machines": [{"alias": self.alias, "age": 0, "polled_age": 0, "received": now,
                              "snapshot": self._light()}]}

    def _records(self) -> list:
        from . import projects

        try:
            return projects.runs()
        except Exception:                                    # noqa: BLE001 - no project here
            return []

    def projects(self) -> dict:
        from . import projects

        declared = projects.declared()
        if declared is None:
            return {"projects": [], "history": None}
        records = self._records()
        return {"projects": [{"name": declared["name"], "description": declared["description"],
                              "focus": declared["focus"], "methods": len(declared["methods"]),
                              "runs": len(records), "running": 0,
                              "last": max((r.get("ended") or 0 for r in records), default=None)}],
                "history": None}

    def project(self, name: str) -> dict | None:
        from . import projects

        declared = projects.declared()
        if declared is None or declared["name"] != name:
            return None
        live = [{**e, "machine": self.alias, "machine_label": None, "polled_age": 0}
                for e in self._light()["experiences"] if e.get("project") == name]
        public = {k: v for k, v in declared.items() if k != "folder"}
        return {"project": public, "runs": self._records(), "live": live, "history": None}

    def experience(self, machine: str, name: str) -> dict | None:
        return next((e for e in self._full["experiences"] if e["name"] == name), None)

    def setup(self, machine: str) -> dict | None:
        return self._full.get("setup")


def source(*, local: bool = False):
    """The dashboard when one is configured (and not `local`), else this machine."""
    if local:
        return Local()
    from .monitor._share import settings

    config = settings()
    if config.get("space") or config.get("url"):
        return Remote()
    return Local()


# ── small formatting helpers ──────────────────────────────────────────────────

def num(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    size = abs(value)
    if size < 1e-3 or size >= 1e6:
        return f"{value:.3g}"
    return f"{value:.4g}"


def ago(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.0f}d ago"


def size(value) -> str:
    if value is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f}{unit}" if unit in ("B", "KB") else f"{value:.1f}{unit}"
        value /= 1024


def lower_is_better(key: str) -> bool:
    return bool(LOWER_IS_BETTER.search(key.split(".")[-2] if key.endswith((".last", ".best"))
                                       else key.split(".")[-1]))


def value(v) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return num(v)
    text = v if isinstance(v, str) else json.dumps(v)
    return text if len(text) <= 40 else text[:37] + "…"


def curve_summary(curves: dict, limit: int = 8) -> list[str]:
    """`step.name: first → last (best b @x, n pts)` per curve."""
    out = []
    for step, points in (curves or {}).items():
        names = sorted({k for p in points for k in p if k not in ("step", "time")})
        for name in names:
            series = [(p.get("step"), p[name]) for p in points if isinstance(p.get(name), (int, float))]
            if not series:
                continue
            from .projects import HIGHER_IS_BETTER

            lower, higher = bool(LOWER_IS_BETTER.search(name)), bool(HIGHER_IS_BETTER.search(name))
            if lower or higher:
                best = (min if lower else max)(series, key=lambda s: s[1])
                tail = f"best {num(best[1])} @{num(best[0])}"
            else:                                   # direction unknown: no "best"
                low, high = min(series, key=lambda s: s[1]), max(series, key=lambda s: s[1])
                tail = f"min {num(low[1])} @{num(low[0])}, max {num(high[1])} @{num(high[0])}"
            arrow = "↓" if lower else "↑" if higher else ""
            out.append(f"{step}.{name}{arrow}: {num(series[0][1])} → {num(series[-1][1])} "
                       f"({tail}, {len(series)} pts)")
    return out[:limit] + ([f"+{len(out) - limit} more curves"] if len(out) > limit else [])


def _name(machine: dict) -> str:
    info = machine["snapshot"].get("machine") or {}
    label = info.get("label")
    return f"{label} ({machine['alias']})" if label else machine["alias"]


def _status(experience: dict, machine: dict) -> str:
    if experience.get("status") == "running" and machine.get("polled_age", 0) > SILENT_AFTER:
        return "silent"
    return experience.get("status") or "new"


def _progress(experience: dict) -> str:
    progress = experience.get("progress")
    step = experience.get("current_step") or ""
    if isinstance(progress, list) and len(progress) >= 2 and progress[1]:
        label = f" {progress[2]}" if len(progress) > 2 and progress[2] else ""
        return f"{step} {num(progress[0])}/{num(progress[1])} ({100 * progress[0] / progress[1]:.0f}%){label}".strip()
    return f"{experience.get('done', 0)}/{experience.get('total', 0)} steps" + (f", at {step}" if step else "")


def _last_values(experience: dict, limit: int = 3) -> str:
    metrics = experience.get("metrics") or {}
    if not metrics:
        return ""
    step = experience.get("current_step") if experience.get("current_step") in metrics else list(metrics)[-1]
    points = metrics[step]
    names = [k for k in dict.fromkeys(k for p in points for k in p) if k not in ("step", "time")][:limit]
    parts = []
    for name in names:
        series = [p[name] for p in points if isinstance(p.get(name), (int, float))]
        if series:
            trend = "" if len(series) < 2 else ("↓" if series[-1] < series[0] else "↑")
            parts.append(f"{name} {num(series[-1])}{trend}")
    return " ".join(parts)


def _error_line(experience: dict) -> str:
    error = experience.get("error") or {}
    last = (error.get("message") or "").strip().splitlines()
    text = error.get("meaning") or (last[-1] if last else "")
    return text if len(text) <= 120 else text[:117] + "…"


def _older(version, than: str) -> bool:
    a = [int(x) for x in re.findall(r"\d+", str(version or "0"))[:3]] + [0, 0, 0]
    b = [int(x) for x in than.split(".")]
    return a[:3] < b


# ── the views ────────────────────────────────────────────────────────────────

def overview(src, *, limit: int = 12) -> str:
    state = src.state()
    listing = src.projects()
    now = state.get("now") or time.time()
    history = listing.get("history") or {}
    lines = [f"useml · {src.where} · {len(state['machines'])} machine(s)"
             + (f" · history kept: {history.get('runs', 0)} runs" if history.get("kept") else
                " · history not kept" if history else "")]
    lines.append("MACHINES")
    running, attention = [], []
    for machine in state["machines"]:
        snap = machine["snapshot"]
        info = snap.get("machine") or {}
        parts = [_name(machine), "online" if machine.get("polled_age", 0) <= SILENT_AFTER
                 else f"silent {ago(machine.get('polled_age'))}", f"useml {info.get('useml', '?')}"]
        for gpu in info.get("gpus") or []:
            parts.append(f"GPU{gpu.get('index')} {gpu.get('util')}% {size(gpu.get('mem_used'))}/{size(gpu.get('mem_total'))}")
        for disk in (info.get("disks") or [])[:1]:
            parts.append(f"disk {size(disk.get('free'))} free")
        health = snap.get("health") or {}
        if health.get("fail") or health.get("warn"):
            parts.append(f"setup: {health.get('fail', 0)} to fix, {health.get('warn', 0)} to look at")
        if _older(info.get("useml"), "1.4.0"):
            parts.append("⚠ old useml: no projects, no automatic curves")
        lines.append("  " + " · ".join(parts))
        for experience in snap.get("experiences", []):
            status = _status(experience, machine)
            ref = f"{machine['alias']}:{experience['name']}"
            updated = (now - experience["updated"]) if experience.get("updated") else None
            where = f"{experience['project']}/{experience.get('method') or '-'}" if experience.get("project") else ""
            if status in ("running", "silent"):
                running.append((updated or 0, f"  {ref} · {status} · {_progress(experience)}"
                                + (f" · {_last_values(experience)}" if _last_values(experience) else "")
                                + (f" · {where}" if where else "") + f" · {ago(updated)}"))
            elif status in ("failed", "died") and (updated is None or updated < 86400):
                attention.append((updated or 0, f"  {ref} · {status} {ago(updated)} · {_error_line(experience)}"))
    for title, items in (("RUNNING", running), ("FAILED IN THE LAST 24H", attention)):
        if items:
            items.sort(key=lambda item: item[0])
            lines.append(f"{title} ({len(items)})")
            lines += [text for _, text in items[:limit]]
            if len(items) > limit:
                lines.append(f"  +{len(items) - limit} more")
    if not running:
        lines.append("RUNNING: nothing")
    projects = listing.get("projects") or []
    lines.append("PROJECTS" if projects else "PROJECTS: none declared (useml project init)")
    for entry in projects:
        lines.append(f"  {entry['name']}" + (f" · focus {entry['focus']}" if entry.get("focus") else "")
                     + f" · {entry.get('methods', 0)} methods · {entry.get('runs', 0)} runs kept"
                     + (f" · {entry['running']} running" if entry.get("running") else "")
                     + (f" · last {ago(time.time() - entry['last'])}" if entry.get("last") else "")
                     + (f" — {entry['description']}" if entry.get("description") else ""))
    lines.append("MORE: useml info project <name> [-m method] · useml info run <name|id> · "
                 "useml info exp <machine>:<name> · useml info machine <machine>")
    return "\n".join(lines)


def _runs_of(page: dict, now: float) -> list[dict]:
    runs = [dict(r) for r in page.get("runs", [])]
    for experience in page.get("live", []):
        if experience.get("status") in ENDED:
            continue
        results = {}
        for step, points in (experience.get("metrics") or {}).items():
            for name in {k for p in points for k in p if k not in ("step", "time")}:
                series = [p[name] for p in points if isinstance(p.get(name), (int, float))]
                if series:
                    results[f"{step}.{name}.last"] = series[-1]
        runs.insert(0, {"id": "live", "live": True, "experience": experience["name"],
                        "method": experience.get("method"), "status": experience.get("status"),
                        "factors": experience.get("factors") or {}, "results": results,
                        "version": experience.get("version"),
                        "machine": experience.get("machine_label") or experience.get("machine"),
                        "ended": experience.get("updated"), "progress": _progress(experience)})
    return runs


def _main_result(runs: list[dict], wanted: str | None = None) -> str | None:
    from .projects import main_result

    keys = [k for r in runs for k in (r.get("results") or {}) if trusted(k)]
    if wanted and wanted in keys:
        return wanted
    return main_result(keys)


def trusted(key: str) -> bool:
    """False for a curve's "best" whose direction is unknown (records before 1.6 picked the
    maximum of any curve)."""
    if not key.endswith(".best"):
        return True
    from .projects import HIGHER_IS_BETTER

    leaf = key[:-5].rsplit(".", 1)[-1]
    return bool(LOWER_IS_BETTER.search(leaf) or HIGHER_IS_BETTER.search(leaf))


def _results(run: dict) -> dict:
    return {k: v for k, v in (run.get("results") or {}).items() if trusted(k)}


def _factors(run: dict) -> dict:
    """The run's factors without plumbing (also for runs kept before 1.4.2)."""
    from .projects import is_plumbing

    return {k: v for k, v in (run.get("factors") or {}).items() if not is_plumbing(k, v)}


def _association(runs, factor, result):
    pairs = [(r["factors"][factor], r["results"][result]) for r in runs
             if factor in (r.get("factors") or {}) and isinstance((r.get("results") or {}).get(result), (int, float))
             and r["factors"][factor] is not None]
    if len(pairs) < 3:
        return None
    ys = [p[1] for p in pairs]
    if all(isinstance(p[0], (int, float)) for p in pairs):
        xs = [float(p[0]) for p in pairs]
        if len(set(xs)) < 2:
            return None
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        return ("r", sxy / math.sqrt(sxx * syy), len(pairs)) if sxx and syy else None
    groups: dict = {}
    for x, y in pairs:
        groups.setdefault(json.dumps(x), []).append(y)
    if len(groups) < 2:
        return None
    mean = sum(ys) / len(ys)
    total = sum((y - mean) ** 2 for y in ys)
    if not total:
        return None
    between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
    return ("η", math.sqrt(between / total), len(pairs))


def _distinct(kept: list[dict]) -> list[dict]:
    """One row per (experience, declaration): a run repeated identically is counted, the
    newest is shown."""
    groups: dict = {}
    for run in sorted(kept, key=lambda r: r.get("ended") or 0):
        key = (run["experience"], run.get("run") or run["id"])
        if key in groups:
            groups[key] = {**run, "times": groups[key]["times"] + 1}
        else:
            groups[key] = {**run, "times": 1}
    return list(groups.values())


def _best(runs: list[dict], key: str | None):
    scored = [r for r in runs if key and isinstance((r.get("results") or {}).get(key), (int, float))]
    if not scored:
        return None
    return (min if lower_is_better(key) else max)(scored, key=lambda r: r["results"][key])


def diff(before: dict, after: dict, *, limit: int = 5) -> list[str]:
    """What changed from one run to another: factors (old→new) and the steps whose code
    changed."""
    a, b = _factors(before), _factors(after)
    changes = [f"{k} {value(a.get(k, '-'))}→{value(b.get(k, '-'))}"
               for k in sorted(set(a) | set(b)) if json.dumps(a.get(k)) != json.dumps(b.get(k))]
    code_a = {s["name"]: s.get("code") for s in before.get("steps") or []}
    code = [s["name"] for s in after.get("steps") or [] if s.get("code") and code_a.get(s["name"]) not in (None, s.get("code"))]
    out = changes[:limit] + ([f"+{len(changes) - limit} more"] if len(changes) > limit else [])
    if code:
        out.append("code of " + ",".join(code))
    return out


def evolution(runs: list[dict], main: str, *, prefix: str = "") -> list[str]:
    """The method's record over time: each run that beat the best so far, by how much, and
    what it changed from the previous best."""
    lower = lower_is_better(main)
    timeline = sorted((r for r in runs if isinstance((r.get("results") or {}).get(main), (int, float))),
                      key=lambda r: r.get("ended") or 0)
    if len(timeline) < 2:
        return []
    best, records, tried = None, [], 0
    for run in timeline:
        score = run["results"][main]
        if best is None or (score < best["results"][main] if lower else score > best["results"][main]):
            gain = "" if best is None else f" ({'+' if score - best['results'][main] > 0 else ''}{num(score - best['results'][main])} after {tried} other run{'s' if tried != 1 else ''})"
            changed = diff(best, run) if best is not None else []
            short = run["experience"][len(prefix):] if run["experience"].startswith(prefix) else run["experience"]
            date = time.strftime("%m-%d", time.localtime(run["ended"])) if run.get("ended") else "?"
            records.append(f"  {date} {short} {main}={num(score)}{gain}" + (f" · changed: {'; '.join(changed)}" if changed else "")
                           + (f" · note: {run['note'][:90]}" if run.get("note") else ""))
            best, tried = run, 0
        else:
            tried += 1
    lines = [f"EVOLUTION of {main} (each new best, what it changed from the previous one):"] + records[-8:]
    if tried:
        lines.append(f"  since the last best: {tried} run{'s' if tried != 1 else ''}, none better")
    return lines


GENERIC = {"metrics", "metric", "results", "result", "scores", "score", "eval", "evaluation",
           "test", "val", "valid", "report", "summary", "final", "output", "outputs"}


def bench_of(run: dict, key: str) -> tuple[str, str] | None:
    """(benchmark, metric) of a result key — the same rule as the dashboard: the metric is the
    last part; the benchmark is the middle (`report.<benchmark>.<metric>`), or, when the
    middle says nothing (`report.metrics.f1`), the run's name without its method and its
    last part (the model). Training curves and counts are not benchmark scores."""
    from .projects import COUNT

    if key.endswith((".last", ".best")):
        return None
    parts = key.split(".")
    if len(parts) < 2 or COUNT.search(parts[-1]):
        return None
    middle = [p for p in parts[1:-1] if p.lower() not in GENERIC]
    if middle:
        return ".".join(middle), parts[-1]
    method = run.get("method") or ""
    short = run["experience"][len(method) + 1:] if run["experience"].startswith(f"{method}/") else run["experience"]
    names = short.split("/")
    return ("/".join(names[:-1]) if len(names) > 1 else parts[0]), parts[-1]


def scores(run: dict) -> dict:
    """{(benchmark, metric): value} of a run."""
    out = {}
    for key, number in (run.get("results") or {}).items():
        where = bench_of(run, key) if isinstance(number, (int, float)) else None
        if where:
            out[where] = number
    return out


def benchmarks(src, name: str) -> str:
    """The project's benchmarks: models, metrics, the best model on the main metric."""
    page = src.project(name)
    if page is None:
        raise InfoError(f"no project {name!r}")
    runs = _distinct([r for r in page["runs"]])
    table: dict = {}
    for run in runs:
        for (bench, metric), value_ in scores(run).items():
            entry = table.setdefault(bench, {"metrics": {}, "models": set(), "methods": set()})
            entry["metrics"][metric] = entry["metrics"].get(metric, 0) + 1
            entry["models"].add(run["experience"])
            entry["methods"].add(run.get("method") or "(unassigned)")
    if not table:
        return f"PROJECT {name}: no benchmark score kept yet"
    lines = [f"BENCHMARKS of {name} ({len(table)}; benchmark · models · methods · best on its main metric)"]
    for bench in sorted(table, key=lambda b: (-len(table[b]["models"]), b)):
        entry = table[bench]
        metric = _bench_main(entry["metrics"])
        top = _leaderboard(runs, bench, metric)[:1]
        lines.append(f"  {bench} · {len(entry['models'])} models · {','.join(sorted(entry['methods']))}"
                     + (f" · best {metric}={num(scores(top[0])[(bench, metric)])} ({_model(top[0], bench)})" if top else ""))
    lines.append(f"MORE: useml info bench {name} <benchmark>")
    return "\n".join(lines)


def _bench_main(metrics: dict) -> str:
    from .projects import MEASURE

    return sorted(metrics, key=lambda m: (not MEASURE.search(m), -metrics[m], m.lower()))[0]


def _model(run: dict, bench: str) -> str:
    method = run.get("method") or ""
    name = run["experience"][len(method) + 1:] if run["experience"].startswith(f"{method}/") else run["experience"]
    return name[len(bench) + 1:] if name.startswith(f"{bench}/") else name


def _leaderboard(runs: list[dict], bench: str, metric: str) -> list[dict]:
    """One run per model (experience): its best on (bench, metric), best first."""
    lower = lower_is_better(metric)
    best: dict = {}
    for run in runs:
        v = scores(run).get((bench, metric))
        if not isinstance(v, (int, float)):
            continue
        kept = best.get(run["experience"])
        if kept is None or (v < scores(kept)[(bench, metric)] if lower else v > scores(kept)[(bench, metric)]):
            best[run["experience"]] = run
    return sorted(best.values(), key=lambda r: scores(r)[(bench, metric)], reverse=not lower)


def bench(src, name: str, benchmark: str, *, metrics: int = 6) -> str:
    """A benchmark's sheet: one row per model at its best run, its metrics, by method."""
    page = src.project(name)
    if page is None:
        raise InfoError(f"no project {name!r}")
    runs = _distinct(page["runs"])
    counts: dict = {}
    for run in runs:
        for (b, metric) in scores(run):
            if b == benchmark:
                counts[metric] = counts.get(metric, 0) + 1
    if not counts:
        known = sorted({b for run in runs for (b, _) in scores(run)})
        raise InfoError(f"no benchmark {benchmark!r} in {name}; e.g. {known[:6]}")
    from .projects import MEASURE

    shown = sorted(counts, key=lambda m: (not MEASURE.search(m), -counts[m], m.lower()))[:metrics]
    main = shown[0]
    models = _leaderboard(runs, benchmark, main)
    best = {m: sorted({scores(r)[(benchmark, m)] for r in models if (benchmark, m) in scores(r)},
                      reverse=not lower_is_better(m))[:2] for m in shown}
    lines = [f"BENCHMARK {benchmark} · project {name} · {len(models)} models at their best run on {main} "
             f"({'lower' if lower_is_better(main) else 'higher'} is better) · *best · _second"]
    lines.append("  model | method | " + " | ".join(shown) + " | version")
    for run in models:
        values = scores(run)
        cells = []
        for m in shown:
            v = values.get((benchmark, m))
            mark = "" if v is None else "*" if v == best[m][0] else "_" if len(best[m]) > 1 and v == best[m][1] else ""
            cells.append("-" if v is None else f"{mark}{num(v)}")
        lines.append(f"  {_model(run, benchmark)}{' ×' + str(run['times']) if run.get('times', 1) > 1 else ''} | "
                     f"{run.get('method') or '-'} | " + " | ".join(cells) + f" | v{run.get('version')}")
    if len(counts) > metrics:
        lines.append(f"  ({len(counts) - metrics} more metrics; --metrics {len(counts)})")
    return "\n".join(lines)


def _params(n: int) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else \
        f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def _arch_index(node: dict, path: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    out[path] = node
    for child in node.get("children") or []:
        name = re.sub(r"^\d+–\d+$", "*", child["name"])
        _arch_index(child, f"{path}/{re.sub(r'….*$', '…', name)}", out)
    return out


def architecture(kept: list[dict], step: str, best: dict, *, depth: int = 3) -> list[str]:
    """The model(s) a step returned, as an indented tree of blocks: `name Class ×n (params)
    extra`; `*` marks a block that differs across runs (class, size, repeats, presence)."""
    mine = next((s.get("architecture") for s in best.get("steps") or [] if s["name"] == step), None)
    if not mine:
        return []
    others = [s.get("architecture") for r in kept for s in r.get("steps") or []
              if s["name"] == step and s.get("architecture")]
    lines = []
    for name, tree in mine.items():
        indexes = [_arch_index(o.get(name) or {"children": []}) for o in others]

        def text(node):
            return f"{node.get('class', '').split('.')[-1]}×{node.get('repeat', 1)} {node.get('extra', '')} {node.get('params')}"

        def walk(node, path, level, label):
            seen = {text(i[path]) if path in i else "absent" for i in indexes}
            star = "*" if len(seen) > 1 else ""
            extra = f" ({node['extra'][:60]})" if node.get("extra") else ""
            lines.append(f"    {'  ' * level}{star}{label} {node.get('class', '').split('.')[-1]}"
                         + (f" ×{node['repeat']}" if node.get("repeat") else "")
                         + f" {_params(node.get('params', 0))}{extra}")
            if level < depth:
                for child in node.get("children") or []:
                    key = re.sub(r"^\d+–\d+$", "*", child["name"])
                    walk(child, f"{path}/{re.sub(r'….*$', '…', key)}", level + 1, child["name"])
            elif node.get("children"):
                lines.append(f"    {'  ' * (level + 1)}… {len(node['children'])} blocks below")
        lines.append(f"    model returned: {name}" + (f" (in {len(others)} runs)" if len(others) > 1 else ""))
        walk(tree, "", 0, name)
    return lines


def schema(runs: list[dict], method: str, main: str | None) -> list[str]:
    """The method as data → steps (their components, as a tree) → benchmarks, with the
    values of its best run; `*name=value [tried]` for components that vary."""
    kept = [r for r in runs if r.get("steps")]
    if not kept:
        return []
    kept.sort(key=lambda r: r.get("ended") or 0, reverse=True)
    order: list = []
    for run in kept:
        for step in run["steps"]:
            if step["name"] not in order:
                order.append(step["name"])
    scored = [r for r in kept if main and isinstance((r.get("results") or {}).get(main), (int, float))]
    # The best among runs that have settings for these steps (not a reference run without them).
    count = {id(r): sum(1 for k in _factors(r) if k.split(".")[0] in order) for r in kept}
    complete = [r for r in kept if count[id(r)] > 0]
    pool = [r for r in scored if r in complete] or scored
    best = (min if main and lower_is_better(main) else max)(pool, key=lambda r: r["results"][main]) if pool else (complete or kept)[0]
    steps = {s["name"]: s for s in best["steps"]}
    reads_known = any("reads" in s for r in kept for s in r["steps"])
    short = best["experience"][len(method) + 1:] if best["experience"].startswith(f"{method}/") else best["experience"]
    lines = [f"SCHEMA (values of {'the best run ' if scored else 'the latest run '}{short}; *varies across runs [values tried]):"]
    data = sorted({d for r in kept for s in r["steps"] for d in s.get("inputs") or []})
    if data:
        lines.append(f"  data: {', '.join(data)}")
    for index, name in enumerate(order):
        step = steps.get(name) or next(s for r in kept for s in r["steps"] if s["name"] == name)
        reads = step.get("reads") if reads_known else ([order[index - 1]] if index else [])
        codes = {s.get("code") for r in kept for s in r["steps"] if s["name"] == name and s.get("code")}
        lines.append(f"  {name} ({step['function']})" + (f" ← {', '.join(reads)}" if reads else "")
                     + (f" · {len(codes)} code versions" if len(codes) > 1 else ""))
        having = [r for r in kept if any(k.startswith(f"{name}.") for k in _factors(r))]
        keys = sorted({k for r in having for k in _factors(r) if k.startswith(f"{name}.")})
        groups: dict = {}
        for key in keys:
            tried = {json.dumps(r["factors"].get(key)) for r in having if r["factors"].get(key) is not None}
            text = f"{key[len(name) + 1:]}={value(_factors(best).get(key, '-'))}"
            if len(tried) > 1:
                text = "*" + text + " [" + ", ".join(value(json.loads(t)) for t in sorted(tried)) + "]"
            head = key[len(name) + 1:].split(".")[0]
            groups.setdefault(head, []).append(text)
        for head, items in groups.items():
            varying = [t for t in items if t.startswith("*")]
            if len(items) > 6 and not varying:
                lines.append(f"    {head}: {len(items)} settings, none varies")
            else:
                shown = varying + [t for t in items if not t.startswith("*")][:6 - len(varying)]
                lines.append(f"    {' · '.join(shown)}" + (f" +{len(items) - len(shown)}" if len(items) > len(shown) else ""))
        lines += architecture(kept, name, best)
    benches = {b for (b, _) in scores(best)}
    if benches:
        lines.append(f"  → {len(benches)} benchmarks measured (useml info project <p> --benchmarks)")
    return lines


def family(experience: str, method: str) -> str:
    """The first part of a name under its method: `wide/v2/w8` → `v2`; a name alone → `·`."""
    short = experience[len(method) + 1:] if experience.startswith(f"{method}/") else experience
    parts = short.split("/")
    return parts[0] if len(parts) > 1 else "·"


def project(src, name: str, *, method: str | None = None, result: str | None = None,
            limit: int = 25, max_factors: int = 10, family_name: str | None = None) -> str:
    """Without `method`: the summary (methods, running, latest runs). With it: the method's
    runs table (its own factors), links between factors and results, versions."""
    page = src.project(name)
    if page is None:
        names = [p["name"] for p in (src.projects().get("projects") or [])]
        raise InfoError(f"no project {name!r}; known: {names or 'none'}")
    now = time.time()
    declared = page["project"]
    runs = _runs_of(page, now)
    for run_ in runs:
        run_["factors"] = _factors(run_)
    of = lambda r: r.get("method") or "(unassigned)"                       # noqa: E731
    lines = [f"PROJECT {declared['name']}" + (f" — {declared['description']}" if declared.get("description") else "")]
    if declared.get("focus"):
        lines.append(f"focus: {declared['focus']}")
    used: dict = {}
    for run in runs:
        for dataset in run.get("datasets") or []:
            used[dataset.split("[")[0]] = used.get(dataset.split("[")[0], 0) + 1
    names = list(dict.fromkeys(list(declared.get("datasets") or []) + list(used)))
    if names and not method:
        shown = names[:12]
        lines.append("datasets: " + ", ".join(f"{d} ({used.get(d, 0)})" for d in shown)
                     + (f" +{len(names) - 12}" if len(names) > 12 else ""))
    methods = list(dict.fromkeys(list((declared.get("methods") or {})) + [of(r) for r in runs]))
    if not method:
        lines.append("METHODS (each ranked by its own main result)")
        for m in methods:
            info = (declared.get("methods") or {}).get(m, {})
            mine = [r for r in runs if of(r) == m and not r.get("live")]
            live = [r for r in runs if of(r) == m and r.get("live")]
            distinct = _distinct(mine)
            main = _main_result(mine, info.get("result"))
            best = _best(distinct, main)
            lines.append(f"  {'▶ ' if m == declared.get('focus') else ''}{m}"
                         + (f" [{info['status']}]" if info.get("status") else "")
                         + f" · {len(distinct)} runs" + (f" ({len(mine)} with repeats)" if len(mine) != len(distinct) else "")
                         + f" · {len({r.get('version') for r in mine})} versions"
                         + (f" · {len(live)} running" if live else "")
                         + (f" · best {main}={num(best['results'][main])} ({best['experience']})" if best else "")
                         + ("" if info.get("result") or not main else " [guessed; set: useml project method set "
                            f"{m} --result <key>]")
                         + (f" — {info['description']}" if info.get("description") else ""))
    else:
        if method not in methods:
            raise InfoError(f"no method {method!r} in {name}; known: {methods}")
        runs = [r for r in runs if of(r) == method]
        info = (declared.get("methods") or {}).get(method, {})
        lines.append(f"METHOD {method}" + (f" [{info['status']}]" if info.get("status") else "")
                     + (f" — {info['description']}" if info.get("description") else ""))
    live = [r for r in runs if r.get("live")]
    if live:
        lines.append(f"RUNNING ({len(live)})")
        for run in live[:limit]:
            results = " ".join(f"{k.rsplit('.', 2)[-2]} {num(v)}" for k, v in list(run["results"].items())[:3])
            lines.append(f"  {run['machine']}:{run['experience']} · {run['progress']}" + (f" · {results}" if results else ""))
    kept = [r for r in runs if not r.get("live")]
    if not kept:
        lines.append("RUNS KEPT: none yet")
        return "\n".join(lines)
    distinct = _distinct(kept)
    if method:
        # Families: runs of one method that share nothing are compared within their family.
        families: dict = {}
        for run in distinct:
            families.setdefault(family(run["experience"], method), []).append(run)
        if len(families) > 1:
            order = sorted(families, key=lambda f: -max(r.get("ended") or 0 for r in families[f]))
            chosen = family_name if family_name in families else next(
                (f for f in order if any(r.get("results") for r in families[f])), order[0])
            lines.append("FAMILIES: " + " · ".join(f"{f} ({len(families[f])})" for f in order)
                         + f" — showing {chosen}; --family <name> for another")
            distinct = families[chosen]
            kept = [r for r in kept if family(r["experience"], method) == chosen]
    if not method:
        recent = sorted(distinct, key=lambda r: r.get("ended") or 0, reverse=True)[:8]
        lines.append(f"LATEST RUNS ({len(distinct)} distinct)")
        for run in recent:
            main = _main_result([r for r in kept if of(r) == of(run)],
                                ((declared.get("methods") or {}).get(of(run)) or {}).get("result"))
            score = f" · {main}={num(run['results'][main])}" if main and isinstance((run.get("results") or {}).get(main), (int, float)) else ""
            lines.append(f"  {run['id'][:8]} {run['experience']} · {of(run)} · {run['status']}{score}"
                         + (f" · ×{run['times']}" if run["times"] > 1 else "")
                         + f" · {ago(now - run['ended']) if run.get('ended') else '?'}")
        lines.append(f"MORE: useml info project {name} -m <method> (its runs table) · useml info run <id>")
        return "\n".join(lines)
    main = _main_result(kept, result or ((declared.get("methods") or {}).get(method) or {}).get("result"))
    keys = sorted({k for r in distinct for k in r["factors"]})
    def differs(key):
        # Among the runs that have this factor's step (a run without the step is another
        # kind of experience); a run with the step but not the factor used its default.
        step = key.split(".")[0] + "."
        having = [r for r in distinct if any(k.startswith(step) for k in r["factors"])]
        return len({json.dumps(r["factors"].get(key)) for r in having}) > 1
    varying = [k for k in keys if differs(k)]
    fixed = [k for k in keys if k not in varying]
    hidden = 0
    if len(varying) > max_factors:
        # Keep the factors with the fewest distinct values first: choices, not identifiers.
        def real(k):    # two runs set different values (not only one leaving it out)
            return len({json.dumps(r["factors"][k]) for r in distinct if r["factors"].get(k) is not None}) > 1
        varying.sort(key=lambda k: (not real(k), -sum(1 for r in distinct if r["factors"].get(k) is not None),
                                    len({json.dumps(r["factors"].get(k)) for r in distinct}), k))
        real_ones = [k for k in varying if real(k)] or varying
        hidden, varying = len(varying) - min(max_factors, len(real_ones)), sorted(real_ones[:max_factors])
    results = [main] if main else []
    from .projects import main_result
    others = [k for k in sorted({k for r in distinct for k in _results(r)} - set(results),
                                key=lambda k: (main_result([k]) != k, k.endswith(".last"), k))]
    results += others[:3]
    if main:
        lower = lower_is_better(main)
        distinct.sort(key=lambda r: (not isinstance((r.get("results") or {}).get(main), (int, float)),
                                     (r["results"][main] if lower else -r["results"][main])
                                     if isinstance((r.get("results") or {}).get(main), (int, float)) else 0))
    lines.append(f"RUNS ({len(distinct)} distinct" + (f", {len(kept)} with repeats" if len(kept) != len(distinct) else "")
                 + (f"; best {main} first, {'lower' if lower_is_better(main) else 'higher'} is better)" if main else ")"))
    if fixed:
        sample = distinct[0]
        lines.append("  same in all: " + " · ".join(f"{k}={value(sample['factors'].get(k))}" for k in fixed[:12])
                     + (f" +{len(fixed) - 12}" if len(fixed) > 12 else ""))
    if hidden:
        lines.append(f"  ({hidden} more factors differ; --factors {len(varying) + hidden} shows them)")
    lines.append("  id | run | status | " + " | ".join(results + varying) + " | version | ended")
    for run in distinct[:limit]:
        short = run["experience"][len(method) + 1:] if run["experience"].startswith(f"{method}/") else run["experience"]
        cells = [run["id"][:8], short + (f" ×{run['times']}" if run["times"] > 1 else ""), run["status"]]
        cells += [num(run["results"][k]) if isinstance((run.get("results") or {}).get(k), (int, float)) else "-" for k in results]
        cells += [value(run["factors"].get(k, "-")) for k in varying]
        cells += [f"v{run.get('version')}", ago(now - run["ended"]) if run.get("ended") else "?"]
        lines.append("  " + " | ".join(cells))
    if len(distinct) > limit:
        lines.append(f"  +{len(distinct) - limit} more runs (--limit {len(distinct)})")
    links = []
    for factor in varying:
        for res in results[:2]:
            found = _association(distinct, factor, res)
            if found and found[1] ** 2 >= 0.25:
                kind, strength, n = found
                links.append((abs(strength), f"{factor} ~ {res}: {kind}={'+' if kind == 'r' and strength > 0 else ''}{strength:.2f} (n={n})"))
    if links:
        links.sort(reverse=True)
        lines.append("STRONGEST LINKS (r: −1..+1 for numbers, η: 0..1 for choices; a hint, not a proof): "
                     + "; ".join(text for _, text in links[:6]))
    lines += schema(distinct, method, main)
    if main:
        lines += evolution(distinct, main, prefix=f"{method}/")
    versions: dict = {}
    for run in sorted(kept, key=lambda r: r.get("ended") or 0):
        versions.setdefault(run.get("version"), []).append(run)
    if len(versions) > 1:
        chain, previous = [], None
        for version, group in versions.items():
            changed = ""
            if previous is not None:
                before = {s["name"]: s.get("code") for s in previous[0].get("steps") or []}
                names_changed = [s["name"] for s in group[0].get("steps") or [] if before.get(s["name"]) != s.get("code")]
                changed = f", changed: {','.join(names_changed)}" if names_changed else ""
            chain.append(f"v{version} ({len(group)} runs{changed})")
            previous = group
        shown = chain if len(chain) <= 8 else chain[:2] + [f"… {len(chain) - 4} more …"] + chain[-2:]
        lines.append("VERSIONS (code of the steps, oldest first): " + " → ".join(shown))
    lines.append("MORE: useml info run <id> [--code] [--configs]")
    return "\n".join(lines)


def _find_run(src, ref: str, project_name: str | None):
    listing = src.projects().get("projects") or []
    names = [project_name] if project_name else [p["name"] for p in listing]
    matches = []
    for name in names:
        page = src.project(name)
        for record in (page or {}).get("runs", []):
            if record["id"].startswith(ref) or record["experience"] == ref:
                matches.append(record)
    if not matches:
        raise InfoError(f"no kept run {ref!r} (an id prefix or an experience name)"
                        + ("; a running experience: useml info exp <machine>:<name>" if ":" not in ref else ""))
    return sorted(matches, key=lambda r: r.get("ended") or 0)[-1]


def run(src, ref: str, *, project_name: str | None = None, code: bool = False,
        configs: bool = False) -> str:
    record = _find_run(src, ref, project_name)
    ended = record.get("ended")
    git = record.get("git") or {}
    lines = [f"RUN {record['experience']} (id {record['id']}) · project {record.get('project')} · "
             f"method {record.get('method') or '-'} · {record['status']}"
             + (f" · ended {time.strftime('%Y-%m-%d %H:%M', time.localtime(ended))} ({ago(time.time() - ended)})" if ended else "")
             + (f" · took {num(record['seconds'])}s" if record.get("seconds") else "")
             + f" · v{record.get('version')}"
             + (f" · commit {git.get('commit', '')[:8]}{'*' if git.get('dirty') else ''}" if git else "")
             + (f" · on {record['machine']}" if record.get("machine") else "")]
    if record.get("note"):
        lines.append(f"note: {record['note']}")
    if record.get("tags"):
        lines.append("tags: " + " · ".join(f"{k}={value(v)}" for k, v in record["tags"].items()))
    if record.get("error") and (record["error"].get("meaning") or record["error"].get("message")):
        lines.append("ERROR: " + (record["error"].get("meaning") or ""))
        message = (record["error"].get("message") or "").strip().splitlines()[-6:]
        lines += [f"  {line}" for line in message]
    if _factors(record):
        lines.append("FACTORS: " + " · ".join(f"{k}={value(v)}" for k, v in _factors(record).items()))
    if _results(record):
        shown = {k: v for k, v in _results(record).items() if not k.endswith(".last") or k[:-5] + ".best" not in record["results"]}
        lines.append("RESULTS: " + " · ".join(f"{k}={num(v)}" for k, v in shown.items()))
    curves = curve_summary(record.get("curves") or {})
    if curves:
        lines.append("CURVES (first → last):")
        lines += [f"  {line}" for line in curves]
    steps = record.get("steps") or []
    if steps:
        lines.append("STEPS: " + " · ".join(f"{s['name']} {('✓ ' + num(s['seconds']) + 's') if s.get('done') else '✗'} ({s['function']})" for s in steps))
    if record.get("datasets"):
        lines.append("DATA: " + ", ".join(record["datasets"]))
    for config in record.get("configs") or []:
        head = f"CONFIG {config['name']} cfg:{config.get('hash')}" + (
            f" overrides {json.dumps(config.get('overrides'))}" if config.get("overrides") else "")
        lines.append(head)
        if configs and config.get("yaml"):
            lines += [f"  {line}" for line in config["yaml"].splitlines()]
    if code and record.get("declaration"):
        lines.append("DECLARATION (python, to start an alternative):")
        lines += [f"  {line}" for line in record["declaration"].splitlines()]
    hints = [h for h, on in (("--code", not code), ("--configs", not configs and record.get("configs"))) if on]
    if hints:
        lines.append("MORE: " + " ".join(hints))
    return "\n".join(lines)


def _split_ref(ref: str) -> tuple[str | None, str]:
    if ":" in ref:
        machine, name = ref.split(":", 1)
        return machine, name
    return None, ref


def experience(src, ref: str, *, log: int = 15) -> str:
    state = src.state()
    machine_name, name = _split_ref(ref)
    found = [(m, e) for m in state["machines"] for e in m["snapshot"].get("experiences", [])
             if e["name"] == name and (machine_name is None or machine_name in
                                        (m["alias"], (m["snapshot"].get("machine") or {}).get("label")))]
    if not found:
        raise InfoError(f"no experience {ref!r} reported now (<machine>:<name>; see useml info)")
    if len(found) > 1:
        raise InfoError(f"{name!r} is on several machines: " + ", ".join(f"{m['alias']}:{name}" for m, _ in found))
    machine, light = found[0]
    heavy = src.experience(machine["alias"], name) or {}
    data = {**heavy, **light}
    status = _status(data, machine)
    lines = [f"EXPERIENCE {machine['alias']}:{name} · {status} · {_progress(data)}"
             + (f" · updated {ago(state['now'] - data['updated'])}" if data.get("updated") else "")
             + (f" · project {data['project']}/{data.get('method') or '-'}" if data.get("project") else "")]
    if data.get("error") and (data["error"].get("meaning") or data["error"].get("message")):
        lines.append("ERROR: " + (data["error"].get("meaning") or ""))
        lines += [f"  {line}" for line in (data["error"].get("message") or "").strip().splitlines()[-8:]]
    if data.get("steps"):
        lines.append("STEPS: " + " · ".join(
            f"{s['name']} " + ("✓ " + num(s['seconds'] or 0) + "s" if s.get("done") else
                               "▶" if s["name"] == data.get("current_step") else "·")
            + f" ({s['function']})" for s in data["steps"]))
    if _factors(data):
        lines.append("FACTORS: " + " · ".join(f"{k}={value(v)}" for k, v in _factors(data).items()))
    curves = curve_summary(data.get("metrics") or {})
    if curves:
        lines.append("CURVES (first → last):")
        lines += [f"  {line}" for line in curves]
    if data.get("results"):
        for step, result in data["results"].items():
            text = json.dumps(result)
            lines.append(f"RESULT {step}: " + (text if len(text) <= 300 else text[:297] + "…"))
    if data.get("datasets"):
        lines.append("DATA: " + ", ".join(data["datasets"]))
    log_lines = [line for line in (data.get("log") or []) if line.strip()][-log:] if log else []
    if log_lines:
        lines.append(f"LOG (last {len(log_lines)} lines):")
        lines += [f"  {line[:200]}" for line in log_lines]
    elif log:
        lines.append("LOG: empty (a run in a notebook before useml 1.4 wrote none)")
    return "\n".join(lines)


def machine(src, ref: str) -> str:
    state = src.state()
    found = [m for m in state["machines"] if ref in (m["alias"], (m["snapshot"].get("machine") or {}).get("label"))]
    if not found:
        raise InfoError(f"no machine {ref!r}; known: " + ", ".join(m["alias"] for m in state["machines"]))
    entry = found[0]
    info = entry["snapshot"].get("machine") or {}
    lines = [f"MACHINE {_name(entry)} · useml {info.get('useml', '?')} · "
             + ("online" if entry.get("polled_age", 0) <= SILENT_AFTER else f"silent {ago(entry.get('polled_age'))}")]
    for gpu in info.get("gpus") or []:
        lines.append(f"  GPU{gpu.get('index')} {gpu.get('name')} · {gpu.get('util')}% · {size(gpu.get('mem_used'))}/{size(gpu.get('mem_total'))}")
    if info.get("ram"):
        lines.append(f"  RAM {size(info['ram']['total'] - info['ram']['available'])}/{size(info['ram']['total'])}")
    for disk in info.get("disks") or []:
        lines.append(f"  disk {disk.get('label')}: {size(disk.get('free'))} free of {size(disk.get('total'))}")
    counts: dict = {}
    for experience in entry["snapshot"].get("experiences", []):
        counts[experience.get("status")] = counts.get(experience.get("status"), 0) + 1
    lines.append("  experiences: " + (", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none"))
    setup = src.setup(entry["alias"]) or {}
    checks = [c for c in setup.get("checks") or [] if c.get("level") != "ok"]
    if setup.get("checks"):
        lines.append(f"HEALTH ({len(checks)} to look at)" if checks else "HEALTH: everything in place")
        for check in checks:
            lines.append(f"  {'✗' if check['level'] == 'fail' else '⚠'} {check['message']}"
                         + (f" → {check['fix']}" if check.get("fix") else ""))
    elif _older(info.get("useml"), "1.3.0"):
        lines.append("HEALTH: unknown (useml older than 1.3)")
    return "\n".join(lines)


__all__ = ["source", "Remote", "Local", "overview", "project", "run", "experience", "machine",
           "curve_summary", "InfoError"]
