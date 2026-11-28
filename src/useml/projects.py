"""useml.projects — a project, its components, and a record of every run.

A PROJECT is declared once, in the repository, so it is the same on every computer that
clones it:

    # useml.toml (or [tool.useml.project] in pyproject.toml)
    [project]
    name = "robust-vision"
    description = "Image classifiers that survive corrupted inputs."
    focus = "wide"                         # the method being worked on now
    datasets = ["cifar10", "cifar10-c"]    # the datasets of the project (more are found in runs)

    [project.methods.baseline]
    description = "ResNet-50, standard training."
    status = "reference"                   # free text: idea, active, reference, dropped…

    [project.methods.wide]
    description = "Wider blocks, same budget."
    experiences = ["vision/wide*"]         # which experiences try it (globs)

An experience belongs to a method by `exp.experience(name, method="wide")`, else by the
`experiences` globs of a method, else to none ("unassigned").

A RUN is one execution of an experience with one declaration (its fingerprint). When it
ends, `record(experience)` describes it:

    factors   what varies from one run to another: the steps' plain arguments
              (`train.epochs: 10`), the config each step reads and its overrides
              (`train.model: resnet50`, `train.model.width: 2`), the datasets, the tags
    results   every number of the steps' JSON outputs (`evaluate.accuracy: 0.91`), and the
              last / best value of every curve (`train.loss.last`)
    version   a hash of the steps' code (two runs with the same version ran the same code;
              their differences are only their factors) and the git commit

A record never holds code: only names (`module:function`), numbers, configs (YAML), and
masked texts. `useml project show` / `useml project export` read them on this machine;
the monitor keeps them in a private dataset, for every machine (see `useml docs projects`).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import time
from pathlib import Path

FORMAT = "useml-run/1"
#: Not a choice but plumbing: where a thing is and how to check it (`…fingerprint`, `…_path`).
PLUMBING_KEY = re.compile(r"(fingerprint|checksum|digest|hash|_?path|_?dir|_?file|_?root|"
                          r"_?url|_?folder|cache)$", re.I)
PLUMBING_VALUE = re.compile(r"^(?:[/~]|<(?:project|useml|data|datasets|redacted)>|[0-9a-f]{12,}$|"
                            r"[a-z]+://)", re.I)
#: Names of quality measures, and of counts — to pick the result a project is ranked by.
MEASURE = re.compile(r"(acc|accuracy|f1|auc|auroc|map\b|mrr|hits|ndcg|bleu|rouge|meteor|"
                     r"score|r2|corr|pearson|spearman|precision|recall|iou|dice|loss|error|"
                     r"perplexity|ppl|wer|cer|fid|rmse|mae|mse|reward|win)", re.I)
COUNT = re.compile(r"(^|[._/])(n|num|count|size|total|len|samples?|items?|rows?|version|"
                   r"seed|id|epoch|epochs|step|steps|index|year)([._]|$)|(^|[._/])n_\w+$|"
                   r"_version$|_count$", re.I)
MAX_CURVE_POINTS = 120
MAX_FACTOR_TEXT = 120
MAX_REPORT_BYTES = 64 * 1024 * 1024     # a report is read whole up to this size
#: Curves whose direction is known: lower is better / higher is better. Any other name
#: gets no "best" (a decreasing `assign` may be a loss, or not).
HIGHER_IS_BETTER = re.compile(r"(acc|accuracy|f1|auc|auroc|map\b|mrr|hits|ndcg|bleu|rouge|"
                              r"meteor|score|r2|corr|pearson|spearman|precision|recall|iou|"
                              r"dice|reward|win)", re.I)
LOWER_IS_BETTER = re.compile(r"(loss|error|err\b|perplexity|ppl|mse|mae|rmse|wer|cer|fid|"
                             r"time|seconds|latency|memory|regret|distance)", re.I)


# ── the declaration ───────────────────────────────────────────────────────────

def declared(start=None) -> dict | None:
    """The `[project]` of the nearest useml settings above `start`, normalised; None when
    there is none."""
    from ._project import section

    folder, table = section("project", start)
    if folder is None or not table:
        return None
    name = str(table.get("name") or folder.name)
    methods = {}
    for method, entry in (table.get("methods") or {}).items():
        entry = entry or {}
        methods[str(method)] = {"description": str(entry.get("description", "")),
                                "status": str(entry.get("status", "")),
                                "experiences": [str(g) for g in entry.get("experiences", [])],
                                "result": str(entry["result"]) if entry.get("result") else None}
    return {"name": name, "description": str(table.get("description", "")),
            "focus": str(table["focus"]) if table.get("focus") else None,
            "datasets": [str(d) for d in table.get("datasets", [])],
            "methods": methods, "folder": str(folder)}


def method_of(name: str, spec: dict, project: dict | None) -> str | None:
    """The method an experience tries: declared in the experience, else matched by a
    method's globs, else None."""
    if spec.get("method"):
        return str(spec["method"])
    for method, entry in ((project or {}).get("methods") or {}).items():
        if any(fnmatch.fnmatch(name, glob) for glob in entry.get("experiences", [])):
            return method
    return None


# ── what varies, what came out ────────────────────────────────────────────────

def _scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float)) or \
        (isinstance(value, str) and len(value) <= MAX_FACTOR_TEXT)


def _flatten(prefix: str, value, out: dict) -> None:
    """Plain values of `value` under dotted keys; references to other steps, paths and
    outputs are not factors (they are the plumbing, not a choice)."""
    if isinstance(value, dict):
        if len(value) == 1:
            (kind, body), = value.items()
            if kind == "$config":
                out[prefix] = body.get("name")
                for key, item in sorted((body.get("overrides") or {}).items()):
                    _flatten(f"{prefix}.{key}", item, out)
                return
            if kind == "$dataset":
                split = body.get("split")
                out[prefix] = body.get("name", "") + (f"[{split}]" if split else "")
                return
            if kind == "$data":
                out[prefix] = body.get("name")
                return
            if kind in ("$step", "$output", "$path", "$experience"):
                return
        for key, item in sorted(value.items()):
            _flatten(f"{prefix}.{key}", item, out)
        return
    if isinstance(value, list):
        if all(_scalar(item) for item in value) and len(json.dumps(value)) <= MAX_FACTOR_TEXT:
            out[prefix] = json.dumps(value)
        else:
            for index, item in enumerate(value):
                _flatten(f"{prefix}[{index}]", item, out)
        return
    if _scalar(value):
        out[prefix] = value


def is_plumbing(key: str, value) -> bool:
    """A path, a fingerprint or a hash: where things are, not a choice of the run."""
    leaf = key.rsplit(".", 1)[-1]
    return bool(PLUMBING_KEY.search(leaf)) or \
        (isinstance(value, str) and bool(PLUMBING_VALUE.match(value)))


def main_result(keys) -> str | None:
    """The result to rank runs by, from every run's result keys (repeated per run) — when
    the method declares none (`result = …`): never a count or a label (`n_test`,
    `runner_version`); a quality measure (accuracy, f1, loss…) first; then the one most runs
    have; then a curve's best value rather than its last."""
    keys = list(keys)
    if not keys:
        return None
    def leaf(key):
        base = key[:-5] if key.endswith((".last", ".best")) else key
        return base.rsplit(".", 1)[-1]
    # Never a count or a label when anything else exists; then only keys most runs have
    # (at least a quarter as many as the most frequent one).
    pool = [k for k in keys if not COUNT.search(leaf(k))] or keys
    most = max(pool.count(k) for k in set(pool))
    common = [k for k in pool if pool.count(k) * 4 >= most]

    def rank(key):
        base = key[:-5] if key.endswith((".last", ".best")) else key
        leaf = base.rsplit(".", 1)[-1]
        return (bool(COUNT.search(leaf)), not MEASURE.search(leaf), -keys.count(key),
                key.endswith(".last"), key)
    return min(set(common), key=rank)


def factors(spec: dict) -> dict:
    """{dotted key: plain value}: every choice the declaration makes (paths, fingerprints
    and hashes are left out: they say where things are, not what was chosen)."""
    out: dict = {}
    for step in spec.get("steps", []):
        names = step.get("arg_names") or []
        for index, value in enumerate(step.get("args") or []):
            label = names[index] if index < len(names) else f"arg{index}"
            _flatten(f"{step['name']}.{label}", value, out)
        for key, value in (step.get("kwargs") or {}).items():
            _flatten(f"{step['name']}.{key}", value, out)
    for key, value in (spec.get("tags") or {}).items():
        _flatten(f"tag.{key}", value, out)
    return {key: value for key, value in out.items() if not is_plumbing(key, value)}


def _numbers(prefix: str, value, out: dict, depth: int = 0) -> None:
    if depth > 6 or len(out) > 400:
        return
    if isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        if math.isfinite(value):
            out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _numbers(f"{prefix}.{key}", item, out, depth + 1)
    elif isinstance(value, list) and len(value) <= 20:
        for index, item in enumerate(value):
            _numbers(f"{prefix}[{index}]", item, out, depth + 1)


def numbers_of(value, prefix: str = "") -> dict:
    """{dotted key: number} of a JSON value: what a report says, without its long lists."""
    out: dict = {}
    _numbers(prefix, value, out)
    return {key.lstrip("."): number for key, number in out.items()}


def results(experience) -> dict:
    """{dotted key: number}: the numbers of the steps' JSON outputs (a report of any size
    up to MAX_REPORT_BYTES; lists longer than 20 are left out)."""
    out: dict = {}
    for step in experience.steps():
        path = experience.path(step["name"]) / "output.json"
        if not step["done"] or not path.exists() or path.stat().st_size > MAX_REPORT_BYTES:
            continue
        try:
            _numbers(step["name"], json.loads(path.read_text()), out)
        except ValueError:
            continue
    return out


def curves(experience) -> tuple[dict, dict]:
    """({step: points (at most MAX_CURVE_POINTS)}, {"<step>.<name>.last|best": number})."""
    from .monitor._snapshot import _metrics

    drawn, summary = {}, {}
    for step in experience.steps():
        points = _metrics(experience.path(step["name"]))
        if not points:
            continue
        if len(points) > MAX_CURVE_POINTS:
            stride = len(points) / MAX_CURVE_POINTS
            points = [points[int(i * stride)] for i in range(MAX_CURVE_POINTS)] + [points[-1]]
        drawn[step["name"]] = points
        names = {key for point in points for key in point if key != "step"}
        for name in sorted(names):
            values = [p[name] for p in points if isinstance(p.get(name), (int, float))]
            if not values:
                continue
            summary[f"{step['name']}.{name}.last"] = values[-1]
            if LOWER_IS_BETTER.search(name):
                summary[f"{step['name']}.{name}.best"] = min(values)
            elif HIGHER_IS_BETTER.search(name):
                summary[f"{step['name']}.{name}.best"] = max(values)
    return drawn, summary


def _code_version(spec: dict) -> str:
    codes = "|".join(f"{s['function']}={s.get('code', '')}" for s in spec.get("steps", []))
    return hashlib.sha256(codes.encode()).hexdigest()[:8]


def _git(experience) -> dict | None:
    folders = sorted((experience.folder / "code").glob("*/code.json"))
    if not folders:
        return None
    try:
        info = json.loads(folders[-1].read_text())
    except (OSError, ValueError):
        return None
    git = info.get("git") or None
    if not git:
        return None
    return {"commit": str(git.get("commit", ""))[:10], "dirty": bool(git.get("dirty"))}


# ── the declaration, written back as Python ──────────────────────────────────

def _python(value, steps: dict, imports: set) -> str:
    """A declared argument → the Python that declares it again."""
    if isinstance(value, dict) and len(value) == 1:
        (kind, body), = value.items()
        if kind == "$step":
            return steps.get(body, body.replace("-", "_").replace(".", "_"))
        if kind == "$config":
            imports.add("config")
            overrides = body.get("overrides") or {}
            extra = f", overrides={overrides!r}" if overrides else ""
            kind_part = f", {body['kind']!r}" if body.get("kind") else ""
            return f"config.load({body.get('name')!r}{kind_part}{extra})"
        if kind == "$dataset":
            imports.add("datasets")
            split = f", split={body['split']!r}" if body.get("split") else ""
            field = f".field({body['field']!r})" if body.get("field") else ""
            return f"datasets.load({body.get('name')!r}{split}){field}"
        if kind == "$data":
            imports.add("data")
            return f"data.load({body.get('name')!r})"
        if kind == "$path":
            imports.add("Path")
            return f"Path({body!r})"
        if kind == "$output":
            imports.add("exp")
            return f"exp.get({(body.get('experience', '') + '/' + body.get('step', '')).strip('/')!r})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_python(v, steps, imports)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_python(v, steps, imports) for v in value) + "]"
    return repr(value)


def declaration_code(spec: dict, name: str, method: str | None = None) -> str:
    """The `with exp.experience(...)` block that declares `spec` again under `name`: the
    start of an alternative (change a value, rerun). Arguments whose parameter name is
    known are written by name (`lr=0.003`), so each value says what it is."""
    steps, imports, lines, functions = {}, {"exp"}, [], {}
    for step in spec.get("steps", []):
        module, _, function = step["function"].partition(":")
        functions.setdefault(module, set()).add(function.split(".")[0])
    taken = {n for names in functions.values() for n in names} | {"e", "exp", "config", "data",
                                                                   "datasets", "Path"}
    for step in spec.get("steps", []):
        module, _, function = step["function"].partition(":")
        local = function.split(".")[-1]
        names = step.get("arg_names") or []
        values = [_python(value, steps, imports) for value in step.get("args") or []]
        # All by name when every name is known, else all by position (never a mix).
        named = len(names) == len(values) and all(n.isidentifier() for n in names)
        args = [f"{n}={v}" for n, v in zip(names, values)] if named else values
        kwargs = [f"{k}={_python(v, steps, imports)}" for k, v in (step.get("kwargs") or {}).items()]
        variable = re.sub(r"\W", "_", step["name"])
        if not variable.isidentifier() or variable in taken:
            variable = f"{variable}_out" if variable.isidentifier() else f"step_{len(steps)}"
        taken.add(variable)
        call = ", ".join([local if "." not in function else function, *args, *kwargs,
                          f"name={step['name']!r}"])
        lines.append(f"    {variable} = e.step({call})")
        steps[step["name"]] = variable
    options = [repr(name)]
    if method:
        options.append(f"method={method!r}")
    if spec.get("tags"):
        options.append(f"tags={spec['tags']!r}")
    head = []
    useml = sorted(i for i in imports if i != "Path")
    head.append(f"from useml import {', '.join(useml)}")
    if "Path" in imports:
        head.insert(0, "from pathlib import Path")
    for module, names in sorted(functions.items()):
        head.append(f"from {module} import {', '.join(sorted(names))}")
    return "\n".join(head + ["", f"with exp.experience({', '.join(options)}) as e:", *lines]) + "\n"


def _architecture(experience, step: dict) -> dict | None:
    """The block tree of the model this step returned (see exp._architecture)."""
    if not step.get("done"):
        return None
    from .exp._architecture import read

    return read(experience.path(step["name"]))


def step_wiring(spec: dict) -> dict:
    """{step: {"reads": the steps whose output it takes, "inputs": the data it loads}} —
    what a method's schema draws as arrows."""
    from .monitor._snapshot import _data

    out = {}
    for step in spec.get("steps", []):
        out[step["name"]] = {"reads": list(step.get("reads") or []),
                             "inputs": _data({"steps": [step]})}
    return out


# ── a run's record ────────────────────────────────────────────────────────────

ENDED = ("done", "failed", "died", "stopped")


def run_id(project: str, experience) -> str:
    spec = experience.spec or {}
    state = experience.state
    try:
        ended = int((experience.folder / "state.json").stat().st_mtime)
    except OSError:
        ended = 0
    key = f"{project}|{experience.name}|{spec.get('fingerprint')}|{state.get('status')}|{ended}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def record(experience, project: dict | None = None, *, machine: str | None = None) -> dict:
    """Everything worth keeping about one run of `experience` (see the module)."""
    from . import __version__

    project = project if project is not None else declared(experience.folder)
    spec = experience.spec or {}
    state = experience.state
    steps = experience.steps()
    drawn, summary = curves(experience)
    try:
        ended = (experience.folder / "state.json").stat().st_mtime
    except OSError:
        ended = None
    started = state.get("started")
    error = None
    if state.get("status") in ("failed", "died", "stopped"):
        error = {"meaning": (state.get("exit") or {}).get("meaning"),
                 "message": "\n".join(str(state.get("error") or "").splitlines()[-12:])}
    from .monitor._snapshot import _configs, _data

    code = {step["name"]: step.get("code") for step in spec.get("steps", [])}
    wiring = step_wiring(spec)
    name = (project or {}).get("name")
    return {
        "format": FORMAT, "id": run_id(name or "", experience), "project": name,
        "method": method_of(experience.name, spec, project), "experience": experience.name,
        "note": spec.get("note", ""), "tags": spec.get("tags") or {},
        "run": spec.get("fingerprint"), "version": _code_version(spec), "git": _git(experience),
        "status": state.get("status", "new"), "started": started, "ended": ended,
        "seconds": (ended - started) if ended and started else None,
        "machine": machine, "useml": __version__, "recorded": time.time(),
        "steps": [{"name": s["name"], "function": s["function"], "done": s["done"],
                   "seconds": s["seconds"], "code": code.get(s["name"]),
                   "reads": wiring.get(s["name"], {}).get("reads", []),
                   "inputs": wiring.get(s["name"], {}).get("inputs", []),
                   **({"architecture": arch} if (arch := _architecture(experience, s)) else {})}
                  for s in steps],
        "declaration": declaration_code(spec, experience.name,
                                        method_of(experience.name, spec, project)),
        "factors": factors(spec), "results": {**results(experience), **summary},
        "curves": drawn, "configs": _configs(spec), "datasets": _data(spec), "error": error,
    }


def runs(start=None, *, ended_only: bool = True) -> list[dict]:
    """Records of this project's experiences on this machine, newest first."""
    from . import exp

    project = declared(start)
    if project is None:
        return []
    out = []
    for experience in exp.ls():
        if ended_only and experience.status not in ENDED:
            continue
        out.append(record(experience, project))
    return sorted(out, key=lambda r: r.get("ended") or 0, reverse=True)


# ── changing the declaration without opening the file ────────────────────────

NAME = re.compile(r"[A-Za-z0-9_-]+")


class ProjectError(ValueError):
    pass


def project_file(start=None) -> tuple[Path, str]:
    """(the file that holds [project], the prefix of its tables): `useml.toml` → "",
    `pyproject.toml` with [tool.useml] → "tool.useml.". Without either: a new useml.toml at
    the root of the git repository (or the current folder)."""
    import subprocess

    from ._project import settings

    folder, _ = settings(start)
    if folder is not None:
        own = folder / "useml.toml"
        return (own, "") if own.is_file() else (folder / "pyproject.toml", "tool.useml.")
    here = Path(start or Path.cwd()).resolve()
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=here,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        top = ""
    return (Path(top) if top else here) / "useml.toml", ""


def _check_name(name: str, what: str) -> str:
    if not NAME.fullmatch(name or ""):
        raise ProjectError(f"{what} {name!r}: letters, digits, - and _ only")
    return name


def _require(start=None) -> dict:
    project = declared(start)
    if project is None:
        raise ProjectError("no [project] here: `useml project init` proposes one from what "
                           "exists")
    return project


def set_field(field: str, value, start=None) -> Path:
    """`name`, `description` or `focus` of the project (None removes it)."""
    from ._settings import edit_file

    if field not in ("name", "description", "focus"):
        raise ProjectError(f"unknown field {field!r}")
    path, prefix = project_file(start)
    if field == "focus" and value is not None:
        project = _require(start)
        if value not in project["methods"]:
            raise ProjectError(f"no method {value!r}; known: {sorted(project['methods'])}. "
                               f"Add it first: useml project method add {value}")
    return edit_file(path, f"{prefix}project", field, value, remove=value is None)


def add_method(name: str, *, description: str = "", status: str = "",
               experiences: list | None = None, result: str | None = None,
               start=None) -> Path:
    from ._settings import edit_file

    _check_name(name, "method")
    path, prefix = project_file(start)
    project = declared(start)
    if project and name in project["methods"]:
        raise ProjectError(f"method {name!r} exists: useml project method set {name} …")
    table = f"{prefix}project.methods.{name}"
    edit_file(path, table, "description", description)
    if status:
        edit_file(path, table, "status", status)
    if experiences:
        edit_file(path, table, "experiences", list(experiences))
    if result:
        edit_file(path, table, "result", result)
    return path


def set_method(name: str, *, description: str | None = None, status: str | None = None,
               experiences: list | None = None, result: str | None = None,
               start=None) -> Path:
    """Change what is given; an empty status or list removes the key."""
    from ._settings import edit_file

    project = _require(start)
    if name not in project["methods"]:
        raise ProjectError(f"no method {name!r}: useml project method add {name}")
    path, prefix = project_file(start)
    table = f"{prefix}project.methods.{name}"
    if description is not None:
        edit_file(path, table, "description", description)
    if status is not None:
        edit_file(path, table, "status", status or None, remove=not status)
    if experiences is not None:
        edit_file(path, table, "experiences", list(experiences) or None, remove=not experiences)
    if result is not None:
        edit_file(path, table, "result", result or None, remove=not result)
    return path


def remove_method(name: str, start=None) -> Path:
    """Remove the method's table (its runs stay; they become unassigned). Clears the focus
    if it pointed at it."""
    from ._settings import remove_table

    project = _require(start)
    path, prefix = project_file(start)
    if not remove_table(path, f"{prefix}project.methods.{name}"):
        raise ProjectError(f"no table [{prefix}project.methods.{name}] in {path.name}")
    if project.get("focus") == name:
        set_field("focus", None, start)
    return path


def set_datasets(add=(), remove=(), start=None) -> Path:
    from ._settings import edit_file

    project = declared(start) or {"datasets": []}
    current = list(project["datasets"])
    updated = [d for d in current if d not in set(remove)]
    for name in add:
        if name not in updated:
            updated.append(_check_name(name, "dataset"))
    path, prefix = project_file(start)
    return edit_file(path, f"{prefix}project", "datasets", updated or None, remove=not updated)


# ── proposing a declaration from what exists ─────────────────────────────────

def propose(start=None) -> dict:
    """What `useml project init` would add, read from the experiences that exist: a method
    per first part of their names (`wide/w4` → method `wide`, matched by `wide/*`), or the
    method they declare; the datasets they read; the focus = the method of the experience
    changed last. Nothing already declared is changed: only what is missing is added."""
    from . import exp
    from .monitor._snapshot import _data

    path, prefix = project_file(start)
    existing = declared(start)
    methods: dict = {}
    unassigned, datasets, latest = [], [], (0.0, None)
    for experience in exp.ls():
        spec = experience.spec or {}
        method = method_of(experience.name, spec, existing) if existing else spec.get("method")
        explicit = bool(method)
        if not method and "/" in experience.name:
            method = experience.name.split("/")[0]
        if not method or not NAME.fullmatch(method):
            unassigned.append(experience.name)
            continue
        entry = methods.setdefault(method, {"experiences": [], "glob": False})
        entry["experiences"].append(experience.name)
        entry["glob"] |= not explicit
        for name in _data(spec):
            base = name.split("[")[0]
            if base not in datasets:
                datasets.append(base)
        try:
            changed = (experience.folder / "experience.json").stat().st_mtime
        except OSError:
            changed = 0.0
        latest = max(latest, (changed, method), key=lambda pair: pair[0])
    declared_methods = (existing or {}).get("methods", {})
    new_methods = {name: {"experiences": [f"{name}/*"] if info["glob"] else [],
                          "count": len(info["experiences"])}
                   for name, info in sorted(methods.items()) if name not in declared_methods}
    new_datasets = [d for d in datasets if d not in (existing or {}).get("datasets", [])]
    folder = path.parent
    return {"file": str(path), "prefix": prefix, "exists": existing is not None,
            "name": None if existing else folder.name,
            "focus": None if existing and existing.get("focus") else latest[1],
            "methods": new_methods, "datasets": new_datasets, "unassigned": unassigned}


def proposal_text(plan: dict) -> str:
    """The proposal as the TOML lines it would add."""
    from ._settings import format_value

    prefix, lines = plan["prefix"], []
    head = []
    if plan["name"]:
        head += [f"name = {format_value(plan['name'])}", 'description = ""']
    if plan["focus"]:
        head.append(f"focus = {format_value(plan['focus'])}")
    if plan["datasets"]:
        head.append(f"datasets = [... + {', '.join(format_value(d) for d in plan['datasets'])}]"
                    if plan["exists"] else f"datasets = {format_value(plan['datasets'])}")
    if head:
        lines += [f"[{prefix}project]", *head, ""]
    for name, info in plan["methods"].items():
        lines += [f"[{prefix}project.methods.{name}]   # {info['count']} experience(s)",
                  'description = ""']
        if info["experiences"]:
            lines.append(f"experiences = {format_value(info['experiences'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def init(start=None, *, name: str | None = None) -> dict:
    """Write the proposal (only additions). Returns it."""
    plan = propose(start)
    if name:
        plan["name"] = name if not plan["exists"] else plan["name"]
    if plan["name"]:
        set_field("name", plan["name"], start)
        set_field("description", "", start)
    for method, info in plan["methods"].items():
        add_method(method, experiences=info["experiences"] or None, start=start)
    if plan["focus"]:
        set_field("focus", plan["focus"], start)
    if plan["datasets"]:
        set_datasets(add=plan["datasets"], start=start)
    return plan


# ── export ────────────────────────────────────────────────────────────────────

def summary(project: dict, records: list[dict]) -> dict:
    """The project as data: its declaration, its datasets and methods with what was run."""
    datasets: dict = {name: {"declared": True, "runs": 0, "methods": set()}
                      for name in project.get("datasets", [])}
    methods: dict = {name: {**entry, "runs": 0, "versions": set(), "last": None}
                     for name, entry in project.get("methods", {}).items()}
    for entry in records:
        method = entry.get("method") or "(unassigned)"
        slot = methods.setdefault(method, {"description": "", "status": "", "experiences": [],
                                           "runs": 0, "versions": set(), "last": None})
        slot["runs"] += 1
        slot["versions"].add(entry.get("version"))
        slot["last"] = max(slot["last"] or 0, entry.get("ended") or 0) or None
        for dataset in entry.get("datasets", []):
            base = dataset.split("[")[0]
            info = datasets.setdefault(base, {"declared": False, "runs": 0, "methods": set()})
            info["runs"] += 1
            info["methods"].add(method)
    for info in datasets.values():
        info["methods"] = sorted(info["methods"])
    for slot in methods.values():
        slot["versions"] = len(slot["versions"] - {None})
    return {"name": project["name"], "description": project.get("description", ""),
            "focus": project.get("focus"), "datasets": datasets, "methods": methods,
            "runs": len(records)}


def to_markdown(project: dict, records: list[dict]) -> str:
    """A readable export: datasets, methods (the focus first), and every run's factors that
    vary and results."""
    data = summary(project, records)
    lines = [f"# {data['name']}", ""]
    if data["description"]:
        lines += [data["description"], ""]
    if data["focus"]:
        lines += [f"Working on: **{data['focus']}**", ""]
    lines += ["## Datasets", ""]
    for name, info in sorted(data["datasets"].items()):
        used = f"{info['runs']} runs ({', '.join(info['methods'])})" if info["runs"] else "no run yet"
        lines.append(f"- {name}: {used}")
    lines += ["", "## Methods", ""]
    order = sorted(data["methods"], key=lambda m: (m != data["focus"], m))
    for name in order:
        info = data["methods"][name]
        status = f" ({info['status']})" if info.get("status") else ""
        lines.append(f"- **{name}**{status}: {info.get('description') or ''} — "
                     f"{info['runs']} runs, {info['versions']} code versions")
    if records:
        keys = sorted({k for r in records for k in r.get("factors", {})})
        varying = [k for k in keys if len({json.dumps(r["factors"].get(k)) for r in records}) > 1]
        metrics = sorted({k for r in records for k in r.get("results", {})})[:8]
        lines += ["", "## Runs", "",
                  "| method | experience | status | " + " | ".join(varying + metrics) + " |",
                  "|" + "---|" * (3 + len(varying) + len(metrics))]
        for r in records:
            cells = [r.get("method") or "", r["experience"], r["status"]]
            cells += [str(r["factors"].get(k, "")) for k in varying]
            cells += [f"{r['results'][k]:.4g}" if k in r.get("results", {}) else ""
                      for k in metrics]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["is_plumbing", "main_result", "project_file", "set_field", "add_method", "set_method", "remove_method",
           "set_datasets", "propose", "proposal_text", "init", "ProjectError",
           "declared", "method_of", "factors", "results", "curves", "record", "runs",
           "run_id", "summary", "to_markdown", "ENDED", "LOWER_IS_BETTER"]
