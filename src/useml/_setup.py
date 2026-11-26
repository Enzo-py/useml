"""`useml init` and `useml check`.

    useml init              look at this machine and this project, PRINT what it would write
    useml init --yes        write it
    useml check             check that everything is in place, line by line

WHAT `init` NEVER DOES: overwrite a value, move or delete a file, touch a cache. It only
ADDS what is missing — a key absent from the machine config, a table absent from the
project's pyproject.toml (or a new `useml.toml`), a line absent from .gitignore — and keeps
a copy of every file it changes (`<file>.useml-backup-<date>`).

WHAT IT LOOKS FOR, to adopt what exists rather than start over:
  machine   a cache root (`--root`, or a `.cache_root` file of the project,
            and the `VARIABLE=path` lines of that file), an existing cache store
            (`--store`, `.store_root`, `<project>/.cache/store/index.sqlite`:
            useml opens it in place, nothing is copied)
  project   a folder of YAML configs with `type:` keys (its kinds are its sub-folders),
            the packages that define those types (found by scanning the project's
            `class X` definitions), an `experiences/` folder
  git       ignore rules for what runs write (step outputs, logs, code snapshots, bundles),
            unless git already ignores them
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SKIPPED_DIRS = {".git", ".venv", "venv", "env", "node_modules", "build", "dist", "__pycache__",
                "site-packages", ".cache", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                "experiences", "data", "outputs", "wandb", "bundles", ".idea", ".vscode"}
CONFIG_DIR_NAMES = ("configs", "config", "conf")


@dataclass
class Change:
    path: Path
    mode: str                  # "create", "append" or "prepend"
    text: str
    why: str


@dataclass
class Plan:
    changes: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def render(self) -> str:
        lines = []
        for change in self.changes:
            verb = {"create": "create", "append": "add to", "prepend": "add at the top of"}[change.mode]
            lines += [f"• {verb} {change.path}   ({change.why})"]
            lines += [f"    + {line}" for line in change.text.rstrip("\n").splitlines()]
        if not self.changes:
            lines.append("• nothing to write: everything is already set up")
        lines += [f"  note: {note}" for note in self.notes]
        lines += [f"  ⚠ {warning}" for warning in self.warnings]
        return "\n".join(lines)

    def apply(self) -> list[Path]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        written = []
        for change in self.changes:
            path = change.path
            path.parent.mkdir(parents=True, exist_ok=True)
            before = path.read_text() if path.exists() else None
            if before is not None:
                shutil.copy2(path, path.with_name(f"{path.name}.useml-backup-{stamp}"))
            if change.mode == "create" or before is None:
                text = change.text
            elif change.mode == "prepend":
                text = change.text + "\n" + before
            else:
                text = before + ("" if before.endswith("\n") or not before else "\n") + \
                    "\n" + change.text
            path.write_text(text)
            written.append(path)
        return written


# ── what exists ───────────────────────────────────────────────────────────────

def _toml(path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                         # pragma: no cover
        import tomli as tomllib
    with open(path, "rb") as file:
        return tomllib.load(file)


def _quote(value) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def legacy_cache_root(project: Path) -> tuple[str | None, dict]:
    """A cache root and `VARIABLE=path` choices from the project's `.cache_root` file."""
    root, choices = None, {}
    path = project / ".cache_root"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                variable, _, value = line.partition("=")
                choices[variable.strip()] = value.strip()
            elif root is None:
                root = line
    return root, choices


def legacy_store(project: Path) -> str | None:
    written = project / ".store_root"
    if written.is_file() and written.read_text().strip():
        return written.read_text().strip()
    default = project / ".cache" / "store"
    return str(default) if (default / "index.sqlite").exists() else None


def _yaml_types(node, out: set) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("type"), str):
            out.add(node["type"])
        for value in node.values():
            _yaml_types(value, out)
    elif isinstance(node, list):
        for value in node:
            _yaml_types(value, out)


def find_config_library(project: Path) -> dict | None:
    """{root, kinds, types, default_kind} for the first YAML config folder found."""
    import yaml

    for name in CONFIG_DIR_NAMES:
        folder = project / name
        if not folder.is_dir():
            continue
        files = sorted(folder.rglob("*.y*ml"))[:500]
        types, by_folder = set(), {}
        for file in files:
            try:
                raw = yaml.safe_load(file.read_text()) or {}
            except (yaml.YAMLError, OSError, UnicodeDecodeError):
                continue
            if isinstance(raw, dict) and "type" in raw:
                _yaml_types(raw, types)
                by_folder.setdefault(file.parent.relative_to(folder).as_posix(), 0)
                by_folder[file.parent.relative_to(folder).as_posix()] += 1
        if not by_folder:
            continue
        kinds = {}
        for sub in sorted(by_folder):
            label = sub.split("/")[-1] if sub != "." else "config"
            singular = label[:-1] if label.endswith("s") and len(label) > 3 else label
            kinds[singular] = sub
        default = "model" if "model" in kinds else max(kinds, key=lambda k: by_folder[kinds[k]])
        return {"root": name, "kinds": kinds, "types": sorted(types), "default_kind": default}
    return None


def find_type_packages(project: Path, types: list[str]) -> tuple[list, list, list]:
    """(packages, pythonpath, types not found): the fewest top-level packages of the
    project whose files define the `types`, by scanning `class Name` lines."""
    pattern = re.compile(r"^class\s+([A-Za-z_]\w*)\b", re.MULTILINE)
    defined: dict = {}
    for folder, dirs, files in os.walk(project):
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS and not d.startswith(".")
                   and d != "tests"]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(folder) / name
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for found in pattern.findall(text):
                if found in types:
                    defined.setdefault(found, []).append(path.relative_to(project))
    packages: dict = {}
    for name, paths in defined.items():
        for relative in paths:
            parts = list(relative.parts)
            base = "."
            if parts[0] == "src" and len(parts) > 1:
                base, parts = "src", parts[1:]
            top = parts[0] if len(parts) > 1 else Path(parts[0]).stem
            packages.setdefault((top, base), set()).add(name)
    chosen, covered = [], set()
    while True:
        best = max(packages.items(), key=lambda item: len(item[1] - covered), default=None)
        if best is None or not (best[1] - covered):
            break
        chosen.append(best[0])
        covered |= best[1]
    missing = sorted(set(types) - covered)
    names = [top for top, _ in chosen]
    pythonpath = sorted({base for _, base in chosen})
    return names, pythonpath, missing


def _git_ignored(project: Path, probe: str) -> bool | None:
    try:
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=project,
                                capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 128:
        return None                                  # not a git repository
    return result.returncode == 0


# ── the plan ──────────────────────────────────────────────────────────────────

def plan(project=None, *, root=None, store=None, machine: bool = True,
         project_setup: bool = True) -> Plan:
    from . import paths

    project = Path(project or Path.cwd()).resolve()
    result = Plan()

    if machine:
        config_path = paths.config_path()
        current = _toml(config_path) if config_path.exists() else {}
        legacy_root, choices = legacy_cache_root(project)
        wanted_root = root or legacy_root
        top, tables = [], []
        if "root" not in current:
            if wanted_root:
                top.append(f"root = {_quote(Path(wanted_root).expanduser())}")
                source = "--root" if root else ".cache_root"
                result.notes.append(f"cache root from {source}: every library cache goes "
                                    f"under it; caches already placed outside the home stay")
            else:
                result.notes.append("no cache root: library caches stay where they are (fine "
                                    "on a laptop). On a server: useml init --root /big/disk/useml")
        elif root and str(current["root"]) != str(root):
            result.warnings.append(f"the machine config already sets root = {current['root']!r}; "
                                   f"kept (edit {config_path} to change it)")
        env = {k: v for k, v in choices.items() if k not in (current.get("env") or {})}
        if env and "env" in current:
            # Appending keys at the end of the file would put them in its LAST table.
            result.warnings.append(f"add under [env] in {config_path} by hand: " + ", ".join(
                f"{key} = {_quote(value)}" for key, value in env.items()))
        elif env:
            tables.append("[env]")
            tables += [f"{key} = {_quote(value)}" for key, value in env.items()]
        store = store or legacy_store(project)
        if store and "root" not in (current.get("store") or {}):
            if "store" in current:
                result.warnings.append(f"existing cache store at {store}: add root = "
                                       f"{_quote(store)} under [store] in {config_path} by hand")
            else:
                tables += ["", "[store]", f"root = {_quote(Path(store).resolve())}"]
                result.notes.append(f"the existing cache store {store} is used IN PLACE "
                                    f"(its index gains one table; older readers still read it)")
        if "hub" not in current:
            tables += ["", "[hub]", "parallel = false        # one download thread at a time"]
        tables = [line for line in tables if line is not None]
        if top and config_path.exists():
            result.changes.append(Change(config_path, "prepend", "\n".join(top) + "\n",
                                         "machine config: cache root"))
            top = []
        if top or tables:
            body = "\n".join(top + ([""] if top and tables else []) + tables).strip("\n") + "\n"
            mode = "append" if config_path.exists() else "create"
            header = "" if config_path.exists() else "# useml — settings of this machine\n"
            result.changes.append(Change(config_path, mode, header + body,
                                         "machine config"))

    if project_setup:
        pyproject = project / "pyproject.toml"
        own = project / "useml.toml"
        settings = (_toml(own) if own.exists() else
                    _toml(pyproject).get("tool", {}).get("useml", {}) if pyproject.exists() else {})
        prefix = "" if own.exists() or not pyproject.exists() else "tool.useml."
        target = own if own.exists() or not pyproject.exists() else pyproject
        blocks = []
        if "config" not in settings:
            library = find_config_library(project)
            if library:
                packages, pythonpath, missing = find_type_packages(project, library["types"])
                kinds = ", ".join(f"{k} = {_quote(v)}" for k, v in library["kinds"].items())
                block = [f"[{prefix}config]", f"root = {_quote(library['root'])}",
                         f"kinds = {{{kinds}}}",
                         f"default_kind = {_quote(library['default_kind'])}",
                         f"packages = [{', '.join(_quote(p) for p in packages)}]"]
                if pythonpath:
                    block.append(f"pythonpath = [{', '.join(_quote(p) for p in pythonpath)}]")
                blocks.append("\n".join(block))
                result.notes.append(f"configs: {library['root']}/ with kinds "
                                    f"{list(library['kinds'])}; {len(library['types'])} types "
                                    f"used, defined in {packages or 'no project package'}")
                if missing:
                    result.warnings.append(
                        f"{len(missing)} types are not defined by a class of the project "
                        f"(from an installed library, or a function): {missing[:8]}"
                        f"{' …' if len(missing) > 8 else ''}. Add their package to "
                        f"`packages` if `useml check` cannot resolve them.")
        if "exp" not in settings:
            blocks.append(f"[{prefix}exp]\nroot = \"experiences\"")
            legacy = [path for path in (project / "experiences").rglob("experience.json")
                      if "steps" not in path.parts] if (project / "experiences").is_dir() else []
            legacy = [path for path in legacy if '"format"' not in path.read_text()[:400]]
            if legacy:
                result.notes.append(f"{len(legacy)} experiences of the former system in "
                                    f"experiences/: left untouched; useml lists only its own")
        if blocks:
            header = "" if target.exists() else "# useml — settings of this project\n"
            text = header + "\n\n".join(blocks) + "\n"
            result.changes.append(Change(target, "append" if target.exists() else "create",
                                         text, "project settings"))

        ignore_lines = []
        probes = {"experiences/**/steps/": "experiences/a/b/steps/x",
                  "experiences/**/code/": "experiences/a/b/code/x",
                  "experiences/**/run.log": "experiences/a/b/run.log",
                  "experiences/**/state.json": "experiences/a/b/state.json",
                  "experiences/**/crash.txt": "experiences/a/b/crash.txt",
                  "bundles/": "bundles/x"}
        for line, probe in probes.items():
            ignored = _git_ignored(project, probe)
            if ignored is None:
                break                                 # not a git repository
            if not ignored:
                ignore_lines.append(line)
        if ignore_lines:
            text = "# useml: what runs write (outputs, logs, code snapshots) and bundles\n" + \
                "\n".join(ignore_lines) + "\n"
            result.changes.append(Change(project / ".gitignore", "append", text, "git"))
        gitignore = project / ".gitignore"
        if gitignore.exists() and re.search(r"^(<<<<<<<|>>>>>>>|=======)", gitignore.read_text(),
                                            re.MULTILINE):
            result.warnings.append(".gitignore holds unresolved merge-conflict markers "
                                   "(<<<<<<< / >>>>>>>): resolve them by hand")
    return result


def init(project=None, *, root=None, store=None, yes: bool = False, machine: bool = True,
         project_setup: bool = True) -> Plan:
    """Print the plan; write it with `yes=True` (or after a confirmation in a terminal)."""
    current = plan(project, root=root, store=store, machine=machine,
                   project_setup=project_setup)
    print(current.render())
    if not current.changes:
        return current
    if not yes:
        if sys.stdin.isatty():
            answer = input("\nwrite these changes? [y/N] ").strip().lower()
            yes = answer in ("y", "yes", "o", "oui")
        else:
            print("\ndry run: nothing written. Run `useml init --yes` to write it.")
    if yes:
        for path in current.apply():
            print(f"written: {path}")
        print("\nnext: `useml check` checks that everything resolves.")
    return current


# ── check ─────────────────────────────────────────────────────────────────────

LEVELS = {"ok": "✓", "warn": "⚠", "fail": "✗"}
#: The order in which areas are printed.
AREAS = ("config", "folders", "caches", "downloads", "store", "processed", "configs",
         "datasets", "experiences", "system")
LOW_FREE_BYTES = 20 << 30          # below 20 GB free, a cache folder is about to fill up


def _writable(folder: Path) -> str | None:
    """None when `folder` (created if missing) accepts a file, else the reason."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".useml-write-test-{os.getpid()}"
        probe.write_text("x")
        probe.unlink()
        return None
    except OSError as error:
        return str(error)


def _config_checks(add) -> None:
    """Every key of the machine config is known and has the right type."""
    from . import _settings, paths

    path = paths.config_path()
    try:
        config = paths.config()
    except ValueError as error:
        add("fail", "config", str(error), f"open {path} and fix the line it names")
        return
    if not config:
        add("ok", "config", f"no machine config ({path}): defaults apply",
            "useml settings edit")
        return
    problems = 0

    def walk(table: dict, prefix: str) -> None:
        nonlocal problems
        for key, value in table.items():
            full = f"{prefix}{key}"
            if isinstance(value, dict) and full in ("env", "hub", "store", "data", "datasets",
                                                    "exp", "monitor"):
                walk(value, f"{full}.")
                continue
            try:
                _settings.parse_value(full, _settings.format_value(value))
            except ValueError as error:
                problems += 1
                add("fail", "config", f"{path.name}: {error}", f"useml settings unset {full}")

    walk(config, "")
    if not problems:
        add("ok", "config", f"machine config {path}: every key known and well typed")


def _folder_checks(add) -> None:
    """Every folder useml writes to: where it is, that it accepts files, its free space,
    and that it is not left in the home folder while a cache root is set elsewhere."""
    from . import paths
    from .store import human

    root = paths.root()
    folders = [("root", "root", root)] if root else []
    try:
        from . import store as stores

        folders.append(("cache store", "store.root", stores.default_root()))
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from .data._catalog import data_root

        folders.append(("processed data", "data.root", data_root()))
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from . import datasets

        folders.append(("datasets", "datasets.root", datasets.root()))
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from . import exp

        folders.append(("experiences", "exp.root", exp.root()))
    except Exception:                                        # noqa: BLE001
        pass
    if not root:
        add("warn", "folders", "no cache root: library caches stay in their default places "
            "(often the home folder)", "useml settings set root /path/on/a/big/disk")
    configured = paths.config()
    for label, key, folder in folders:
        folder = Path(folder)
        reason = _writable(folder)
        if reason:
            add("fail", "folders", f"{label} {folder}: cannot write ({reason})",
                f"useml settings set {key} /a/writable/folder")
            continue
        free = shutil.disk_usage(folder).free
        table, _, name = key.partition(".")
        explicitly_set = bool((configured.get(table) or {}).get(name)) if name else True
        if root and key != "root" and paths.is_under_home_or_temporary(folder) \
                and not paths.is_under_home_or_temporary(root):
            add("warn", "folders", f"{label} {folder} is in the home folder, the root is "
                f"{root}", f"useml settings unset {key}" if explicitly_set else None)
        elif free < LOW_FREE_BYTES:
            add("warn", "folders", f"{label} {folder}: only {human(free)} free",
                "useml store gc --apply" if key == "store.root" else None)
        else:
            add("ok", "folders", f"{label} {folder}: writable, {human(free)} free")


def checks(*, measure: bool = False) -> list[dict]:
    """Everything that should be in place, as [{"level", "area", "message", "fix"}].

    level: "ok", "warn" or "fail". fix: the command that repairs it, or None. Nothing is
    changed here: the fixes are only proposed."""
    from . import hub, paths
    from .store import human

    out = []

    def add(level, area, message, fix=None):
        out.append({"level": level, "area": area, "message": message, "fix": fix})

    _config_checks(add)
    _folder_checks(add)
    stray = hub.scan(measure=measure)
    if stray:
        size = sum(entry["bytes"] or 0 for entry in stray)
        add("warn", "caches", f"{len(stray)} caches left in the home"
            + (f" ({human(size)})" if measure else "") + ": "
            + ", ".join(entry["path"] for entry in stray[:5]),
            "useml hub scan --move --apply" if paths.root() else None)
    else:
        add("ok", "caches", "no cache left in the home folder")
    status = hub.status()
    add("ok" if not status["parallel_allowed"] else "warn", "downloads",
        "downloads: sequential" if not status["parallel_allowed"]
        else "downloads: parallel allowed", None if not status["parallel_allowed"]
        else "useml settings unset hub.parallel")
    try:
        from . import store as stores

        opened = stores.open()
        total, budget = opened.total_bytes(), stores.budget()
        if budget and total > budget:
            add("warn", "store", f"cache store: {human(total)}, over its budget of "
                f"{human(budget)}", "useml store gc --apply")
        else:
            add("ok", "store", f"cache store {opened.root}: {len(opened.declared())} spaces, "
                f"{human(total)}" + (f" of {human(budget)}" if budget else ""))
    except Exception as error:                               # noqa: BLE001
        add("fail", "store", f"cache store: {error}", "useml store check")
    try:
        from . import data

        unused = [t for t in data.tables() if not t["names"]]
        if unused:
            add("warn", "processed", f"{len(unused)} processed tables no name points to "
                f"({human(sum(t['bytes'] for t in unused))})", "useml data tables")
    except Exception as error:                               # noqa: BLE001
        add("warn", "processed", f"processed data: {error}")
    try:
        from . import config as configs

        library = configs.library()
        names = library.ls()
        broken, unresolved = [], set()
        for kind, files in names.items():
            for name in files:
                try:
                    config = library.load_raw(name, kind)
                except Exception as error:                   # noqa: BLE001
                    broken.append(f"{kind}/{name}: {error}")
                    continue
                types: set = set()
                _yaml_types({"type": config.type, **config.kwargs}, types)
                for type_name in types:
                    try:
                        library.namespace.resolve(type_name)
                    except Exception:                        # noqa: BLE001
                        unresolved.add(type_name)
        for line in broken[:10]:
            add("fail", "configs", line, "useml config show " + line.split(":")[0].split("/")[-1])
        if unresolved:
            add("fail", "configs", f"types not found in {list(library.namespace.packages)}: "
                f"{sorted(unresolved)[:10]} — add their package to `packages`")
        elif not broken:
            add("ok", "configs", f"configs {library.root}: "
                + ", ".join(f"{len(v)} {k}" for k, v in names.items())
                + "; every config reads and every `type:` resolves")
    except Exception as error:                               # noqa: BLE001
        if "no config library" not in str(error):           # most projects have none
            add("warn", "configs", f"configs: {error}")
    try:
        from . import datasets

        entries = datasets.ls(measure=False)
        if entries:
            ready = [e["name"] for e in entries if e["current"]]
            outdated = [e["name"] for e in entries if e["prepared"] and not e["current"]]
            never = [e["name"] for e in entries if not e["prepared"]]
            add("ok", "datasets", f"datasets {datasets.root()}: {len(entries)} drivers, "
                f"{len(ready)} ready" + (f", {len(never)} never prepared" if never else ""))
            for name in outdated:
                add("warn", "datasets", f"dataset {name}: its driver changed (rebuilt at next "
                    "load)", f"useml datasets prepare {name}")
    except Exception as error:                               # noqa: BLE001
        add("fail", "datasets", f"datasets: {error}")
    try:
        from . import exp

        experiences = exp.ls()
        running = [e.name for e in experiences if e.status == "running"]
        add("ok", "experiences", f"experiences {exp.root()}: {len(experiences)}"
            + (f", {len(running)} running" if running else ""))
        for experience in experiences:
            if experience.status == "died":
                add("fail", "experiences", f"{experience.name} died without a word",
                    f"useml exp why {experience.name}")
    except Exception as error:                               # noqa: BLE001
        add("warn", "experiences", f"experiences: {error}")
    try:
        from .exp._launch import logout_kills_processes, tmux_available

        add("ok" if tmux_available() else "warn", "system",
            "tmux available: runs get a terminal to reattach to" if tmux_available()
            else "no tmux: runs still survive disconnection, without a terminal to reattach",
            None if tmux_available() else
            "brew install tmux" if sys.platform == "darwin" else "sudo apt install tmux")
        warning = logout_kills_processes()
        if warning:
            add("fail", "system", warning, "loginctl enable-linger $USER")
    except Exception:                                        # noqa: BLE001
        pass
    order = {area: index for index, area in enumerate(AREAS)}
    return sorted(out, key=lambda c: order.get(c["area"], len(order)))


def check(*, measure: bool = False) -> list[tuple[str, str]]:
    """[(level, message)], level in ✓ ⚠ ✗ — the short form of `checks`."""
    return [(LEVELS[c["level"]], c["message"]) for c in checks(measure=measure)]
