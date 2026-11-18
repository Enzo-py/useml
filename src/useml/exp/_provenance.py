"""What code ran: recorded at every launch, so a result can always be traced — and bundled.

    <experience>/code/<launch time>/
        code.json        project root, commit, branch, dirty, python, platform, GPUs,
                         import roots of the step functions, useml version
        diff.patch       `git diff HEAD --binary`: what was not committed
        untracked/       small untracked source files (.py .yaml .toml .json …)
        packages.txt     every installed distribution, name==version

Reading the commit and the diff costs well under a second; NOTHING in the repository is
modified (no stash, no ref, no tag). Without git, the source folders of the step
functions are copied instead.

Each step's `done.json` names the launch whose code produced it (`"code"` folder).
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".txt", ".md",
                   ".sh", ".pyx", ".c", ".cpp", ".cu", ".h"}
MAX_FILE_BYTES = 1 << 20
MAX_UNTRACKED_BYTES = 20 << 20


def _git(*args, cwd) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode(errors="replace") if result.returncode == 0 else None


def project_root(start=None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    top = _git("rev-parse", "--show-toplevel", cwd=start)
    return Path(top.strip()).resolve() if top else start


def import_roots(spec: dict, root: Path) -> list[str]:
    """Folders (relative to `root`) that must be on sys.path to import the step functions
    and the config packages: '.' for a top-level package, 'src' for a src layout…"""
    import importlib

    modules = {step["function"].split(":")[0] for step in spec.get("steps", [])}
    try:
        from .. import config

        modules.update(config.library().namespace.packages)
    except Exception:                                       # noqa: BLE001 - no library
        pass
    roots = []
    for module_name in sorted(modules):
        try:
            module = importlib.import_module(module_name)
        except Exception:                                   # noqa: BLE001
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        parts = module_name.split(".")
        # a/b/c/__init__.py → climb 3 folders; a/b/c.py → climb 2 from c.py's folder
        base = path.parent
        for _ in range(len(parts) if path.name == "__init__.py" else len(parts) - 1):
            base = base.parent
        try:
            relative = base.relative_to(root).as_posix() or "."
        except ValueError:
            continue                                        # an installed package
        if relative not in roots:
            roots.append(relative)
    return roots or ["."]


def _packages() -> str:
    from importlib.metadata import distributions

    names = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                    if d.metadata and d.metadata["Name"]}, key=str.lower)
    return "\n".join(names) + "\n"


def _gpus() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def record_code(folder: Path, spec: dict) -> Path:
    """Snapshot the code of this launch under `<folder>/code/<time>/`; return it."""
    from .. import __version__
    from ._runtime import write_state

    root = project_root()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = folder / "code" / stamp
    suffix = 1
    while target.exists():
        suffix += 1
        target = folder / "code" / f"{stamp}-{suffix}"
    target.mkdir(parents=True)
    info = {"root": str(root), "time": time.time(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(),
            "host": platform.node(), "useml": __version__, "gpus": _gpus(),
            "pythonpath": import_roots(spec, root), "git": None}
    commit = _git("rev-parse", "HEAD", cwd=root)
    if commit:
        diff = _git("diff", "HEAD", "--binary", cwd=root) or ""
        (target / "diff.patch").write_text(diff)
        untracked, total, skipped = [], 0, []
        for relative in (_git("ls-files", "--others", "--exclude-standard", cwd=root) or "").splitlines():
            path = root / relative
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or total + size > MAX_UNTRACKED_BYTES:
                skipped.append(relative)
                continue
            destination = target / "untracked" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            untracked.append(relative)
            total += size
        info["git"] = {"commit": commit.strip(),
                       "branch": (_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root) or "").strip(),
                       "dirty": bool(diff.strip()), "untracked": untracked,
                       "untracked_skipped": skipped}
    else:
        copied = []
        for relative in info["pythonpath"]:
            base = (root / relative).resolve()
            for module in {s["function"].split(":")[0].split(".")[0] for s in spec["steps"]}:
                for candidate in (base / module, base / f"{module}.py"):
                    if candidate.is_dir():
                        shutil.copytree(candidate, target / "files" / relative / module,
                                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                                        dirs_exist_ok=True)
                        copied.append(f"{relative}/{module}")
                    elif candidate.is_file():
                        (target / "files" / relative).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(candidate, target / "files" / relative / candidate.name)
                        copied.append(f"{relative}/{candidate.name}")
        info["copied"] = copied
    (target / "packages.txt").write_text(_packages())
    (target / "code.json").write_text(json.dumps(info, indent=1))
    write_state(folder, code=target.name)
    return target
