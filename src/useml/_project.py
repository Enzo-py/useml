"""Project settings: `useml.toml` at the project root, or `[tool.useml]` in pyproject.toml.

    # useml.toml                         # pyproject.toml
    [config]                             [tool.useml.config]
    root = "configs"                     root = "configs"
    [exp]                                [tool.useml.exp]
    root = "experiences"                 root = "experiences"

The nearest folder above the current one holding either file is the project. Paths in it
are relative to that folder.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _toml(path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                         # pragma: no cover
        import tomli as tomllib
    with open(path, "rb") as file:
        return tomllib.load(file)


def settings(start=None) -> tuple[Path | None, dict]:
    """(project folder, {"config": {...}, "exp": {...}}) — the nearest useml settings above
    `start`, or (None, {})."""
    start = Path(start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        own = folder / "useml.toml"
        if own.is_file():
            return folder, _toml(own)
        pyproject = folder / "pyproject.toml"
        if pyproject.is_file():
            found = _toml(pyproject).get("tool", {}).get("useml")
            if found is not None:
                return folder, found
    return None, {}


def section(name: str, start=None) -> tuple[Path | None, dict | None]:
    folder, found = settings(start)
    return (folder, found.get(name)) if folder is not None and name in found else (None, None)
