"""Caches left in the home: find them, measure them, move them under the root.

    useml hub scan            # what is where, and how big
    useml hub scan --move     # move what can be moved (asks nothing, dry run without it)

A MOVE NEVER OVERWRITES. Each entry of the old cache is renamed into the new one if the
name is free there; a folder present on both sides is merged one level down, the same
way; a FILE present on both sides stays where it is and is reported. Within one disk a
rename is instantaneous; across disks the entry is copied, then removed.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ._registry import disk_size


@dataclass(frozen=True)
class StrayCache:
    path: Path
    variable: str | None      # the variable that redirects it, None if none does
    note: str = ""


def candidates() -> list[StrayCache]:
    home = Path.home()
    return [
        StrayCache(home / ".cache" / "huggingface", "HF_HOME"),
        StrayCache(home / ".cache" / "torch", "TORCH_HOME"),
        StrayCache(home / "nltk_data", "NLTK_DATA"),
        StrayCache(home / ".cache" / "pip", "PIP_CACHE_DIR"),
        StrayCache(home / ".cache" / "uv", "UV_CACHE_DIR"),
        StrayCache(home / ".cache" / "matplotlib", "MPLCONFIGDIR"),
        StrayCache(home / ".matplotlib", "MPLCONFIGDIR"),
        StrayCache(home / ".triton", "TRITON_CACHE_DIR"),
        StrayCache(home / ".nv" / "ComputeCache", "CUDA_CACHE_PATH"),
        StrayCache(home / ".cache" / "vllm", "VLLM_CACHE_ROOT"),
        StrayCache(home / ".cache" / "clip", None,
                   "OpenAI CLIP's download_root: pass download_root= or use hf://"),
        StrayCache(home / ".cache" / "torch_extensions", None, "TORCH_EXTENSIONS_DIR"),
    ]


def target_of(cache: StrayCache) -> Path | None:
    """Where the root puts this cache, None without a root or a variable."""
    root = paths.root()
    if root is None or cache.variable is None:
        return None
    choice = paths.configured_choices().get(cache.variable)
    return Path(choice).expanduser() if choice else root / paths.CACHE_VARIABLES[cache.variable]


def scan(*, measure: bool = True) -> list[dict]:
    """Every candidate that exists, with its size and where it would go."""
    out = []
    for cache in candidates():
        if not cache.path.exists() or cache.path.is_symlink():
            continue
        target = target_of(cache)
        in_use = cache.variable is not None and \
            Path(os.environ.get(cache.variable, "/nonexistent")).expanduser().resolve() \
            == cache.path.resolve()
        out.append({"path": str(cache.path), "variable": cache.variable,
                    "bytes": disk_size(cache.path) if measure else None,
                    "target": str(target) if target else None, "in_use": in_use,
                    "note": cache.note})
    return out


def _merge(source: Path, target: Path, report: dict, apply: bool) -> None:
    target_exists = target.exists()
    if apply and not target_exists:
        target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        destination = target / entry.name
        if not destination.exists() and not destination.is_symlink():
            report["moved"].append(str(entry))
            if apply:
                shutil.move(str(entry), str(destination))
        elif entry.is_dir() and not entry.is_symlink() and destination.is_dir():
            _merge(entry, destination, report, apply)
        else:
            report["conflicts"].append(str(entry))
    if apply and source.exists() and not any(source.iterdir()):
        source.rmdir()


def move(path, *, apply: bool = False) -> dict:
    """Move the stray cache at `path` under the root. Dry run unless `apply=True`."""
    path = Path(path).expanduser()
    cache = next((c for c in candidates() if c.path == path), None)
    if cache is None:
        raise ValueError(f"{path} is not a known cache location")
    target = target_of(cache)
    if target is None:
        raise ValueError(f"{path}: no root configured, or no variable redirects this cache")
    if target.resolve() == path.resolve():
        return {"source": str(path), "target": str(target), "moved": [], "conflicts": [],
                "applied": False}
    report = {"source": str(path), "target": str(target), "moved": [], "conflicts": [],
              "applied": apply}
    _merge(path, target, report, apply)
    return report
