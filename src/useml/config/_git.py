"""Git: the commit that runs, relative paths for citations, and finding a hash in history."""
from __future__ import annotations

import subprocess
from pathlib import Path


def git(*args, cwd=None) -> str:
    try:
        return subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
    except Exception:                             # noqa: BLE001 - no git, no repository
        return ""


def repository_root(near) -> Path | None:
    """RESOLVED: `/var` is a link to `/private/var` on macOS, and an unresolved root would
    not be a prefix of the resolved config folder."""
    folder = Path(near) if near else Path.cwd()
    folder = folder if folder.is_dir() else folder.parent
    root = git("rev-parse", "--show-toplevel", cwd=folder if folder.exists() else None)
    return Path(root).resolve() if root else None


def git_head(near=None) -> str:
    folder = Path(near).parent if near else None
    return git("rev-parse", "--short", "HEAD", cwd=folder if folder and folder.exists() else None)


def relative(path) -> str:
    """Relative to the repository when there is one, else the file name. Never absolute:
    it names the machine of whoever ran it, which helps nobody reproduce."""
    path = Path(path)
    root = repository_root(path)
    try:
        return str(path.resolve().relative_to(root)) if root else path.name
    except ValueError:
        return path.name


def find(library, config_hash: str, kind: str | None = None) -> list[dict]:
    """The versions, in git history, whose fingerprint is `config_hash`. The fingerprint
    is DERIVED, written nowhere (a `git log -S` finds nothing): each version is read back
    and hashed — a few tens of milliseconds each, and never out of sync."""
    import yaml

    from ._config import META, TYPE, Config

    target = str(config_hash).strip().lower()
    root = repository_root(library.root)
    if root is None:
        return []
    kinds = {kind: library.kinds[kind]} if kind else library.kinds
    out, seen = [], set()
    for name_of_kind, folder in kinds.items():
        try:
            inside = folder.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        for line in git("log", "--format=%H|%h|%ad", "--date=short", "--", str(folder),
                        cwd=root).splitlines():
            full, short, date = (line.split("|") + ["", ""])[:3]
            for file in git("ls-tree", "--name-only", f"{full}:{inside}", cwd=root).splitlines():
                if not file.endswith((".yaml", ".yml")) or (short, file) in seen:
                    continue
                text = git("show", f"{full}:{inside}/{file}", cwd=root)
                try:
                    raw = yaml.safe_load(text) or {}
                    config = Config(name=Path(file).stem, kind=name_of_kind,
                                    path=folder / file, type=str(raw.get(TYPE, "")),
                                    kwargs={k: v for k, v in raw.items() if k not in (META, TYPE)},
                                    meta=raw.get(META) or {})
                except Exception:                 # noqa: BLE001 - unreadable version
                    continue
                if config.hash != target:
                    continue
                seen.add((short, file))
                out.append({"commit": short, "date": date, "kind": name_of_kind,
                            "path": f"{inside}/{file}", "name": config.name,
                            "display_name": config.display_name})
    return out
