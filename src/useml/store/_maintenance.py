"""Check, clean, move a store. Everything that deletes is a DRY RUN by default: the function
returns the plan, and only carries it out with `apply=True` (or through `collect_garbage`).
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from ._spaces import COSTS


def real_size(path) -> tuple[int, int, float]:
    """(bytes, files, last modification) under `path`, each inode counted ONCE: a pool of
    hard links is not counted twice."""
    path = Path(path)
    seen, total, count, latest = set(), 0, 0, 0.0
    if not path.exists():
        return 0, 0, 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            status = entry.stat(follow_symlinks=False)
                            count += 1
                            latest = max(latest, status.st_mtime)
                            if (status.st_dev, status.st_ino) not in seen:
                                seen.add((status.st_dev, status.st_ino))
                                total += status.st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total, count, latest


def check(store, *, deep: bool = False) -> dict:
    """The store's inconsistencies:
      • `missing` — an entry whose file disappeared (it will be recomputed);
      • `orphans` — a file under `files/` that no entry references (lost space);
      • `temporaries` — a `.tmp` left by an interrupted write;
      • `unreadable` — with `deep`, a JSON entry that does not parse."""
    entries = store.entries()
    referenced = {entry["file"] for entry in entries if entry["file"]}
    missing = [entry for entry in entries
               if entry["file"] and not (store.files_root / entry["file"]).exists()]
    orphans, temporaries = [], []
    if store.files_root.exists():
        for path in store.files_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(store.files_root).as_posix()
            if ".tmp" in path.name:
                temporaries.append(path)
            elif relative not in referenced:
                orphans.append(path)
    unreadable = []
    if deep:
        for entry in entries:
            if entry["kind"] != "json" or not entry["file"]:
                continue
            path = store.files_root / entry["file"]
            if not path.exists():
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                unreadable.append(entry)
    return {"missing": missing, "orphans": orphans, "temporaries": temporaries,
            "unreadable": unreadable}


def repair(store, report: dict, *, temporaries_older_than: float = 3600) -> dict:
    """Carry out a `check`: drop entries without a file or unreadable, delete orphans, and
    temporaries older than `temporaries_older_than` seconds (a younger one may belong to a
    write in progress in another process)."""
    for entry in report["missing"] + report["unreadable"]:
        store.delete_digest(entry["space"], entry["digest"])
    limit = time.time() - temporaries_older_than
    removed_temporaries = 0
    for path in report["temporaries"]:
        try:
            if path.stat().st_mtime < limit:
                path.unlink()
                removed_temporaries += 1
        except OSError:
            pass
    for path in report["orphans"]:
        path.unlink(missing_ok=True)
    return {"missing": len(report["missing"]), "unreadable": len(report["unreadable"]),
            "orphans": len(report["orphans"]), "temporaries": removed_temporaries}


def garbage_plan(store, *, spaces: list[str] | None = None,
                 older_than_days: float | None = None, max_bytes: int | None = None,
                 include_pinned: bool = False, include_precious: bool = False) -> list[dict]:
    """The entries to remove.

    `older_than_days` — not read for that long. `max_bytes` — remove until the whole
    store (or the named spaces) fits under this total, what is FASTEST TO RECOMPUTE first
    (cheap, then expensive), and within one cost the least recently read. Never chosen: a
    pinned entry (an experiment keeps it), unless `include_pinned`; an entry of a
    `precious` space, unless `include_precious` or the space is named in `spaces`.

    Managed folders count in the total but are not evicted here: their format is their
    owner's. If they alone exceed the budget, the plan cannot reach it, and says so in
    `garbage_report`."""
    pinned = set() if include_pinned else store.pinned_digests()
    precious = COSTS.index("precious")
    all_entries = store.entries()
    candidates = [entry for entry in all_entries
                  if (spaces is None or entry["space"] in spaces)
                  and (entry["space"], entry["digest"]) not in pinned
                  and (include_precious or spaces is not None
                       or store.cost_rank(entry["space"]) < precious)]
    candidates.sort(key=lambda entry: (store.cost_rank(entry["space"]), entry["accessed"]))
    plan, chosen = [], set()
    if older_than_days is not None:
        limit = time.time() - older_than_days * 86400
        for entry in candidates:
            if entry["accessed"] < limit:
                plan.append({**entry, "reason": f"unread for {older_than_days:g} days"})
                chosen.add((entry["space"], entry["digest"]))
    if max_bytes is not None:
        rows = store.spaces()
        total = sum(row["bytes"] or 0 for row in rows
                    if spaces is None or row["space"] in spaces)
        total -= sum(entry["size"] for entry in plan)
        for entry in candidates:
            if total <= max_bytes:
                break
            if (entry["space"], entry["digest"]) in chosen:
                continue
            plan.append({**entry, "reason": f"over the {max_bytes} B budget"})
            chosen.add((entry["space"], entry["digest"]))
            total -= entry["size"]
    return plan


def collect_garbage(store, plan: list[dict]) -> int:
    """Delete the entries of a plan; returns the bytes freed."""
    freed = 0
    for entry in plan:
        if store.delete_digest(entry["space"], entry["digest"]):
            freed += entry["size"]
    return freed


def move(store, destination, *, delete_source: bool = False) -> dict:
    """Copy the store to `destination` (the index through SQLite's backup API, consistent
    even while a run writes), check that each file arrived with its size, and delete the
    source only then, and only with `delete_source`."""
    destination = Path(destination).expanduser().resolve()
    source = store.root.resolve()
    if destination == source:
        raise ValueError("destination is the store itself")
    if (destination / "index.sqlite").exists():
        raise FileExistsError(f"{destination} already holds a store: refusing to merge")
    destination.mkdir(parents=True, exist_ok=True)
    store.flush()
    target = sqlite3.connect(destination / "index.sqlite")
    with target:
        store.connection.backup(target)
    target.close()
    for folder in ("files", "managed"):
        if (source / folder).exists():
            shutil.copytree(source / folder, destination / folder, dirs_exist_ok=True,
                            symlinks=True, ignore=shutil.ignore_patterns("*.tmp*"))
    problems = [entry["file"] for entry in store.entries() if entry["file"] and (
        not (destination / "files" / entry["file"]).exists()
        or (destination / "files" / entry["file"]).stat().st_size != entry["size"])]
    if problems:
        raise RuntimeError(f"{len(problems)} file(s) did not copy correctly, source kept: "
                           f"{problems[:5]}")
    entries = len(store.entries())
    if delete_source:
        store.close()
        shutil.rmtree(source)
    return {"destination": str(destination), "entries": entries}


def merge_by_rename(source, target) -> tuple[int, int]:
    """Move `source` into `target` by renames, overwriting nothing. Returns (entries moved,
    entries left because present on both sides). Raises OSError(EXDEV) across disks."""
    source, target = Path(source), Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(source, target)
            return 1, 0
        except FileNotFoundError:
            return 0, 0
        except OSError as error:
            if error.errno == errno.EXDEV or not target.exists():
                raise
    moved, left = 0, 0
    for child in list(source.iterdir()):
        destination = target / child.name
        if not destination.exists() and not destination.is_symlink():
            try:
                os.rename(child, destination)
                moved += 1
            except FileNotFoundError:
                pass
        elif child.is_dir() and not child.is_symlink() and destination.is_dir():
            child_moved, child_left = merge_by_rename(child, destination)
            moved, left = moved + child_moved, left + child_left
        else:
            left += 1
    try:
        source.rmdir()
    except OSError:
        pass
    return moved, left


def adopt(store, space: str, former) -> dict:
    """Move a former folder of `space` (written before the store existed) into its managed
    folder, by renames: instant on one disk, hard links survive. Across disks nothing
    moves and `other_disk` says so."""
    former = Path(former).expanduser()
    target = store.managed(space)
    report = {"space": space, "source": str(former), "target": str(target), "moved": 0,
              "left": 0, "other_disk": False}
    if not former.exists() or former.resolve() == target.resolve():
        return report
    try:
        if not any(target.iterdir()):
            target.rmdir()                       # let a single rename take the whole folder
        report["moved"], report["left"] = merge_by_rename(former, target)
    except OSError as error:
        target.mkdir(parents=True, exist_ok=True)
        if error.errno != errno.EXDEV:
            raise
        report["other_disk"] = True
    return report


def free_disk_bytes(path) -> int:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free
