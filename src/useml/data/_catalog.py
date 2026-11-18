"""Names: how a processed dataset is found again in one line.

    <data root>/names/<name>.json           which table, which source, when, how many rows
    <data root>/names/<name>.digests.npy    the source's digests, in the source's order

`load(name)` reads these two small files and returns a lazy `Processed`. Nothing else is
read until a row is asked for.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")


def data_root() -> Path:
    """`$USEML_DATA_ROOT`, else `[data] root` in the config, else `<home>/data`."""
    from .. import paths

    explicit = os.environ.get("USEML_DATA_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    configured = paths.config().get("data", {}).get("root")
    if configured:
        return Path(os.path.expandvars(str(configured))).expanduser()
    return paths.home() / "data"


def tables_root() -> Path:
    return data_root() / "tables"


def names_root() -> Path:
    return data_root() / "names"


def check_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError(f"name {name!r}: letters, digits and _ . @ + - only")
    return name


def save(name: str, *, table_path: Path, digests: np.ndarray, source: str,
         identity: dict, rows: int) -> None:
    root = names_root()
    root.mkdir(parents=True, exist_ok=True)
    check_name(name)
    temporary = root / f".{name}.{os.getpid()}.tmp.npy"
    np.save(temporary, digests)
    os.replace(temporary, root / f"{name}.digests.npy")
    record = {"name": name, "table": str(Path(table_path).relative_to(tables_root())),
              "source": source, "identity": identity, "rows": rows,
              "created": time.time(), "last_used": time.time()}
    previous = root / f"{name}.json"
    if previous.exists():
        record["created"] = json.loads(previous.read_text()).get("created", record["created"])
    temporary = root / f".{name}.{os.getpid()}.tmp.json"
    temporary.write_text(json.dumps(record, indent=1, default=str))
    os.replace(temporary, previous)


def record(name: str) -> dict:
    path = names_root() / f"{check_name(name)}.json"
    if not path.exists():
        known = ", ".join(entry["name"] for entry in ls(measure=False)) or "none"
        raise KeyError(f"no processed dataset named {name!r}. Known: {known}")
    return json.loads(path.read_text())


def load(name: str):
    """The processed dataset saved as `name`, lazily."""
    from ._table import Table
    from ._view import Processed

    entry = record(name)
    digests = np.load(names_root() / f"{name}.digests.npy", mmap_mode="r")
    try:                                         # last use, for `ls` and cleaning decisions
        entry["last_used"] = time.time()
        (names_root() / f"{name}.json").write_text(json.dumps(entry, indent=1, default=str))
    except OSError:
        pass
    return Processed(Table(tables_root() / entry["table"]), digests, name=name,
                     source=entry["source"])


def ls(*, measure: bool = True) -> list[dict]:
    """Every named processed dataset, most recently used first, with its table's size."""
    root = names_root()
    if not root.is_dir():
        return []
    out, sizes = [], {}
    for path in root.glob("*.json"):
        if path.name.startswith("."):
            continue
        entry = json.loads(path.read_text())
        if measure:
            table = tables_root() / entry["table"]
            if entry["table"] not in sizes:
                sizes[entry["table"]] = sum(file.stat().st_size for file in table.rglob("*")
                                            if file.is_file()) if table.exists() else 0
            entry["table_bytes"] = sizes[entry["table"]]
        out.append(entry)
    return sorted(out, key=lambda entry: -entry.get("last_used", 0))


def tables() -> list[dict]:
    """Every table, with its size and the names that use it."""
    from ._table import Table

    root = tables_root()
    users: dict = {}
    for entry in ls(measure=False):
        users.setdefault(entry["table"], []).append(entry["name"])
    out = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/*")):
        if not (path / "manifest.json").exists():
            continue
        table = Table(path)
        relative = str(path.relative_to(root))
        out.append({"table": relative, "bytes": table.size_bytes(),
                    "shards": len(table.shards()), "outputs": table.outputs,
                    "identity": table.manifest.get("identity"),
                    "names": users.get(relative, [])})
    return out


def remove(name: str, *, table: bool = False) -> int:
    """Forget `name`; with `table=True`, also delete its table if no other name uses it.
    Returns the bytes freed."""
    from ._table import Table, remove as remove_table

    entry = record(name)
    for suffix in (".json", ".digests.npy"):
        (names_root() / f"{name}{suffix}").unlink(missing_ok=True)
    if not table:
        return 0
    still_used = [other for other in ls(measure=False) if other["table"] == entry["table"]]
    if still_used:
        return 0
    return remove_table(Table(tables_root() / entry["table"]))
