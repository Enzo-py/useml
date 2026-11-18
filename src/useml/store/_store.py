"""Store — a key → value cache: a SQLite index, large values in files.

    store = useml.store.open()                       # <home>/store, or [store] root
    scores = store.declare("results/scores", cost="precious")
    scores.put({"scorer": "clip", "dataset": "cifar10"}, {"accuracy": 0.41})
    scores.get({"scorer": "clip", "dataset": "cifar10"})

KEYS. A key is a dict, a list or a string. It is serialised to canonical JSON (same fields
→ same text, whatever their order), then hashed (`key_digest`, 16 bytes of BLAKE2b). The
plain key is kept in the index: `entries()` and the diagnostics read it back.

WHAT THE STORE GUARANTEES:
  • ONE WRITE PER KEY. `put` on an existing key rewrites nothing (unless `overwrite=True`).
  • ATOMICITY. A file is written to a temporary name, then `os.replace`d; the index
    points to a file only once it is complete. A crash leaves the old value or nothing,
    never a truncated file read back as good.
  • SEVERAL PROCESSES. SQLite in WAL mode with a long timeout: parallel runs write to the
    same index without duplicates or losses. After a fork, the child opens its own
    connection.
  • ACCOUNTS. Size, creation, last access and reads per entry; hits / misses / writes per
    space and per day. Reads are counted in memory and written in batches: one SQLite
    write per read would make the cache slower than what it saves.

SMALL INLINE, LARGE IN A FILE. A JSON value under `INLINE_LIMIT` bytes lives in the index;
larger values, bytes and numpy arrays live under `<root>/files/<space>/<2 hex>/<digest>`.
A producer with its own format writes through `put_file`.

NUMPY. An array, or a dict of arrays, goes to a `.npz`. Compressed when it holds integers
or booleans (masks shrink a lot); uncompressed when it holds floats, which barely compress
and would only cost CPU at every read.

COMPATIBLE WITH INDEXES OF SCHEMA 2 (an earlier format): opening one upgrades
it in place (schema 3 adds the `spaces` table). No file moves.
"""
from __future__ import annotations

import atexit
import hashlib
import io
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ._spaces import Space, cost_rank

INLINE_LIMIT = 64 * 1024
SCHEMA_VERSION = 3
_ARRAY_KEY = "__array__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    space TEXT NOT NULL,
    digest TEXT NOT NULL,
    key_json TEXT NOT NULL,
    kind TEXT NOT NULL,
    value TEXT,
    file TEXT,
    size INTEGER NOT NULL,
    created REAL NOT NULL,
    accessed REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    version TEXT,
    meta TEXT,
    PRIMARY KEY (space, digest)
);
CREATE INDEX IF NOT EXISTS entries_accessed ON entries (space, accessed);
CREATE TABLE IF NOT EXISTS pins (
    space TEXT NOT NULL,
    digest TEXT NOT NULL,
    owner TEXT NOT NULL,
    PRIMARY KEY (space, digest, owner)
);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    space TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    misses INTEGER NOT NULL DEFAULT 0,
    writes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, space)
);
CREATE TABLE IF NOT EXISTS migrations (
    source TEXT PRIMARY KEY,
    migrated REAL NOT NULL,
    entries INTEGER NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS file_digests (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spaces (
    name TEXT PRIMARY KEY,
    cost TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    producer TEXT NOT NULL DEFAULT '',
    declared REAL NOT NULL
);
"""


class UndeclaredSpaceError(KeyError):
    pass


def canonical_key(key) -> str:
    """The key as canonical JSON: same fields → same text, whatever their order."""
    return json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def key_digest(key) -> str:
    return hashlib.blake2b(canonical_key(key).encode("utf-8"), digest_size=16).hexdigest()


class Store:
    """A store under `root`: `root/index.sqlite`, `root/files/<space>/…`, and
    `root/managed/<space>/` for caches with their own format (see `managed`)."""

    def __init__(self, root, *, strict: bool = True) -> None:
        """`strict` — refuse to write into an undeclared space (the default)."""
        self.root = Path(root).expanduser()
        self.strict = strict
        self.files_root = self.root / "files"
        self.managed_root = self.root / "managed"
        self.index_path = self.root / "index.sqlite"
        self._connection = None
        self._connection_pid = None
        self._lock = threading.RLock()
        self._declared: dict[str, Space] | None = None
        self._pending_hits: dict = {}
        self._pending_usage: dict = {}
        self._pending_operations = 0
        self._digest_memo: dict = {}
        atexit.register(self.flush)

    # ── connection ───────────────────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None or self._connection_pid != os.getpid():
            self.root.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.index_path, timeout=120, isolation_level=None,
                                         check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT value FROM settings WHERE name = 'schema'").fetchone()
            if row is None:
                connection.execute("INSERT INTO settings (name, value) VALUES ('schema', ?)",
                                   (str(SCHEMA_VERSION),))
            elif int(row[0]) < 2:
                raise RuntimeError(
                    f"{self.index_path} uses schema {row[0]}, older than the oldest "
                    f"this useml reads (2)")
            elif int(row[0]) < SCHEMA_VERSION:
                # 2 → 3: the `spaces` table, created above by the schema script.
                connection.execute("UPDATE settings SET value = ? WHERE name = 'schema'",
                                   (str(SCHEMA_VERSION),))
            elif int(row[0]) > SCHEMA_VERSION:
                raise RuntimeError(f"{self.index_path} uses schema {row[0]}, newer than "
                                   f"this useml ({SCHEMA_VERSION}): upgrade useml")
            self._connection = connection
            self._connection_pid = os.getpid()
            self._declared = None
        return self._connection

    @contextmanager
    def _transaction(self):
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    # ── spaces ───────────────────────────────────────────────────────────────

    def declare(self, name: str, *, cost: str = "expensive", description: str = "",
                producer: str = "") -> "SpaceView":
        """Declare (or re-declare) a space and return a view bound to it. Re-declaring
        with another cost updates it: the code is the source of truth."""
        space = Space(name, cost, description, producer)
        known = self.declared().get(name)
        if known != space:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO spaces (name, cost, description, producer, declared) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET cost = "
                    "excluded.cost, description = excluded.description, producer = "
                    "excluded.producer", (name, cost, description, producer, time.time()))
            self._declared = None
        return SpaceView(self, name)

    def declared(self) -> dict[str, Space]:
        """Every declared space, read from the index (cached in memory)."""
        if self._declared is None:
            with self._lock:
                rows = self.connection.execute(
                    "SELECT name, cost, description, producer FROM spaces").fetchall()
            self._declared = {row[0]: Space(*row) for row in rows}
        return self._declared

    def space(self, name: str) -> "SpaceView":
        """A view on a DECLARED space."""
        self._check_space(name)
        return SpaceView(self, name)

    def _check_space(self, name: str) -> None:
        if self.strict and name not in self.declared():
            self._declared = None                # declared by another process meanwhile?
            if name not in self.declared():
                known = ", ".join(sorted(self.declared())) or "none"
                raise UndeclaredSpaceError(
                    f"space {name!r} is not declared in {self.root}. Call "
                    f"store.declare({name!r}, cost=...) first. Declared: {known}")

    def cost_rank(self, name: str) -> int:
        space = self.declared().get(name)
        return cost_rank(space.cost if space else None)

    # ── reading ──────────────────────────────────────────────────────────────

    def _row(self, space: str, digest: str):
        with self._lock:
            return self.connection.execute(
                "SELECT kind, value, file FROM entries WHERE space = ? AND digest = ?",
                (space, digest)).fetchone()

    def contains(self, space: str, key) -> bool:
        return self._row(space, key_digest(key)) is not None

    def meta(self, space: str, key) -> dict | None:
        """The metadata of an entry (without reading it), or None if absent."""
        with self._lock:
            row = self.connection.execute(
                "SELECT meta FROM entries WHERE space = ? AND digest = ?",
                (space, key_digest(key))).fetchone()
        if row is None:
            return None
        return json.loads(row[0]) if row[0] else {}

    def get(self, space: str, key, default=None):
        """The value, or `default`. A missing or unreadable file is a MISS and its entry is
        removed: serving half a value would be worse than recomputing it."""
        digest = key_digest(key)
        row = self._row(space, digest)
        if row is None:
            self._count(space, "misses")
            return default
        kind, inline_value, file_name = row
        try:
            value = self._decode(kind, inline_value, file_name)
        except (OSError, ValueError, EOFError):
            self.delete_digest(space, digest)
            self._count(space, "misses")
            return default
        self._count(space, "hits", digest)
        return value

    def path(self, space: str, key) -> Path | None:
        """The file of an entry (written by `put_file`, or a large value), or None."""
        digest = key_digest(key)
        row = self._row(space, digest)
        if row is None or row[2] is None:
            self._count(space, "misses")
            return None
        path = self.files_root / row[2]
        if not path.exists():
            self.delete_digest(space, digest)
            self._count(space, "misses")
            return None
        self._count(space, "hits", digest)
        return path

    def _decode(self, kind: str, inline_value, file_name):
        if kind == "json":
            if inline_value is not None:
                return json.loads(inline_value)
            return json.loads((self.files_root / file_name).read_text(encoding="utf-8"))
        if kind == "npz":
            import numpy as np

            with np.load(self.files_root / file_name, allow_pickle=False) as archive:
                if archive.files == [_ARRAY_KEY]:
                    return archive[_ARRAY_KEY]
                return {name: archive[name] for name in archive.files}
        if kind == "bytes":
            return (self.files_root / file_name).read_bytes()
        if kind == "file":
            path = self.files_root / file_name
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        raise ValueError(f"unknown entry kind {kind!r}")

    # ── writing ──────────────────────────────────────────────────────────────

    def put(self, space: str, key, value, *, version: str | None = None,
            meta: dict | None = None, overwrite: bool = False,
            owner: str | None = None) -> bool:
        """Write `value` under `key`; True if something was written. On an existing key
        nothing is rewritten, unless `overwrite=True`. `owner` pins the entry (see `pin`)."""
        self._check_space(space)
        digest = key_digest(key)
        if not overwrite and self._row(space, digest) is not None:
            if owner:
                self.pin(space, key, owner)
            return False
        kind, inline_value, payload, extension = _encode(value)
        file_name = None
        size = len(inline_value.encode("utf-8")) if inline_value is not None else 0
        if payload is not None:
            file_name = _relative_file(space, digest, extension)
            size = _write_atomically(self.files_root / file_name, payload)
        self._record(space, digest, key, kind, inline_value, file_name, size, version, meta,
                     overwrite)
        if owner:
            self.pin(space, key, owner)
        return True

    def put_file(self, space: str, key, write, *, extension: str = "",
                 version: str | None = None, meta: dict | None = None,
                 overwrite: bool = False, owner: str | None = None) -> Path:
        """For a producer with its own format: `write(temporary_path)` fills a temporary
        file, which becomes the entry. Returns the final path. On an existing key (without
        `overwrite`), `write` is not called."""
        self._check_space(space)
        digest = key_digest(key)
        row = self._row(space, digest)
        if row is not None and row[2] is not None and not overwrite:
            existing = self.files_root / row[2]
            if existing.exists():
                if owner:
                    self.pin(space, key, owner)
                return existing
        file_name = _relative_file(space, digest, extension)
        final_path = self.files_root / file_name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{threading.get_ident()}.tmp{extension}")
        try:
            write(temporary_path)
            os.replace(temporary_path, final_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._record(space, digest, key, "file", None, file_name, final_path.stat().st_size,
                     version, meta, True)
        if owner:
            self.pin(space, key, owner)
        return final_path

    def import_file(self, space: str, key, source, *, extension: str = "",
                    meta: dict | None = None, remove_source: bool = False) -> Path:
        """Make an existing file an entry (atomic copy); remove the source on request."""
        import shutil

        source = Path(source)
        final = self.put_file(space, key, lambda temporary: shutil.copyfile(source, temporary),
                              extension=extension,
                              meta={**(meta or {}), "imported_from": str(source)})
        if remove_source:
            source.unlink(missing_ok=True)
        return final

    def _record(self, space, digest, key, kind, inline_value, file_name, size, version,
                meta, overwrite) -> None:
        now = time.time()
        statement = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
        with self._transaction() as connection:
            previous = connection.execute(
                "SELECT file, created, hits FROM entries WHERE space = ? AND digest = ?",
                (space, digest)).fetchone()
            created, hits = (previous[1], previous[2]) if previous else (now, 0)
            connection.execute(
                f"{statement} INTO entries (space, digest, key_json, kind, value, file, size,"
                " created, accessed, hits, version, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (space, digest, canonical_key(key), kind, inline_value, file_name, size,
                 created, now, hits, version,
                 json.dumps(meta, default=str) if meta else None))
        if overwrite and previous and previous[0] and previous[0] != file_name:
            (self.files_root / previous[0]).unlink(missing_ok=True)
        self._count(space, "writes")

    def delete(self, space: str, key) -> bool:
        return self.delete_digest(space, key_digest(key))

    def delete_digest(self, space: str, digest: str) -> bool:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT file FROM entries WHERE space = ? AND digest = ?",
                (space, digest)).fetchone()
            connection.execute("DELETE FROM entries WHERE space = ? AND digest = ?",
                               (space, digest))
            connection.execute("DELETE FROM pins WHERE space = ? AND digest = ?",
                               (space, digest))
        if row and row[0]:
            (self.files_root / row[0]).unlink(missing_ok=True)
        return row is not None

    # ── identity of a file by its content ────────────────────────────────────

    def file_digest(self, path) -> str:
        """Hash of the CONTENT of `path`, remembered by (path, size, mtime): a file is read
        once while it does not change, and an image rewritten in place changes its hash."""
        path = Path(path)
        status = path.stat()
        resolved = str(path.resolve())
        signature = (status.st_size, status.st_mtime_ns)
        cached = self._digest_memo.get(resolved)
        if cached is not None and cached[0] == signature:
            return cached[1]
        with self._lock:
            row = self.connection.execute(
                "SELECT size, mtime_ns, digest FROM file_digests WHERE path = ?",
                (resolved,)).fetchone()
        if row is not None and (row[0], row[1]) == signature:
            self._digest_memo[resolved] = (signature, row[2])
            return row[2]
        digest = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO file_digests (path, size, mtime_ns, digest)"
                " VALUES (?, ?, ?, ?)", (resolved, signature[0], signature[1], value))
        self._digest_memo[resolved] = (signature, value)
        return value

    # ── pins ─────────────────────────────────────────────────────────────────

    def pin(self, space: str, key, owner: str) -> None:
        """`owner` (an experiment, typically) needs this entry: `gc` keeps it while a pin
        holds it."""
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO pins (space, digest, owner) VALUES (?, ?, ?)",
                (space, key_digest(key), owner))

    def unpin_owner(self, owner: str) -> int:
        with self._transaction() as connection:
            return connection.execute("DELETE FROM pins WHERE owner = ?", (owner,)).rowcount

    def pinned_digests(self) -> set:
        with self._lock:
            return set(self.connection.execute("SELECT space, digest FROM pins").fetchall())

    # ── managed folders ──────────────────────────────────────────────────────

    def managed(self, space: str) -> Path:
        """`<root>/managed/<space>/`, created: for a cache too big or too numerous for one
        index row per item (memory-mapped shards, millions of rows). It lives IN the store
        (same root, same `move`, counted by `spaces()`), with its own format."""
        self._check_space(space)
        path = self.managed_root / space
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── listing ──────────────────────────────────────────────────────────────

    def keys(self, space: str) -> list:
        """The PLAIN keys of the space."""
        self.flush()
        with self._lock:
            rows = self.connection.execute(
                "SELECT key_json FROM entries WHERE space = ?", (space,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def entries(self, space: str | None = None) -> list[dict]:
        self.flush()
        query = ("SELECT space, digest, key_json, kind, file, size, created, accessed, hits,"
                 " version, meta FROM entries")
        parameters: tuple = ()
        if space is not None:
            query += " WHERE space = ?"
            parameters = (space,)
        with self._lock:
            rows = self.connection.execute(query, parameters).fetchall()
        names = ("space", "digest", "key_json", "kind", "file", "size", "created",
                 "accessed", "hits", "version", "meta")
        return [dict(zip(names, row)) for row in rows]

    def spaces(self, *, managed: bool = True) -> list[dict]:
        """Per space: entries, bytes, oldest, last access, reads, pins, cost — plus the
        size of its managed folder (`managed_bytes`), counted once per inode."""
        from ._maintenance import real_size

        self.flush()
        with self._lock:
            rows = self.connection.execute(
                "SELECT e.space, COUNT(*), SUM(e.size), MIN(e.created), MAX(e.accessed),"
                " SUM(e.hits), SUM(CASE WHEN e.file IS NULL THEN 0 ELSE 1 END),"
                " (SELECT COUNT(DISTINCT p.digest) FROM pins p WHERE p.space = e.space)"
                " FROM entries e GROUP BY e.space").fetchall()
        names = ("space", "entries", "bytes", "oldest", "last_access", "hits", "files",
                 "pinned")
        out = {row[0]: dict(zip(names, row)) for row in rows}
        for name, declared in self.declared().items():
            out.setdefault(name, {"space": name, "entries": 0, "bytes": 0, "oldest": None,
                                  "last_access": None, "hits": 0, "files": 0, "pinned": 0})
        for row in out.values():
            declared = self.declared().get(row["space"])
            row["cost"] = declared.cost if declared else None
            row["description"] = declared.description if declared else ""
            row["managed_bytes"] = 0
            if managed:
                size, _, latest = real_size(self.managed_root / row["space"])
                row["managed_bytes"] = size
                row["bytes"] = (row["bytes"] or 0) + size
                if latest:
                    row["last_access"] = max(row["last_access"] or 0, latest)
        return sorted(out.values(), key=lambda row: -(row["bytes"] or 0))

    def usage(self, days: int = 30) -> list[dict]:
        self.flush()
        since = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        with self._lock:
            rows = self.connection.execute(
                "SELECT day, space, hits, misses, writes FROM usage WHERE day >= ?"
                " ORDER BY day", (since,)).fetchall()
        return [dict(zip(("day", "space", "hits", "misses", "writes"), row)) for row in rows]

    def total_bytes(self) -> int:
        return sum(row["bytes"] or 0 for row in self.spaces())

    def setting(self, name: str):
        with self._lock:
            row = self.connection.execute(
                "SELECT value FROM settings WHERE name = ?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_setting(self, name: str, value) -> None:
        with self._transaction() as connection:
            connection.execute("INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)",
                               (name, json.dumps(value)))

    # ── accounts ─────────────────────────────────────────────────────────────

    def _count(self, space: str, what: str, digest: str | None = None) -> None:
        with self._lock:
            day = time.strftime("%Y-%m-%d")
            counters = self._pending_usage.setdefault(
                (day, space), {"hits": 0, "misses": 0, "writes": 0})
            counters[what] += 1
            if digest is not None:
                self._pending_hits[(space, digest)] = \
                    self._pending_hits.get((space, digest), 0) + 1
            self._pending_operations += 1
            if self._pending_operations >= 500:
                self.flush()

    def flush(self) -> None:
        """Write the pending access counts (also run at process exit)."""
        with self._lock:
            if not self._pending_operations:
                return
            hits, usage = self._pending_hits, self._pending_usage
            self._pending_hits, self._pending_usage, self._pending_operations = {}, {}, 0
            now = time.time()
            try:
                with self._transaction() as connection:
                    connection.executemany(
                        "UPDATE entries SET hits = hits + ?, accessed = ?"
                        " WHERE space = ? AND digest = ?",
                        [(count, now, space, digest)
                         for (space, digest), count in hits.items()])
                    connection.executemany(
                        "INSERT INTO usage (day, space, hits, misses, writes)"
                        " VALUES (?, ?, ?, ?, ?) ON CONFLICT (day, space) DO UPDATE SET"
                        " hits = hits + excluded.hits, misses = misses + excluded.misses,"
                        " writes = writes + excluded.writes",
                        [(day, space, counters["hits"], counters["misses"],
                          counters["writes"]) for (day, space), counters in usage.items()])
            except sqlite3.Error:
                pass                     # lost counts must never make a run fail

    def vacuum(self) -> None:
        self.flush()
        with self._lock:
            self.connection.execute("VACUUM")

    def close(self) -> None:
        self.flush()
        if self._connection is not None and self._connection_pid == os.getpid():
            self._connection.close()
        self._connection = None

    def __repr__(self) -> str:
        return f"Store({str(self.root)!r})"


class SpaceView:
    """A store bound to one space: `view.put(key, value)` is `store.put(space, key, value)`."""

    def __init__(self, store: Store, name: str) -> None:
        self.store, self.name = store, name

    def __getattr__(self, attribute):
        method = getattr(self.store, attribute)
        if attribute in ("put", "put_file", "get", "path", "contains", "meta", "delete",
                         "pin", "keys", "entries", "managed", "import_file"):
            return lambda *args, **kwargs: method(self.name, *args, **kwargs)
        raise AttributeError(attribute)

    def __contains__(self, key) -> bool:
        return self.store.contains(self.name, key)

    def __repr__(self) -> str:
        return f"SpaceView({self.name!r} in {self.store.root})"


# ── encoding ──────────────────────────────────────────────────────────────────

def _encode(value):
    """(kind, inline value | None, bytes to write | None, extension)."""
    if isinstance(value, (bytes, bytearray)):
        return "bytes", None, bytes(value), ".bin"
    arrays = _as_arrays(value)
    if arrays is not None:
        import numpy as np

        buffer = io.BytesIO()
        compress = any(array.dtype.kind in "biu" for array in arrays.values())
        (np.savez_compressed if compress else np.savez)(buffer, **arrays)
        return "npz", None, buffer.getvalue(), ".npz"
    text = json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= INLINE_LIMIT:
        return "json", text, None, ""
    return "json", None, text.encode("utf-8"), ".json"


def _as_arrays(value) -> dict | None:
    """{name: array} for an array or a non-empty dict of arrays, else None."""
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, np.ndarray):
        return {_ARRAY_KEY: value}
    if isinstance(value, dict) and value and \
            all(isinstance(item, np.ndarray) for item in value.values()):
        return value
    return None


def _relative_file(space: str, digest: str, extension: str) -> str:
    return f"{space}/{digest[:2]}/{digest}{extension}"


def _write_atomically(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary_path.write_bytes(payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return len(payload)
