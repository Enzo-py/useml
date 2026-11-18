"""What has been downloaded, where it is, how big, when it was last used.

One SQLite file, `<home>/hub/registry.sqlite`, shared by every process of the machine (WAL
journal, a long timeout). It is a record, not the source of truth: the files are. An entry
whose path no longer exists is reported as missing, never trusted.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS downloads (
    uri TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    created REAL NOT NULL,
    last_used REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 1
);
"""


def disk_size(path) -> int:
    """Bytes really used under `path`, each file counted once even through symbolic links
    (a Hugging Face snapshot is a folder of links to shared blobs)."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.resolve().stat().st_size
    seen, total = set(), 0
    for folder, _, names in os.walk(path, followlinks=False):
        for name in names:
            try:
                status = os.stat(os.path.join(folder, name))   # follows the link
            except OSError:
                continue                                       # dangling link
            identity = (status.st_dev, status.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += status.st_size
    return total


class Registry:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self._connection = None
        self._pid = None
        self._lock = threading.Lock()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None or self._pid != os.getpid():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=60, isolation_level=None,
                                         check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            self._connection, self._pid = connection, os.getpid()
        return self._connection

    def lookup(self, uri: str) -> Path | None:
        """The recorded path, if it still exists; marks the entry as used."""
        with self._lock:
            row = self.connection.execute(
                "SELECT path FROM downloads WHERE uri = ?", (uri,)).fetchone()
            if row is None or not Path(row[0]).exists():
                return None
            self.connection.execute(
                "UPDATE downloads SET last_used = ?, uses = uses + 1 WHERE uri = ?",
                (time.time(), uri))
            return Path(row[0])

    def record(self, uri: str, kind: str, path, size: int | None = None) -> None:
        now = time.time()
        size = disk_size(path) if size is None else int(size)
        with self._lock:
            self.connection.execute(
                "INSERT INTO downloads (uri, kind, path, size, created, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(uri) DO UPDATE SET "
                "path = excluded.path, size = excluded.size, last_used = excluded.last_used,"
                " uses = uses + 1", (uri, kind, str(path), size, now, now))

    def forget(self, uri: str) -> bool:
        with self._lock:
            return self.connection.execute(
                "DELETE FROM downloads WHERE uri = ?", (uri,)).rowcount > 0

    def entries(self, *, refresh_sizes: bool = False) -> list[dict]:
        rows = self.connection.execute(
            "SELECT uri, kind, path, size, created, last_used, uses FROM downloads "
            "ORDER BY last_used DESC").fetchall()
        out = []
        for uri, kind, path, size, created, last_used, uses in rows:
            exists = Path(path).exists()
            if refresh_sizes and exists:
                size = disk_size(path)
                with self._lock:
                    self.connection.execute("UPDATE downloads SET size = ? WHERE uri = ?",
                                            (size, uri))
            out.append({"uri": uri, "kind": kind, "path": path, "size": size if exists else 0,
                        "created": created, "last_used": last_used, "uses": uses,
                        "missing": not exists})
        return out

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
