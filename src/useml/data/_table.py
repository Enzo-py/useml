"""A TABLE: every row one processor (with one set of parameters) has computed, whatever the
dataset it came from. Rows are found by the digest of their content key.

    <data root>/tables/<processor>/<fingerprint>/
        manifest.json                      identity, outputs (shape, dtype), code hash
        shards/<id>.keys.npy               (n, 16) uint8 digests — written LAST
        shards/<id>.<output>.npy           (n, ...) rows, float16 / float32 / int8 / ints
        shards/<id>.<output>.scale.npy     (n,) float32, for int8 rows only
        .lock                              held by the one process writing

APPEND-ONLY SHARDS. A shard is never rewritten. Its keys file is written last, so a shard
without one is incomplete (an interrupted write) and is ignored, then removed by the next
writer. Readers need no lock: they see complete shards only.

ONE WRITER. Writing takes an exclusive lock on `.lock` (fcntl). A second writer waits,
says for whom, then computes only what the first one did not. No row is ever stored twice.

LAZY. Opening a table reads nothing. The index (16 bytes per row: an 8-byte prefix and a
row number, sorted) is built on the first lookup; shards are memory-mapped, so reading a
row reads only its pages.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from .. import _progress
from ._keys import DIGEST_SIZE, prefixes

SHARD_BYTES = 256 * 1024 * 1024          # a shard is flushed at ~256 MB of rows


class Table:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.shards_path = self.path / "shards"
        self._manifest: dict | None = None
        self._index_shards: tuple = ()
        self._digests: np.ndarray | None = None      # (N, 16) all rows, shard order
        self._sorted_prefix: np.ndarray | None = None
        self._order: np.ndarray | None = None
        self._offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._maps: dict = {}

    # ── manifest ─────────────────────────────────────────────────────────────

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            path = self.path / "manifest.json"
            self._manifest = json.loads(path.read_text()) if path.exists() else {}
        return self._manifest

    def exists(self) -> bool:
        return (self.path / "manifest.json").exists()

    @property
    def outputs(self) -> dict:
        return self.manifest.get("outputs", {})

    def write_manifest(self, manifest: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / f".manifest.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, indent=1, default=str))
        os.replace(temporary, self.path / "manifest.json")
        self._manifest = manifest

    # ── shards and index ─────────────────────────────────────────────────────

    def shards(self) -> list[str]:
        """Complete shards, oldest first."""
        if not self.shards_path.is_dir():
            return []
        return sorted(path.name[:-len(".keys.npy")]
                      for path in self.shards_path.glob("*.keys.npy"))

    def _ensure_index(self) -> None:
        shards = tuple(self.shards())
        if shards == self._index_shards and self._digests is not None:
            return
        parts, offsets = [], [0]
        for shard in _progress.bar(shards, desc=f"indexing {self.path.parent.name}",
                                   unit="shard"):
            keys = np.load(self.shards_path / f"{shard}.keys.npy")
            parts.append(keys.reshape(-1, DIGEST_SIZE))
            offsets.append(offsets[-1] + len(keys))
        self._digests = np.concatenate(parts) if parts else \
            np.zeros((0, DIGEST_SIZE), dtype=np.uint8)
        prefix = prefixes(self._digests)
        self._order = np.argsort(prefix, kind="stable")
        self._sorted_prefix = prefix[self._order]
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._index_shards = shards

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._digests)

    def locate(self, digests: np.ndarray) -> np.ndarray:
        """(M,) global row of each digest, -1 where the table does not have it."""
        self._ensure_index()
        digests = np.ascontiguousarray(digests, dtype=np.uint8).reshape(-1, DIGEST_SIZE)
        rows = np.full(len(digests), -1, dtype=np.int64)
        if not len(digests) or not len(self._digests):
            return rows
        wanted = prefixes(digests)
        position = np.searchsorted(self._sorted_prefix, wanted)
        inside = position < len(self._sorted_prefix)
        candidate = np.where(inside, position, 0)
        same_prefix = inside & (self._sorted_prefix[candidate] == wanted)
        found = self._order[candidate]
        exact = same_prefix & np.all(self._digests[found] == digests, axis=1)
        rows[exact] = found[exact]
        # Two digests sharing their first 8 bytes (rare): scan the run of equal prefixes.
        for index in np.flatnonzero(same_prefix & ~exact):
            cursor = position[index] + 1
            while cursor < len(self._sorted_prefix) and \
                    self._sorted_prefix[cursor] == wanted[index]:
                row = self._order[cursor]
                if np.array_equal(self._digests[row], digests[index]):
                    rows[index] = row
                    break
                cursor += 1
        return rows

    def _map(self, shard: str, name: str):
        key = (shard, name)
        if key not in self._maps:
            self._maps[key] = np.load(self.shards_path / f"{shard}.{name}.npy", mmap_mode="r")
        return self._maps[key]

    def read(self, rows: np.ndarray, output: str, *, dtype=np.float32) -> np.ndarray:
        """The rows `rows` (global numbers) of `output`, dequantised. Reads shard by shard,
        each in increasing row order (sequential pages), then puts them back in order."""
        self._ensure_index()
        rows = np.asarray(rows, dtype=np.int64)
        if (rows < 0).any():
            raise KeyError(f"{int((rows < 0).sum())} requested rows are not in the table")
        spec = self.outputs[output]
        stored = spec["stored"]
        out_dtype = np.dtype(spec["dtype"]) if stored not in FLOAT_STORED else np.dtype(dtype)
        out = np.empty((len(rows), *spec["shape"]), dtype=out_dtype)
        shard_of = np.searchsorted(self._offsets, rows, side="right") - 1
        for shard_index in np.unique(shard_of):
            selected = np.flatnonzero(shard_of == shard_index)
            local = rows[selected] - self._offsets[shard_index]
            order = np.argsort(local, kind="stable")
            shard = self._index_shards[shard_index]
            values = np.asarray(self._map(shard, output)[local[order]])
            if stored == "int8":
                scale = np.asarray(self._map(shard, f"{output}.scale")[local[order]])
                values = values.astype(np.float32) * scale.reshape(-1, *([1] * (values.ndim - 1)))
            out[selected[order]] = values
        return out

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.path.rglob("*") if path.is_file())

    # ── writing ──────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def lock(self):
        """Exclusive writer lock. Waits (and says so) while another process writes."""
        import fcntl

        self.path.mkdir(parents=True, exist_ok=True)
        handle = open(self.path / ".lock", "a+")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read().strip() or "another process"
                with _progress.step(f"waiting for {holder}, which is writing "
                                    f"{self.path.parent.name}/{self.path.name}", after=0):
                    fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid {os.getpid()} on {os.uname().nodename}")
            handle.flush()
            self._remove_incomplete()
            yield self
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _remove_incomplete(self) -> None:
        """Files of shards whose keys were never written (an interrupted writer). Called
        with the lock held, so no writer is in the middle of one."""
        if not self.shards_path.is_dir():
            return
        complete = set(self.shards())
        for path in self.shards_path.iterdir():
            if path.name.split(".")[0] not in complete:
                path.unlink(missing_ok=True)

    def writer(self) -> "Writer":
        return Writer(self)


FLOAT_STORED = ("float32", "float16", "int8")


def _quantize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per row: q = round(x / s), s = max|x| / 127."""
    flat = values.reshape(len(values), -1).astype(np.float32)
    scale = np.abs(flat).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    quantized = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    return quantized.reshape(values.shape), scale.astype(np.float32)


def _save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def _save_parts(path: Path, parts: list) -> None:
    """Write the concatenation of `parts` as one .npy WITHOUT building it in RAM: each part
    is copied into a memory-mapped file, then freed."""
    shape = (sum(len(part) for part in parts), *parts[0].shape[1:])
    temporary = path.with_name(f".{path.name}.tmp.npy")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=parts[0].dtype,
                                       shape=shape)
    start = 0
    while parts:
        part = parts.pop(0)
        target[start:start + len(part)] = part
        start += len(part)
    target.flush()
    del target
    os.replace(temporary, path)


class Writer:
    """Buffers rows and writes them as shards. Use with the table's lock held."""

    def __init__(self, table: Table, shard_bytes: int = SHARD_BYTES) -> None:
        self.table = table
        self.shard_bytes = shard_bytes
        self._digests: list = []
        self._rows: dict = {}
        self._bytes = 0
        self.written = 0

    def add(self, digests: np.ndarray, outputs: dict) -> None:
        """Rows are converted to their STORED dtype on arrival: the buffer holds float16
        (or int8), not the float32 the model returned — half (or a quarter) the RAM."""
        self._digests.append(np.asarray(digests, dtype=np.uint8))
        for name, values in outputs.items():
            stored = self.table.outputs[name]["stored"]
            if stored == "int8":
                values, scale = _quantize(values)
                self._rows.setdefault(f"{name}.scale", []).append(scale)
                self._bytes += scale.nbytes
            else:
                values = values.astype(stored, copy=False)
            self._rows.setdefault(name, []).append(values)
            self._bytes += values.nbytes
        if self._bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._digests:
            return
        table = self.table
        table.shards_path.mkdir(parents=True, exist_ok=True)
        shard = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        for name in list(self._rows):
            _save_parts(table.shards_path / f"{shard}.{name}.npy", self._rows.pop(name))
        digests = np.concatenate(self._digests)
        _save(table.shards_path / f"{shard}.keys.npy", digests)      # LAST: marks complete
        self.written += len(digests)
        self._digests, self._rows, self._bytes = [], {}, 0


def remove(table: Table) -> int:
    size = table.size_bytes() if table.path.exists() else 0
    shutil.rmtree(table.path, ignore_errors=True)
    return size
