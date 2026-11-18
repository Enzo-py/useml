"""The prepared INDEX of a dataset: its records, written once, read by position.

    <dataset root>/.useml/index/
        records.jsonl        one JSON object per record (paths stored relative)
        offsets.npy          (N + 1,) uint64 byte offsets: record i is bytes [o[i], o[i+1])
        split-<name>.npy     positions of each split
        meta.json            count, fields, path fields, splits, fingerprint

`len()` reads one small array; `ds[i]` seeks to one line. Nothing is parsed until asked,
so a dataset of millions of records opens instantly. Written in a temporary folder, then
renamed: a crash never leaves half an index.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .. import _progress


def index_folder(root: Path) -> Path:
    return root / ".useml" / "index"


def write(root: Path, records, *, splits: dict, path_fields: list, fingerprint: str,
          expected: int | None = None) -> dict:
    """Write `records` (an iterable of dicts) and the split positions."""
    final = index_folder(root)
    temporary = final.with_name(f"index.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    offsets = [0]
    fields: dict = {}
    with open(temporary / "records.jsonl", "wb") as file:
        for record in _progress.bar(records, total=expected, desc=f"indexing {root.name}",
                                    unit="record"):
            line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            file.write(line)
            offsets.append(offsets[-1] + len(line))
            if len(offsets) <= 1001:                     # field names from the first records
                for key, value in record.items():
                    fields.setdefault(key, type(value).__name__)
    np.save(temporary / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    for name, positions in splits.items():
        np.save(temporary / f"split-{name}.npy", np.asarray(positions, dtype=np.int64))
    meta = {"count": len(offsets) - 1, "fields": fields, "path_fields": list(path_fields),
            "splits": {name: int(len(positions)) for name, positions in splits.items()},
            "fingerprint": fingerprint, "written": time.time()}
    (temporary / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(temporary, final)
    return meta


class Index:
    """Read access to a written index (lazy)."""

    def __init__(self, root: Path) -> None:
        self.folder = index_folder(root)
        self._meta = None
        self._offsets = None
        self._file = None

    def exists(self) -> bool:
        return (self.folder / "meta.json").exists()

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads((self.folder / "meta.json").read_text())
        return self._meta

    @property
    def offsets(self) -> np.ndarray:
        if self._offsets is None:
            self._offsets = np.load(self.folder / "offsets.npy", mmap_mode="r")
        return self._offsets

    def __len__(self) -> int:
        return int(self.meta["count"])

    def split(self, name: str) -> np.ndarray:
        path = self.folder / f"split-{name}.npy"
        if not path.exists():
            known = ", ".join(self.meta["splits"]) or "none"
            raise KeyError(f"no split {name!r}; splits: {known}")
        return np.load(path)

    def _handle(self):
        if self._file is None or self._file.closed:
            self._file = open(self.folder / "records.jsonl", "rb")
        return self._file

    def read(self, position: int) -> dict:
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        handle = self._handle()
        handle.seek(start)
        return json.loads(handle.read(end - start))

    def read_many(self, positions) -> list[dict]:
        """Records at `positions`, read in file order (sequential), returned in order."""
        positions = np.asarray(positions, dtype=np.int64)
        order = np.argsort(positions, kind="stable")
        out: list = [None] * len(positions)
        for index in order:
            out[index] = self.read(int(positions[index]))
        return out

    def iterate(self):
        """Every record, streamed in order."""
        with open(self.folder / "records.jsonl", "rb") as file:
            for line in file:
                yield json.loads(line)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.folder.iterdir() if path.is_file())
