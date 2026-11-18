"""Processed: one dataset, seen through one table. Lazy: nothing is read until asked.

    feats = useml.data.load("photos-clip")
    len(feats)                   # rows, in the dataset's order
    feats[0]                     # one row (array, or dict of arrays for several outputs)
    feats[100:200]               # a batch, read from the memory-mapped shards
    feats[[5, 3, 9]]
    for batch in feats.batches(4096, shuffle=True): ...
    feats.numpy()                # everything in memory (says how much first)

A view holds only the (N, 16) digests of its items; the table rows are located on first
access (one `searchsorted`), then read on demand.
"""
from __future__ import annotations

import numpy as np

from .. import _progress
from ._table import FLOAT_STORED, Table


class Processed:
    def __init__(self, table: Table, digests: np.ndarray, *, name: str | None = None,
                 source: str | None = None) -> None:
        self.table = table
        self.digests = digests
        self.name = name
        self.source = source
        self._rows: np.ndarray | None = None

    # ── shape ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.digests)

    @property
    def outputs(self) -> dict:
        """{output: {"shape": per-row shape, "dtype": read dtype, "stored": on disk}}."""
        return self.table.outputs

    @property
    def rows(self) -> np.ndarray:
        if self._rows is None:
            with _progress.step(f"locating {len(self)} rows"):
                self._rows = self.table.locate(self.digests)
        return self._rows

    @property
    def missing(self) -> int:
        return int((self.rows < 0).sum())

    @property
    def complete(self) -> bool:
        return self.missing == 0

    def nbytes(self) -> int:
        """Bytes of the whole view once in memory (float outputs are read as float32)."""
        total = 0
        for spec in self.outputs.values():
            width = int(np.prod(spec["shape"])) if spec["shape"] else 1
            itemsize = 4 if spec["stored"] in FLOAT_STORED else np.dtype(spec["dtype"]).itemsize
            total += width * itemsize * len(self)
        return total

    # ── reading ──────────────────────────────────────────────────────────────

    def _read(self, positions, output=None):
        rows = self.rows[positions]
        if (rows < 0).any():
            raise KeyError(f"{int((rows < 0).sum())} of these items were never processed "
                           f"by {self.table.path.parent.name}: run useml.data.process(...) "
                           f"on them first")
        names = [output] if output else list(self.outputs)
        values = {name: self.table.read(rows, name) for name in names}
        return values[names[0]] if len(names) == 1 else values

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            out = self._read(np.asarray([index]))
            return {k: v[0] for k, v in out.items()} if isinstance(out, dict) else out[0]
        if isinstance(index, slice):
            return self._read(np.arange(len(self))[index])
        return self._read(np.asarray(index))

    def get(self, positions, output: str | None = None):
        """Rows at `positions` for one output (or all)."""
        return self._read(np.asarray(positions), output)

    def numpy(self, output: str | None = None, *, chunk: int = 65536):
        """The whole view in memory, read in chunks with a progress bar."""
        from ..store import human

        if self.nbytes() > 1 << 30:
            _progress.note(f"loading {self.name or 'view'}: {len(self)} rows, "
                           f"≈ {human(self.nbytes())} in memory")
        names = [output] if output else list(self.outputs)
        parts = {name: [] for name in names}
        for start in _progress.bar(range(0, len(self), chunk), desc=f"reading {self.name}",
                                   unit="chunk"):
            batch = self.get(np.arange(start, min(start + chunk, len(self))))
            for name in names:
                parts[name].append(batch if len(self.outputs) == 1 else batch[name])
        values = {name: np.concatenate(chunks) if chunks else
                  np.zeros((0, *self.outputs[name]["shape"])) for name, chunks in parts.items()}
        return values[names[0]] if len(names) == 1 else values

    def __array__(self, dtype=None, copy=None):
        values = self.numpy()
        if isinstance(values, dict):
            raise TypeError("several outputs: use .numpy(output)")
        return values.astype(dtype) if dtype else values

    def batches(self, batch_size: int = 1024, *, shuffle: bool = False, seed=None,
                output: str | None = None, progress: bool = True):
        """Yield (positions, rows) batches. Shuffled batches are read in sorted order
        inside each batch (sequential pages), returned in shuffled order."""
        order = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        starts = range(0, len(self), batch_size)
        iterator = _progress.bar(starts, desc=self.name, unit="batch") if progress else starts
        for start in iterator:
            positions = order[start:start + batch_size]
            yield positions, self.get(positions, output)

    def torch(self, output: str | None = None, device=None):
        import torch

        values = self.numpy(output)
        if isinstance(values, dict):
            return {name: torch.from_numpy(value).to(device) for name, value in values.items()}
        return torch.from_numpy(values).to(device)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{name}: {tuple(spec['shape'])} {spec['stored']}"
                           for name, spec in self.outputs.items())
        return (f"Processed({self.name or self.source!r}, {len(self)} rows, {shapes}, "
                f"table={self.table.path.parent.name}/{self.table.path.name})")
