"""`Dataset`: a prepared dataset, a split of it, a filter of it, or one of its fields.

    cifar = datasets.load("cifar10")             # prepares if needed, reads nothing else
    train = cifar.split("train")                 # positions only
    len(train), train[0], train[10:20]           # records (dicts; path fields are Paths)
    for record in train: ...                     # streamed in file order
    cats = train.filter(lambda r: r["label"] == 3, name="cats")
    images = train.field("image")                # the values of one field

A Dataset is also a SOURCE for `useml.data.process`: its items are records (or the field's
values), its content keys follow the driver's `key`. A dataset by name and split can be
rebuilt in another process, so it can feed a detached job or an experience step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _progress
from ..data._keys import text_key
from ..data._sources import Source


class Dataset(Source):
    def __init__(self, driver, index, *, positions=None, split: str | None = None,
                 field: str | None = None, label: str | None = None) -> None:
        self.driver = driver
        self.index = index
        self.positions = positions          # None: every record
        self.split_name = split
        self.field_name = field
        self.label = label                   # a filter's name (not rebuildable)
        self._keys = None

    # ── naming ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        parts = [self.driver.name]
        if self.split_name:
            parts.append(self.split_name)
        if self.label:
            parts.append(self.label)
        if self.field_name:
            parts.append(self.field_name)
        return "/".join(parts)

    @property
    def root(self) -> Path:
        return self.driver.root

    @property
    def card(self):
        return self.driver.card

    @property
    def splits(self) -> dict:
        """{split: number of records}."""
        return dict(self.index.meta["splits"])

    @property
    def fields(self) -> dict:
        return dict(self.index.meta["fields"])

    @property
    def fingerprint(self) -> str:
        return self.index.meta["fingerprint"]

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {len(self)} records)"

    # ── selection ────────────────────────────────────────────────────────────

    def _with(self, **changes) -> "Dataset":
        current = {"positions": self.positions, "split": self.split_name,
                   "field": self.field_name, "label": self.label}
        current.update(changes)
        return Dataset(self.driver, self.index, **current)

    def split(self, name: str) -> "Dataset":
        if self.split_name or self.label:
            raise ValueError("split() applies to a whole dataset")
        return self._with(positions=self.index.split(name), split=name)

    def filter(self, predicate, *, name: str | None = None) -> "Dataset":
        """The records for which `predicate(record)` is true. Not rebuildable in another
        process (a function is not data): to feed an experience, express it as a split."""
        kept = [int(position) for position, record in
                zip(self._positions(), _progress.bar(self._records_stream(), total=len(self),
                                                     desc=f"filtering {self.name}",
                                                     unit="record"))
                if predicate(record)]
        return self._with(positions=np.asarray(kept, dtype=np.int64),
                          label=name or "filtered")

    def field(self, name: str) -> "Dataset":
        """The values of one field (a path field gives Paths): what a processor reads."""
        if name not in self.fields:
            raise KeyError(f"{self.name}: no field {name!r}; fields: {', '.join(self.fields)}")
        return self._with(field=name)

    def records(self) -> "Dataset":
        """The same selection, giving whole records again."""
        return self._with(field=None)

    # ── reading ──────────────────────────────────────────────────────────────

    def _positions(self) -> np.ndarray:
        return np.arange(len(self.index)) if self.positions is None else self.positions

    def __len__(self) -> int:
        return len(self.index) if self.positions is None else len(self.positions)

    def _resolve(self, record: dict) -> dict:
        for name in self.index.meta["path_fields"]:
            value = record.get(name)
            if value not in (None, ""):
                record[name] = self.root / value
        return record

    def _shape(self, record: dict):
        record = self._resolve(record)
        return record[self.field_name] if self.field_name else record

    def record(self, position: int) -> dict:
        """The record at INDEX position `position` (whatever the selection)."""
        return self._resolve(self.index.read(int(position)))

    def __getitem__(self, item):
        positions = self._positions()
        if isinstance(item, (int, np.integer)):
            return self._shape(self.index.read(int(positions[item])))
        if isinstance(item, slice):
            chosen = positions[item]
        else:
            chosen = positions[np.asarray(item, dtype=np.int64)]
        return [self._shape(record) for record in self.index.read_many(chosen)]

    def _records_stream(self):
        if self.positions is None:
            yield from self.index.iterate()
        else:
            # Splits and filters keep positions in increasing order: the seeks go forward
            # through the file, and nothing is buffered.
            for position in self.positions:
                yield self.index.read(int(position))

    def __iter__(self):
        for record in self._records_stream():
            yield self._shape(record)

    # ── as a source of useml.data ────────────────────────────────────────────

    def keys(self) -> list[str]:
        """Content keys, per the driver's `key`: a path field → hash of the file (the same
        image in two datasets is one row), a text field → hash of the text, a callable →
        its value, None → `<dataset>:<id>`."""
        if self._keys is not None:
            return self._keys
        key = self.driver.key if self.field_name is None else self.field_name
        path_fields = set(self.index.meta["path_fields"])
        out = []
        store = None
        for record in _progress.bar(self._records_stream(), total=len(self),
                                    desc=f"keys of {self.name}", unit="record"):
            if callable(key):
                out.append(f"item:{key(self._resolve(dict(record)))}")
            elif key is None:
                out.append(f"{self.driver.name}:{record[self.driver.id_field]}")
            elif key in path_fields:
                if store is None:
                    from .. import store as stores

                    store = stores.open()
                out.append(f"file:{store.file_digest(self.root / record[key])}")
            else:
                out.append(text_key(str(record[key])))
        if store is not None:
            store.flush()
        self._keys = out
        return out

    def spec(self) -> dict | None:
        if self.label:
            return None
        return {"kind": "dataset", "name": self.driver.name, "split": self.split_name,
                "field": self.field_name}

    # ── describing ───────────────────────────────────────────────────────────

    def summary(self, *, sample: int = 5000) -> str:
        """Counts, splits, and per field: type, how often it is empty, distinct values
        (the most frequent when few), text length — measured on up to `sample` records."""
        card = self.card
        lines = [f"{card.title if card else self.driver.name} — {self.name}: {len(self)} records"]
        if card and card.summary:
            lines.append(f"  {card.summary}")
        if self.splits:
            lines.append("  splits: " + ", ".join(f"{k} {v}" for k, v in self.splits.items()))
        positions = self._positions()
        if len(positions) > sample:
            positions = np.sort(np.random.default_rng(0).choice(positions, sample, replace=False))
        records = self.index.read_many(positions)
        names = list(self.fields)
        lines.append(f"  fields (on {len(records)} records):")
        for name in names:
            values = [record.get(name) for record in records]
            present = [value for value in values if value not in (None, "")]
            text = f"    {name:<16} {type(present[0]).__name__ if present else '—':<6}"
            if len(present) < len(values):
                text += f" empty {100 * (1 - len(present) / max(len(values), 1)):.0f}%"
            hashable = [value if not isinstance(value, (list, dict)) else str(value)
                        for value in present]
            distinct = set(hashable)
            if name in self.index.meta["path_fields"]:
                text += "  (paths)"
            elif len(distinct) <= 20 and present:
                counts = sorted(((hashable.count(v), v) for v in distinct), reverse=True)
                text += "  " + ", ".join(f"{v}: {c}" for c, v in counts[:6])
                text += " …" if len(counts) > 6 else ""
            elif present and isinstance(present[0], str):
                lengths = [len(v) for v in present]
                text += f"  {len(distinct)} distinct, {sum(lengths) / len(lengths):.0f} chars"
            else:
                text += f"  {len(distinct)} distinct"
            meaning = card.fields.get(name) if card else None
            if meaning:
                text += f"  — {meaning}"
            lines.append(text)
        return "\n".join(lines)
