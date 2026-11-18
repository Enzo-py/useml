"""Sources: what a processor reads. An indexable list of items, each with a content key.

    images("/data/photos/*.jpg")              # paths; key = hash of the file's bytes
    images(paths, key="stat")                    # key = path + size + mtime (no read)
    texts(captions)                              # strings; key = hash of the text
    items(samples, key=lambda s: s.id)           # anything, with your own key
    source("mypkg.datasets:vg_images", split="train")   # built by a function

`images`, `texts` and `source(...)` can be rebuilt in another process, so they can feed a
DETACHED job (`process(..., detach=True)`). `items` holds arbitrary Python objects and
cannot: give a factory to `source` instead.

Keys are computed once per source object and kept (`keys()`); content hashes of files are
also remembered across runs by the store (`Store.file_digest`, keyed by path, size and
modification time), so a second run does not read every image again.
"""
from __future__ import annotations

import glob
import importlib
import json
import os
from pathlib import Path

from .. import _progress
from ._keys import fingerprint, text_key


class Source:
    """An indexable collection: `len`, `source[i]` (the item), `keys()` (content keys)."""

    name: str = "source"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError

    def spec(self) -> dict | None:
        """How to rebuild this source in another process, or None if it cannot be."""
        return None

    def fingerprint(self) -> str:
        return fingerprint(self.keys())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {len(self)} items)"


class Items(Source):
    def __init__(self, values, key=None, *, name: str = "items") -> None:
        self.values = list(values)
        self.key = key
        self.name = name
        self._keys: list[str] | None = None

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def keys(self) -> list[str]:
        if self._keys is None:
            if self.key is None:
                if not all(isinstance(value, str) for value in self.values):
                    raise ValueError("items(...) of non-strings needs key=: a function "
                                     "giving each item's content key")
                self._keys = [f"item:{value}" for value in self.values]
            else:
                self._keys = [f"item:{self.key(value)}" for value in
                              _progress.bar(self.values, desc=f"keys of {self.name}",
                                            unit="item")]
        return self._keys


class Texts(Items):
    def __init__(self, values, *, name: str = "texts") -> None:
        super().__init__([str(value) for value in values], name=name)

    def keys(self) -> list[str]:
        if self._keys is None:
            self._keys = [text_key(value) for value in
                          _progress.bar(self.values, desc=f"keys of {self.name}",
                                        unit="text")]
        return self._keys

    def spec(self) -> dict:
        return {"kind": "texts", "name": self.name, "values": self.values}


class Images(Items):
    """Image paths. `key="content"` (default) hashes the bytes: the same image anywhere is
    one row. `key="stat"` uses path, size and mtime: no read, but a copy elsewhere is
    another row."""

    def __init__(self, paths, *, key: str = "content", name: str = "images") -> None:
        if key not in ("content", "stat"):
            raise ValueError("key must be 'content' or 'stat'")
        super().__init__([str(path) for path in paths], name=name)
        self.key_mode = key

    def keys(self) -> list[str]:
        if self._keys is None:
            if self.key_mode == "stat":
                out = []
                for path in _progress.bar(self.values, desc=f"stat {self.name}", unit="file"):
                    status = os.stat(path)
                    out.append(f"stat:{os.path.abspath(path)}:{status.st_size}:"
                               f"{status.st_mtime_ns}")
            else:
                from .. import store as stores

                store = stores.open()
                out = [f"file:{store.file_digest(path)}" for path in
                       _progress.bar(self.values, desc=f"hashing {self.name}", unit="file")]
                store.flush()
            self._keys = out
        return self._keys

    def spec(self) -> dict:
        return {"kind": "images", "name": self.name, "values": self.values,
                "key": self.key_mode}


class FromFactory(Source):
    """A source built by `module:function(**kwargs)`, lazily, on first use."""

    def __init__(self, target: str, kwargs: dict, *, name: str | None = None) -> None:
        if ":" not in target:
            raise ValueError(f"{target!r}: expected 'module:function'")
        self.target, self.kwargs = target, dict(kwargs)
        self.name = name or target.split(":")[1]
        self._built: Source | None = None
        json.dumps(self.kwargs)                    # must be serialisable, fail now

    @property
    def built(self) -> Source:
        if self._built is None:
            module_name, _, attribute = self.target.partition(":")
            function = importlib.import_module(module_name)
            for part in attribute.split("."):
                function = getattr(function, part)
            with _progress.step(f"building source {self.target}"):
                built = function(**self.kwargs)
            self._built = as_source(built, name=self.name)
        return self._built

    def __len__(self) -> int:
        return len(self.built)

    def __getitem__(self, index):
        return self.built[index]

    def keys(self) -> list[str]:
        return self.built.keys()

    def spec(self) -> dict:
        return {"kind": "factory", "target": self.target, "kwargs": self.kwargs,
                "name": self.name}


def images(paths, *, key: str = "content", name: str | None = None) -> Images:
    """`paths`: a glob pattern, a folder (every image under it), or a list of paths."""
    if isinstance(paths, (str, Path)):
        pattern = str(paths)
        if Path(pattern).is_dir():
            found = sorted(str(path) for path in Path(pattern).rglob("*")
                           if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp",
                                                      ".bmp", ".gif", ".tif", ".tiff"})
        else:
            found = sorted(glob.glob(pattern, recursive=True))
        if not found:
            raise FileNotFoundError(f"no image matches {pattern}")
        return Images(found, key=key, name=name or Path(pattern.split("*")[0]).name or "images")
    return Images(paths, key=key, name=name or "images")


def texts(values, *, name: str = "texts") -> Texts:
    return Texts(values, name=name)


def items(values, key=None, *, name: str = "items") -> Items:
    return Items(values, key, name=name)


def source(target: str, *, name: str | None = None, **kwargs) -> FromFactory:
    """A source built by `target` ("module:function") with `kwargs`. The function returns a
    Source, or a list of strings / paths."""
    return FromFactory(target, kwargs, name=name)


def as_source(value, *, name: str = "items") -> Source:
    if isinstance(value, Source):
        return value
    values = list(value)
    if values and all(isinstance(item, (str, Path)) and Path(item).suffix and
                      Path(item).exists() for item in values[:20]):
        return Images(values, name=name)
    if all(isinstance(item, str) for item in values):
        return Texts(values, name=name)
    raise TypeError("cannot read this as a source: wrap it with useml.data.items(values, "
                    "key=...)")


def from_spec(spec: dict) -> Source:
    kind = spec["kind"]
    if kind == "images":
        return Images(spec["values"], key=spec.get("key", "content"), name=spec["name"])
    if kind == "texts":
        return Texts(spec["values"], name=spec["name"])
    if kind == "factory":
        return FromFactory(spec["target"], spec["kwargs"], name=spec["name"])
    raise ValueError(f"unknown source kind {kind!r}")
