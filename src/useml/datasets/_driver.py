"""Drivers: how a dataset is fetched, read, checked and split — DECLARED, not hidden.

    from useml import datasets as ds

    @ds.register
    class Captions(ds.Driver):
        name = "captions"
        version = 1
        card = ds.Card(title="Captions", summary="images with one caption each",
                       fields={"image": "the photo", "text": "its caption"})
        sources = [ds.Source("https://example.org/captions.zip", extract=True)]
        files = [ds.Files("annotations.csv",
                          columns={"id": "image_id", "text": "caption", "image": "file"},
                          paths={"image": "images"})]
        split_field = None                     # no native split here…
        hash_splits = {"train": 0.8, "val": 0.1, "test": 0.1}   # …so a stable one
        key = "image"                          # content key: the image file

NOTHING IS IMPLICIT. Sources are listed in order; every file read is declared with its
columns; a wrong path or column fails at preparation, with the columns the file has —
never silently in the middle of an iteration.

PREPARATION (`prepare()`, run by `datasets.load`): dependencies first, then each source
not yet fetched, then the index (`process()` → records), then `check()`. It is resumable
and SELF-INVALIDATING: what was done is recorded with a fingerprint of what produced it.

  • a source is fetched again only if ITS declaration changed;
  • the index is rebuilt when `version`, `files`, the split settings, the key or the
    CODE of `process()` change — no sentinel file to delete by hand.

A driver with its own format overrides `process()` and yields record dicts; it declares
in `path_fields` which fields hold paths relative to the dataset root.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import _progress
from ._files import Files, SchemaError, read

_REGISTRY: dict = {}


class DatasetError(RuntimeError):
    pass


class ManualDownloadRequired(DatasetError):
    """A source the user must fetch by hand (licence, login, rate limit)."""


@dataclass(frozen=True)
class Card:
    """The human description of a dataset: what it is, what each field means."""
    title: str
    summary: str = ""
    fields: dict = field(default_factory=dict)
    url: str = ""
    license: str = ""
    citation: str = ""


@dataclass(frozen=True)
class Source:
    """A download through `useml.hub` (sequential, resumable, recorded)."""
    uri: str
    dest: str = "."
    extract: bool = False
    keep_archive: bool = False
    note: str = ""

    def describe(self) -> dict:
        return {"uri": self.uri, "dest": self.dest, "extract": self.extract}


@dataclass(frozen=True)
class Manual:
    """Files the user places by hand under the dataset root."""
    path: str
    hint: str = ""

    def describe(self) -> dict:
        return {"manual": self.path}


def _fingerprint(value) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def _code(function) -> str:
    try:
        return hashlib.blake2b(inspect.getsource(function).encode(), digest_size=8).hexdigest()
    except (OSError, TypeError):
        return ""


class Driver:
    name: str = ""
    version = 1
    card: Card | None = None
    sources: list = []
    files: list = []
    depends: tuple = ()
    path_fields: tuple = ()
    id_field: str = "id"
    key = None                 # a field name (path → file hash, text → text hash), a callable, or None
    split_field: str | None = None
    splits: dict = {}          # {name: callable(record) -> bool}
    hash_splits: dict = {}     # {name: ratio}, used when no native split is declared

    def __init__(self, root) -> None:
        self.root = Path(root)

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def _state_path(self) -> Path:
        return self.root / ".useml" / "state.json"

    def _state(self) -> dict:
        try:
            return json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self, **fields) -> None:
        state = self._state()
        state.update(fields)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=1))
        os.replace(temporary, self._state_path)

    def all_path_fields(self) -> list:
        fields = list(self.path_fields)
        for spec in self.files:
            fields += [name for name in spec.paths if name not in fields]
        return fields

    def files_signature(self) -> list:
        """(path, size, modification time) of every declared file: a file edited or
        replaced on disk rebuilds the index, without reading it to find out."""
        out = []
        for spec in self.files:
            pattern = spec.path
            matches = sorted(self.root.glob(pattern)) if any(c in pattern for c in "*?[") \
                else [self.root / pattern]
            for path in matches:
                try:
                    status = path.stat()
                    out.append([str(path.relative_to(self.root)), status.st_size,
                                status.st_mtime_ns])
                except OSError:
                    out.append([str(path), None, None])
        return out

    def index_fingerprint(self) -> str:
        """What the index depends on: the declaration, the code of `process()`, and the
        size and date of the declared files. The card is not in it: rewording a
        description does not rebuild anything. A driver with its own `process()` reading
        undeclared files is rebuilt with `prepare(force=True)` when they change."""
        process = type(self).process
        return _fingerprint({
            "files_on_disk": self.files_signature(),
            "version": self.version, "files": [spec.describe() for spec in self.files],
            "id": self.id_field, "key": self.key if not callable(self.key) else _code(self.key),
            "split_field": self.split_field,
            "splits": {name: _code(rule) for name, rule in self.splits.items()},
            "hash_splits": self.hash_splits, "paths": self.all_path_fields(),
            "process": _code(process) if process is not Driver.process else "declared"})

    # ── the steps ────────────────────────────────────────────────────────────

    def download(self) -> None:
        """Fetch every source not fetched yet (or whose declaration changed)."""
        from .. import hub

        done = dict(self._state().get("sources", {}))
        for source in self.sources:
            mark = _fingerprint(source.describe())
            if isinstance(source, Manual):
                if not (self.root / source.path).exists():
                    raise ManualDownloadRequired(
                        f"dataset {self.name!r}: {source.path!r} must be placed by hand under "
                        f"{self.root}." + (f"\n{source.hint}" if source.hint else "") +
                        f"\nThen run: useml datasets prepare {self.name}")
                continue
            if done.get(source.uri) == mark:
                continue
            destination = self.root / source.dest
            with _progress.step(f"{self.name}: fetching {source.uri}"):
                if source.extract:
                    (self.root / ".useml" / "downloads").mkdir(parents=True, exist_ok=True)
                    archive = hub.get(source.uri, dest=self.root / ".useml" / "downloads")
                    hub.extract(archive, destination)
                    if not source.keep_archive:
                        Path(archive).unlink(missing_ok=True)
                        hub.registry().forget(
                            f"{source.uri} -> {(self.root / '.useml' / 'downloads').resolve()}")
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    hub.get(source.uri, dest=destination)
            done[source.uri] = mark
            self._save_state(sources=done)

    def process(self):
        """The records: by default, every declared file read with its columns."""
        if not self.files:
            raise DatasetError(f"dataset {self.name!r} declares no `files` and does not "
                               f"override process()")
        for spec in self.files:
            yield from read(spec, self.root)

    def check(self, dataset) -> list[str]:
        """Problems found in the prepared dataset ([] when fine). The default: not empty,
        and the path fields of the first records exist."""
        problems = []
        if len(dataset) == 0:
            problems.append("no record")
        for position in range(min(50, len(dataset))):
            record = dataset.record(position)
            for name in self.all_path_fields():
                value = record.get(name)
                if value and not Path(value).exists():
                    problems.append(f"record {position}: {name} = {value} does not exist")
                    break
            if len(problems) >= 5:
                break
        return problems

    # ── splits ───────────────────────────────────────────────────────────────

    def compute_splits(self, records: list) -> dict:
        """{split: positions}. Native first (`split_field`, then `splits` rules), else a
        stable hash of (dataset name, record id) — reproducible on any machine,
        independent of the order, stable when the dataset grows, and different from one
        dataset to another (the salt is the name)."""
        if self.split_field:
            out: dict = {}
            for position, record in enumerate(records):
                value = record.get(self.split_field)
                if value not in (None, ""):
                    out.setdefault(str(value), []).append(position)
            return out
        if self.splits:
            return {name: [p for p, r in enumerate(records) if rule(r)]
                    for name, rule in self.splits.items()}
        if self.hash_splits:
            total = sum(self.hash_splits.values())
            bounds, cumulated = [], 0.0
            for name, ratio in self.hash_splits.items():
                cumulated += ratio / total
                bounds.append((name, cumulated))
            out = {name: [] for name in self.hash_splits}
            for position, record in enumerate(records):
                digest = hashlib.blake2b(f"{self.name}:{record[self.id_field]}".encode(),
                                         digest_size=8).digest()
                value = int.from_bytes(digest, "big") / 2 ** 64
                name = next((n for n, bound in bounds if value < bound), bounds[-1][0])
                out[name].append(position)
            return out
        return {}

    # ── the preparation ─────────────────────────────────────────────────────────

    def prepare(self, *, force: bool = False) -> dict:
        """Dependencies, downloads, index, check. Returns the index metadata."""
        from . import _index
        from ._dataset import Dataset

        for dependency in self.depends:
            prepare(dependency)
        self.root.mkdir(parents=True, exist_ok=True)
        if force:
            self._save_state(sources={})
        self.download()
        index = _index.Index(self.root)
        fingerprint = self.index_fingerprint()
        if force or not index.exists() or index.meta.get("fingerprint") != fingerprint:
            reason = ("forced" if force else "first preparation" if not index.exists()
                      else "the driver changed")
            _progress.note(f"{self.name}: building the index ({reason})")
            with _progress.step(f"{self.name}: reading the records"):
                records = list(self.process())
            seen = set()
            for position, record in enumerate(records):
                if not isinstance(record, dict):
                    raise DatasetError(f"{self.name}: process() must yield dicts, got "
                                       f"{type(record).__name__}")
                if self.id_field not in record:
                    record[self.id_field] = str(position)
                identifier = record[self.id_field]
                if identifier in seen:
                    raise DatasetError(f"{self.name}: two records with {self.id_field} = "
                                       f"{identifier!r}; ids must be unique")
                seen.add(identifier)
            splits = self.compute_splits(records)
            _index.write(self.root, records, splits=splits,
                         path_fields=self.all_path_fields(), fingerprint=fingerprint,
                         expected=len(records))
            index = _index.Index(self.root)
        problems = self.check(Dataset(self, index))
        if problems:
            self._save_state(checked=False, problems=problems)
            raise DatasetError(f"dataset {self.name!r} failed its check: " + "; ".join(problems))
        self._save_state(checked=True, problems=[], prepared=time.time())
        return index.meta


# ── registry and roots ────────────────────────────────────────────────────────

def register(cls):
    """`@datasets.register` on a Driver subclass: `datasets.load(cls.name)` finds it."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__}: set `name` before registering")
    for source in cls.sources:
        if not isinstance(source, (Source, Manual)):
            raise TypeError(f"{cls.__name__}.sources: Source(...) or Manual(...), got {source!r}")
    for spec in cls.files:
        if not isinstance(spec, Files):
            raise TypeError(f"{cls.__name__}.files: Files(...), got {spec!r}")
    _REGISTRY[cls.name] = cls
    return cls


def _project_settings() -> tuple:
    from .._project import section

    return section("datasets")


def discover() -> dict:
    """Import the project's driver packages (`[tool.useml.datasets] packages`), once."""
    folder, settings = _project_settings()
    if settings:
        for entry in settings.get("pythonpath", []):
            path = str((folder / entry).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        for package in settings.get("packages", []):
            importlib.import_module(package)
    return _REGISTRY


def root() -> Path:
    """`$USEML_DATASETS_ROOT`, else the project's `[datasets] root`, else the machine's
    `[datasets] root`, else `<home>/datasets`."""
    from .. import paths

    explicit = os.environ.get("USEML_DATASETS_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    folder, settings = _project_settings()
    if settings and settings.get("root"):
        return (folder / settings["root"]).resolve()
    configured = paths.config().get("datasets", {}).get("root")
    if configured:
        return Path(os.path.expandvars(str(configured))).expanduser()
    return paths.home() / "datasets"


def driver(name: str) -> Driver:
    if name not in _REGISTRY:
        discover()
    if name not in _REGISTRY:
        import difflib

        close = difflib.get_close_matches(name, _REGISTRY, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"no dataset driver {name!r}.{hint} Registered: "
                       f"{', '.join(sorted(_REGISTRY)) or 'none'} (drivers are found in "
                       f"[tool.useml.datasets] packages)")
    return _REGISTRY[name](root() / name)


def prepare(name: str, *, force: bool = False) -> dict:
    return driver(name).prepare(force=force)
