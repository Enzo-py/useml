"""Reading declared data files into records, with their columns checked first.

    Files("annotations.csv", columns={"id": "image_id", "text": "caption", "image": "file"},
          paths={"image": "images"})

`columns` maps each RECORD field to a COLUMN of the file: exact, one to one. Every column
named must exist — a missing one raises before any record is produced, with the columns
the file does have and the closest ones. Columns not named are dropped (`keep=` lists
extra columns kept as they are).

`types` converts fields: "int", "float", "bool", "str", "json" (a JSON text → its value),
or any function. A CSV holds only text: without it, `label` would be "3", not 3.

`paths` marks fields holding file paths, with the folder (relative to the dataset root)
they are relative to. They are stored relative and given back as absolute `Path`s.

Formats: csv, tsv, json (a list of objects), jsonl, json_dict (an object of objects: the
key becomes the `__key__` column), parquet (needs pyarrow). `path` may be a glob.
"""
from __future__ import annotations

import csv
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = ("csv", "tsv", "json", "jsonl", "json_dict", "parquet")


def _boolean(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "t"):
        return True
    if text in ("0", "false", "no", "n", "f", ""):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


_CONVERTERS = {"int": int, "float": float, "str": str, "bool": _boolean, "json": _json,
               int: int, float: float, str: str, bool: _boolean}


class SchemaError(ValueError):
    """A declared file or column that the data does not have."""


@dataclass(frozen=True)
class Files:
    path: str
    columns: dict
    format: str | None = None
    paths: dict = field(default_factory=dict)
    keep: tuple = ()
    types: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        fmt = self.format or _guess_format(self.path)
        if fmt not in FORMATS:
            raise ValueError(f"Files({self.path!r}): format {fmt!r}; use one of {FORMATS}")
        object.__setattr__(self, "format", fmt)
        for name, kind in self.types.items():
            if name not in self.columns:
                raise ValueError(f"Files({self.path!r}): types= names {name!r}, which "
                                 f"columns= does not map")
            if kind not in _CONVERTERS and not callable(kind):
                raise ValueError(f"Files({self.path!r}): type of {name!r} must be one of "
                                 f"{sorted(_CONVERTERS)} or a function")
        unknown = set(self.paths) - set(self.columns)
        if unknown:
            raise ValueError(f"Files({self.path!r}): paths= names fields {sorted(unknown)} "
                             f"that columns= does not map")

    def describe(self) -> dict:
        return {"path": self.path, "format": self.format, "columns": dict(self.columns),
                "paths": dict(self.paths), "keep": list(self.keep),
                "types": {k: v if isinstance(v, str) else getattr(v, "__name__", str(v))
                          for k, v in self.types.items()}}


def _guess_format(path: str) -> str:
    suffix = Path(path.replace("*", "x")).suffix.lower()
    return {".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl",
            ".ndjson": "jsonl", ".parquet": "parquet"}.get(suffix, "")


def _rows(path: Path, fmt: str) -> tuple[list[str], list[dict]]:
    if fmt in ("csv", "tsv"):
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t" if fmt == "tsv" else ",")
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    if fmt == "jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif fmt == "json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise SchemaError(f"{path}: expected a JSON list, got {type(rows).__name__}")
    elif fmt == "json_dict":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SchemaError(f"{path}: expected a JSON object, got {type(data).__name__}")
        rows = [{"__key__": key, **value} for key, value in data.items()]
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ImportError("parquet files need pyarrow: pip install pyarrow") from error
        rows = parquet.read_table(path).to_pylist()
    columns = sorted({key for row in rows[:1000] for key in row}) if rows else []
    return columns, rows


def read(spec: Files, root: Path) -> list[dict]:
    """The records of `spec` under `root`. Raises SchemaError before producing anything if
    a file or a column is missing."""
    matches = sorted(root.glob(spec.path)) if any(c in spec.path for c in "*?[") \
        else [root / spec.path]
    if not matches or not all(path.exists() for path in matches):
        raise SchemaError(f"declared file {spec.path!r} not found under {root}")
    records = []
    for path in matches:
        columns, rows = _rows(path, spec.format)
        wanted = [*spec.columns.values(), *spec.keep]
        missing = [column for column in wanted if column not in columns]
        if missing:
            hints = {column: difflib.get_close_matches(column, columns, n=2, cutoff=0.5)
                     for column in missing}
            detail = "; ".join(f"{column!r}" + (f" (close to {', '.join(close)})" if close
                                                else "") for column, close in hints.items())
            raise SchemaError(f"{path.relative_to(root)}: declared columns not found: "
                              f"{detail}. The file has: {', '.join(columns[:30])}")
        for number, row in enumerate(rows, start=1):
            record = {name: row[column] for name, column in spec.columns.items()}
            for name, kind in spec.types.items():
                convert = _CONVERTERS.get(kind, kind)
                if record[name] in (None, ""):
                    record[name] = None
                    continue
                try:
                    record[name] = convert(record[name])
                except (TypeError, ValueError) as error:
                    raise SchemaError(f"{path.relative_to(root)}, record {number}: "
                                      f"{name} = {record[name]!r} is not "
                                      f"{getattr(kind, '__name__', kind)} ({error})") from None
            for column in spec.keep:
                record[column] = row[column]
            for name, folder in spec.paths.items():
                if record[name] not in (None, ""):
                    record[name] = str(Path(folder) / str(record[name]))
            records.append(record)
    return records
