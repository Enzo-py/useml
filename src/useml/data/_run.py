"""`process(source, processor, **params)` — compute what is missing, return a lazy view.

    feats = useml.data.process(useml.data.images("/data/photos"), ClipImage,
                               name="photos-clip", model="openai/clip-vit-base-patch32")

STEPS, each with feedback when it lasts:
  1. KEYS. The source's content keys (hashing images the first time: a bar).
  2. WHAT IS MISSING. Looked up in the processor's table. Nothing missing → the model is
     never loaded, the call returns at once.
  3. THE LOCK. One writer per table; a second process waits, then computes only what the
     first left.
  4. THE MODEL, loaded once ("loading …" after 2 s, then its duration).
  5. THE BATCHES. Items are loaded by the processor's `load` in worker threads, one batch
     ahead of the model. A bar counts items. After the first batch, the size to write is
     estimated and checked against the free disk and the `[data] max_size` budget,
     BEFORE anything is written.
  6. INTERRUPTED (Ctrl-C, a crash): the rows already computed are flushed. Running again
     resumes.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import _progress
from . import _catalog
from ._keys import digests as digests_of
from ._processor import Processor, resolve
from ._sources import Source, as_source
from ._table import FLOAT_STORED, Table
from ._view import Processed


class BudgetError(RuntimeError):
    """Writing this table would exceed the free disk or the `[data] max_size` budget."""


def table_for(processor: Processor, params: dict) -> Table:
    return Table(_catalog.tables_root() / processor.name / processor.fingerprint(params))


def _unique_missing(table: Table, digests: np.ndarray) -> np.ndarray:
    """Positions (first occurrence) of the digests the table lacks."""
    if not len(digests):
        return np.zeros(0, dtype=np.int64)
    absent = np.flatnonzero(table.locate(digests) < 0)
    if not len(absent):
        return absent
    as_void = np.ascontiguousarray(digests[absent]).view(np.dtype((np.void, 16))).ravel()
    _, first = np.unique(as_void, return_index=True)
    return np.sort(absent[first])


def process(source, processor, *, name: str | None = None, batch_size: int | None = None,
            detach: bool = False, dry_run: bool = False, **params):
    """Process `source` with `processor` (a decorated function or class, its name, or
    "module:qualname") and its `params`. Returns a lazy `Processed` — or, with
    `detach=True`, a `Job` running in its own process. `dry_run` only says what is
    missing."""
    processor = resolve(processor)
    source = as_source(source)
    if name is not None:
        _catalog.check_name(name)
    if detach:
        from ._jobs import submit

        return submit(source, processor, params, name=name, batch_size=batch_size)
    return run(source, processor, params, name=name, batch_size=batch_size, dry_run=dry_run)


def run(source: Source, processor: Processor, params: dict, *, name=None, batch_size=None,
        dry_run: bool = False, callback=None):
    table = table_for(processor, params)
    digests = digests_of(source.keys())
    missing = _unique_missing(table, digests)
    if dry_run:
        return {"items": len(source), "missing": int(len(missing)),
                "table": str(table.path), "exists": table.exists()}
    if len(missing):
        with table.lock():
            missing = _unique_missing(table, digests)        # another writer may have run
            if len(missing):
                _compute(table, source, processor, params, digests, missing,
                         batch_size or processor.batch_size, callback)
    elif callback is not None:
        callback(0, 0)
    if not table.exists():
        raise ValueError(f"{source.name} is empty: nothing to process")
    if table.manifest.get("code") and table.manifest["code"] != processor.code_hash():
        warnings.warn(f"processor {processor.name!r} version {processor.version!r}: its code "
                      f"changed since this table was built. If what it computes changed, "
                      f"bump `version=`; the rows read now come from the former code.",
                      stacklevel=2)
    if name is not None:
        _catalog.save(name, table_path=table.path, digests=digests, source=source.name,
                      identity=processor.identity(params), rows=len(digests))
    return Processed(table, digests, name=name, source=source.name)


def _as_outputs(value, expected: int) -> dict:
    """A processor's return value → {output: (B, ...) numpy array}."""
    if not isinstance(value, dict):
        value = {"data": value}
    out = {}
    for name, array in value.items():
        if hasattr(array, "detach"):                          # torch.Tensor
            array = array.detach().float().cpu().numpy() if array.is_floating_point() \
                else array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.ndim == 0 or len(array) != expected:
            raise ValueError(f"output {name!r}: {len(array) if array.ndim else 'a scalar'} "
                             f"rows for a batch of {expected}: return one row per item")
        out[name] = array
    return out


def _output_specs(processor: Processor, outputs: dict) -> dict:
    specs = {}
    for name, array in outputs.items():
        is_float = array.dtype.kind == "f"
        specs[name] = {"shape": list(array.shape[1:]),
                       "dtype": "float32" if is_float else str(array.dtype),
                       "stored": processor.dtype_of(name) if is_float else str(array.dtype)}
    return specs


def _row_bytes(specs: dict) -> int:
    total = 0
    for spec in specs.values():
        width = int(np.prod(spec["shape"])) if spec["shape"] else 1
        stored = spec["stored"]
        total += width * (1 if stored == "int8" else np.dtype(stored).itemsize)
        total += 4 if stored == "int8" else 0
    return total


def _check_space(table: Table, specs: dict, rows: int) -> None:
    from ..store import free_disk_bytes, human, parse_size
    from .. import paths

    needed = _row_bytes(specs) * rows
    free = free_disk_bytes(table.path)
    if needed > 1 << 30:
        _progress.note(f"{table.path.parent.name}: ≈ {human(needed)} to write for {rows} rows "
                       f"({human(_row_bytes(specs))} per row); {human(free)} free")
    if needed > 0.95 * free:
        raise BudgetError(f"≈ {human(needed)} to write but only {human(free)} free on "
                          f"{table.path}. Use dtype='int8' (½ of float16), fewer items, or "
                          f"free space (`useml data tables`).")
    limit = paths.config().get("data", {}).get("max_size")
    if limit:
        used = sum(path.stat().st_size for path in _catalog.tables_root().rglob("*")
                   if path.is_file())
        if used + needed > parse_size(limit):
            raise BudgetError(f"[data] max_size = {limit}: {human(used)} used + "
                              f"≈ {human(needed)} to write is over budget. Remove unused "
                              f"tables (`useml data tables`, `useml data rm --table`) or "
                              f"raise the budget.")


def _compute(table, source, processor, params, digests, missing, batch_size,
             callback) -> None:
    with _progress.step(f"loading processor {processor.name}"):
        instance = processor.instantiate(params)
    load = processor.loader(instance)
    batches = [missing[start:start + batch_size]
               for start in range(0, len(missing), batch_size)]
    writer = table.writer()
    loaders = ThreadPoolExecutor(max(1, processor.workers)) if load else None

    def fetch(positions):
        values = [source[int(position)] for position in positions]
        return list(loaders.map(load, values)) if loaders else values

    progress = _progress.bar(total=len(missing), desc=f"{processor.name} ← {source.name}",
                             unit="item")
    done, last_report = 0, 0.0
    try:
        with ThreadPoolExecutor(1) as prefetcher:
            pending = prefetcher.submit(fetch, batches[0])
            for index, positions in enumerate(batches):
                values = pending.result()
                if index + 1 < len(batches):
                    pending = prefetcher.submit(fetch, batches[index + 1])
                outputs = _as_outputs(instance(values), len(positions))
                if index == 0:
                    _prepare_table(table, processor, params, outputs, len(missing))
                else:
                    _check_shapes(table, outputs)
                writer.add(digests[positions], outputs)
                done += len(positions)
                progress.update(len(positions))
                if callback is not None and time.monotonic() - last_report > 1.0:
                    callback(done, len(missing))
                    last_report = time.monotonic()
        writer.flush()
    except BudgetError:
        writer._digests, writer._rows = [], {}                  # nothing was written
        raise
    except BaseException:
        writer.flush()
        _progress.note(f"interrupted: {writer.written} rows of {processor.name} kept; run "
                       f"again to resume")
        raise
    finally:
        progress.close()
        if loaders:
            loaders.shutdown(wait=False)
    if callback is not None:
        callback(done, len(missing))


def _prepare_table(table, processor, params, outputs, rows) -> None:
    specs = _output_specs(processor, outputs)
    if table.exists():
        _check_shapes(table, outputs)
    else:
        from .. import __version__

        table.write_manifest({"identity": processor.identity(params), "outputs": specs,
                              "code": processor.code_hash(), "created": time.time(),
                              "useml": __version__})
    _check_space(table, table.outputs, rows)


def _check_shapes(table, outputs) -> None:
    for name, array in outputs.items():
        spec = table.outputs.get(name)
        if spec is None:
            raise ValueError(f"output {name!r} is not in this table ({list(table.outputs)})")
        if list(array.shape[1:]) != spec["shape"]:
            raise ValueError(f"output {name!r}: rows of shape {tuple(array.shape[1:])}, the "
                             f"table holds {tuple(spec['shape'])}. Fixed-size rows only: pad, "
                             f"pool, or change a parameter (another table).")
        if (array.dtype.kind == "f") != (spec["stored"] in FLOAT_STORED):
            raise ValueError(f"output {name!r}: dtype {array.dtype} does not match the "
                             f"table's {spec['stored']}")
