"""useml.data — process datasets ONCE, outside experiences, and load the result in one line.

    import useml
    from useml import data

    @data.processor("clip-image", version=1, batch_size=64, dtype="float16")
    class ClipImage:                       # in a module (needed only to detach)
        def __init__(self, model="openai/clip-vit-base-patch32", device="cuda"): ...
        def load(self, path): ...            # optional, runs in worker threads
        def __call__(self, images): ...      # batch → (B, D)

    data.process(data.images("/data/photos"), ClipImage, name="photos-clip")
    data.process(data.images("/data/more_photos"), ClipImage, name="more-clip")
    #  → the images both folders share are not computed again

    feats = data.load("photos-clip")           # in an experience: one line, nothing read yet
    feats[0], feats[100:200], feats.numpy()

THE PIECES:
  sources      `images`, `texts`, `items`, `source` — what to read, with content keys
  processors   `@processor` — a batch of items → one fixed-size row per item
  tables       one per (processor, version, parameters, dtype), shared by all datasets
  names        `process(..., name=)` → `load(name)`; `ls()` lists them
  jobs         `process(..., detach=True)` → runs in its own process; `jobs()`, `follow()`

See `_run` (what `process` does), `_table` (storage), `_view` (reading).
"""
from ._catalog import data_root, load, ls, record, remove as rm, tables
from ._jobs import Job, job, jobs
from ._processor import Processor, processor
from ._run import BudgetError, process, table_for
from ._sources import Source, images, items, source, texts
from ._view import Processed

__all__ = [
    "processor", "Processor", "images", "texts", "items", "source", "Source",
    "process", "load", "ls", "rm", "tables", "record", "data_root", "table_for",
    "Processed", "Job", "job", "jobs", "BudgetError",
]
