# Processed data: `useml.data`

Process a dataset **once**, outside experiments, and load the result by name.

## The idea in three terms

- A **source** is the list of items to process, for example the image files of a folder.
- A **processor** turns a batch of items into one fixed-size row per item, for example
  "embed each image with CLIP" (one image → 512 numbers).
- A **table** holds every row a processor computed with given settings, **whatever the
  dataset**. Rows are found by a hash of the item's content.

Because tables belong to the processor, two datasets that share images share their rows.

## Write a processor

```python
# my_project/processors.py
import numpy as np
from useml import data

@data.processor("clip-image", version=1, batch_size=64, dtype="float16")
class ClipImage:
    def __init__(self, model="openai/clip-vit-base-patch32", device="cuda"):
        ...                                # load the model: runs only if rows are missing

    def load(self, path):                  # optional: per item, in worker threads
        from PIL import Image
        return Image.open(path).convert("RGB")

    def __call__(self, images):            # a batch → (B, D) array, or a dict of arrays
        ...
        return embeddings

@data.processor("caption-length", dtype="float32")
def caption_length(texts, *, unit="char"):
    return np.array([[len(t)] for t in texts], dtype=np.float32)
```

- **Parameters** are the constructor arguments (class) or keyword arguments (function).
  `device` and `batch_size` do not change the table (`ignore=` sets this list).
- **Bump `version`** when the code changes what it computes. The code is also hashed: a
  table built by different code under the same version gives a warning.
- **Rows must have a fixed shape.** A different shape raises.

## Process, then load

```python
from useml import data
from my_project.processors import ClipImage

feats = data.process(data.images("/data/photos"), ClipImage, name="photos-clip")
feats = data.load("photos-clip")         # anywhere later: one line, nothing read yet
```

Sources:

| Source | Content key |
|---|---|
| `data.images("/data/photos")` (folder, glob or list of paths) | hash of the file bytes (`key="stat"`: path + size + mtime, no read) |
| `data.texts(captions)` | hash of the text |
| `data.items(objects, key=lambda o: o.id)` | your key |
| `data.source("my_project.datasets:build", split="train")` | whatever the function returns |

What `process` does:

1. computes the content keys (a progress bar while hashing images the first time);
2. looks up which rows the table lacks — **if none, the model is never loaded**;
3. takes the table's lock: a second process waits, then computes only what is left;
4. loads the model ("⏳ loading…" after 2 s), runs the batches with a progress bar, loads
   the next batch in worker threads meanwhile;
5. after the first batch, checks the size to write against the free disk and
   `[data] max_size`, **before writing anything**;
6. if interrupted, keeps the rows already computed; running again resumes.

`data.process(..., dry_run=True)` only says how many items are missing.

## Read a processed dataset

```python
feats = data.load("photos-clip")
len(feats)                 # rows, in the dataset's order
feats[0]                   # one row
feats[100:200]             # a batch (several outputs → a dict of arrays)
feats[[5, 3, 9]]
feats.numpy()              # everything in memory (announced first if over 1 GB)
feats.torch(device="cuda")
for positions, rows in feats.batches(4096, shuffle=True, seed=0):
    ...
feats.missing, feats.complete, feats.nbytes(), feats.outputs
```

Rows are memory-mapped: reading a row reads only its disk pages.

## Storage formats

| `dtype=` | Bytes per number | Read back as |
|---|---|---|
| `"float16"` (default) | 2 | float32 |
| `"float32"` | 4 | float32 |
| `"int8"` | 1, plus 4 bytes per row for the scale | float32 (error ≤ 0.4 % of the row's largest value) |

Integer and boolean outputs keep their own dtype. A dict gives one dtype per output:
`dtype={"embedding": "int8"}`.

## Detached processing

```python
job = data.process(data.images("/data/photos"), "my_project.processors:ClipImage",
                   name="photos-clip", detach=True)
job.follow()          # a progress bar; Ctrl-C stops following, not the job
feats = job.result()
```

The processor must be importable (in a module) and the source rebuildable (`images`,
`texts` or `source(...)`). Submitting a job that is already running attaches to it.

```bash
useml data ls                    # named datasets, rows, table size
useml data tables                # every table and who uses it
useml data info photos-clip
useml data jobs
useml data follow <job>
useml data rm photos-clip --table    # also delete the table if no other name uses it
```
