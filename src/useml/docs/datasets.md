# Datasets: `useml.datasets`

Many datasets, each **declared once** by a driver, loaded by name.

## The idea in four terms

- A **driver** is a class that says where a dataset comes from, which files to read and
  how their columns map to fields. Example: "download `annotations.zip`, read
  `annotations.csv`, column `caption` is the field `text`".
- A **record** is one item of the dataset, as a dict: `{"id": "s12", "text": "a red
  cube", "image": Path(...), "label": 2}`.
- The **index** is the list of records, written once on disk; reading record 12 reads only
  that line.
- A **split** is a named part of the dataset (`train`, `val`, `test`).

## Load a dataset

```python
from useml import datasets as ds

captions = ds.load("captions")                 # prepares on first use
train = ds.load("captions", split="train")
len(train), train[0], train[10:20], train[[3, 1, 4]]
for record in train:                           # streamed
    ...
captions.splits                                # {"train": 8000, "val": 1000, "test": 1000}
print(captions.summary())                      # counts, splits, fields and their meaning
```

Path fields come back as absolute `Path`s. Nothing is read until asked: opening a dataset
of millions of records reads one small file.

## Write a driver

```python
# mypkg/drivers.py
from useml import datasets as ds

@ds.register
class Captions(ds.Driver):
    name = "captions"
    version = 1
    card = ds.Card(
        title="Captions",
        summary="photos with one caption each",
        fields={"text": "the caption", "image": "the photo", "label": "category 0-9"},
        url="https://example.org/captions", license="CC BY 4.0")
    sources = [ds.Source("https://example.org/captions.zip", extract=True)]
    files = [ds.Files("annotations.csv",
                      columns={"id": "image_id", "text": "caption", "image": "file",
                               "label": "category"},
                      paths={"image": "images"},
                      types={"label": int})]
    hash_splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    key = "image"
```

Then tell useml where drivers live:

```toml
# pyproject.toml (or [datasets] in useml.toml)
[tool.useml.datasets]
packages = ["mypkg.drivers"]      # imported on the first datasets.load
pythonpath = ["."]                # if the package is not installed
root = "data"                     # optional: datasets under the project (default: the machine's root)
```

### Sources

| Declaration | What happens |
|---|---|
| `ds.Source(uri, dest=".", extract=False, keep_archive=False)` | fetched with `useml.hub.get` (`hf://`, `https://`, `gdrive://`): sequential, resumed if interrupted; archives are unpacked into `dest`, then deleted unless `keep_archive=True` |
| `ds.Manual(path, hint="...")` | a file you place by hand under the dataset root; if missing, loading stops and prints the hint |

A source is fetched once. It is fetched again only if its declaration changes.

### Files and columns

`ds.Files(path, columns, format=None, paths={}, types={}, keep=())`:

- `columns` maps each **record field** to a **column** of the file. Every column named
  must exist: a missing one stops the preparation with the columns the file has and the
  closest ones (`'captoin' (close to caption)`).
- `paths` marks fields holding file paths, with their folder.
- `types` converts fields: `"int"`, `"float"`, `"bool"`, `"str"`, `"json"`, or a function. A
  CSV holds only text: without it, `label` would be `"3"`, not `3`.
- `keep` keeps extra columns as they are.
- Formats: `csv`, `tsv`, `json` (a list), `jsonl`, `json_dict` (an object of objects; the key
  becomes the `__key__` column), `parquet` (needs pyarrow). `path` may be a glob.

### Your own format

Override `process()` and yield record dicts. Declare which fields hold paths relative to
the dataset root in `path_fields`:

```python
@ds.register
class Scenes(ds.Driver):
    name = "scenes"
    sources = [ds.Source("hf://datasets/<org>/<name>")]
    path_fields = ("image",)

    def process(self):
        for item in read_my_format(self.root / "scenes.json"):
            yield {"id": item["uid"], "image": f"images/{item['file']}",
                   "objects": item["objects"]}

    def check(self, dataset):          # optional: [] when fine, else the problems
        return [] if len(dataset) > 1000 else ["fewer records than expected"]
```

### Splits

Native splits first, else a stable hash:

| Declaration | Splits |
|---|---|
| `split_field = "split"` | the value of that field of each record |
| `splits = {"train": lambda r: ..., "test": lambda r: ...}` | your rules |
| `hash_splits = {"train": 0.8, "test": 0.2}` | a hash of (dataset name, record id) |

The hash split is reproducible on any machine, does not depend on the order, and stays
stable when the dataset grows: a record never changes side because others arrived. The
salt is the dataset's name, so two datasets with similar ids do not share a partition.
Prefer native splits when they exist: they respect groupings (same scene, same author)
that a hash would break.

### Content keys

`key` says what identifies an item when it is processed with `useml.data`:

| `key` | Content key |
|---|---|
| a path field (`"image"`) | the hash of the file: the same image in two datasets is processed once |
| a text field | the hash of the text |
| a function of the record | its value |
| `None` (default) | `<dataset>:<id>` |

### Dependencies

`depends = ("other",)` prepares `other` first, for a dataset that reuses its files.

## Preparation, and when it runs again

`datasets.load(name)` prepares the dataset when needed:

1. its dependencies;
2. each source not fetched yet;
3. the index, by reading the declared files (or `process()`);
4. the ids must be unique; the splits are computed;
5. `check()`: by default, at least one record, and the path fields of the first records
   exist.

**It runs again by itself** when what produced the index changes: `version`, `files`, the
split settings, `key`, the code of `process()`, or the declared files on disk (their size
or date). Downloads are not repeated. Rewording the card rebuilds nothing.
`datasets.prepare(name, force=True)` rebuilds everything.

Datasets live in `$USEML_DATASETS_ROOT`, else `[tool.useml.datasets] root` of the project,
else `[datasets] root` of the machine config, else `<home>/datasets`. Each dataset has its
folder: `<root>/<name>/`.

## With processed data and experiences

```python
from useml import data, datasets as ds, exp

train = ds.load("captions", "train")
feats = data.process(train.field("image"), ClipImage, name="captions-train-clip")

with exp.experience("captions/baseline") as e:
    e.step(train_model, ds.load("captions", "train"), name="model")
```

- `dataset.field("image")` gives the values of one field: what a processor reads.
- A dataset by name and split can be rebuilt in another process: it can feed a detached
  job or an experience step. A **filtered** dataset (`.filter(fn)`) cannot (a function is
  not data): declare the subset as a split, or filter inside the step.
- A bundle does not copy raw datasets (they can weigh terabytes); its manifest lists
  them, and their drivers are in the bundled code.

## Commands

```bash
useml datasets ls                         # every driver: ready, outdated, never prepared
useml datasets info captions --split test  # card, splits, fields
useml datasets prepare captions --force   # rebuild
useml datasets check captions
```
