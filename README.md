# useml

A toolkit for machine-learning projects on big machines: one root for every cache, one
gate for every download, datasets processed once and loaded by name, YAML configs, and
experiments that run in the background, survive a closed laptop, and say why they crashed.

## Quick start

```bash
pip install "useml[all]"
cd my-project
useml init            # shows what it would set up (adds, never overwrites); confirm to write
useml check           # checks that everything resolves
useml docs            # the full documentation, installed with the package
```

## Documentation

The guides are in [`src/useml/docs/`](src/useml/docs/index.md) and installed with the
package (`useml docs <topic>`):

- [Quickstart](src/useml/docs/quickstart.md) · [Adopt useml in a project](src/useml/docs/adopt.md)
- [Caches and downloads](src/useml/docs/paths-and-hub.md) · [Cache store](src/useml/docs/store.md)
  · [Datasets](src/useml/docs/datasets.md) · [Processed data](src/useml/docs/data.md) · [Configs](src/useml/docs/config.md)
  · [Experiences](src/useml/docs/exp.md)
- [Monitor: a private dashboard reachable from anywhere](src/useml/docs/monitor.md)
- [Troubleshooting](src/useml/docs/troubleshooting.md) · [Reference](src/useml/docs/reference.md)
  · [FAQ](src/useml/docs/faq.md)
- [Instructions for AI coding assistants](src/useml/docs/SKILL.md)

The sections below are a tour.

## One root per machine

```toml
# ~/.config/useml/config.toml
root = "/data/me/useml"

[hub]
parallel = false        # default: one download thread at a time
```

`import useml` (first, before torch / transformers) points `HF_HOME`, `TORCH_HOME`,
`TMPDIR`, pip, triton… under the root. A cache already placed outside the home is never
moved. `useml paths` shows where everything goes.

## One gate for downloads

```python
import useml

useml.hub.get("hf://openai/clip-vit-large-patch14")
useml.hub.get("hf://datasets/org/name@main/file.json")
useml.hub.get("https://example.org/annotations.zip", extract_archive=True)
useml.hub.get("gdrive://<file id>")
```

Already downloaded → answered from disk, no network. Every download is recorded:

```bash
useml hub ls            # what is on disk, how big, last used
useml hub scan          # caches left in the home (~/.cache/huggingface…)
useml hub scan --move --apply
```

## The network policy

By default, only one thread may download at a time, in every library:

- Hugging Face `xet` (Rust, its own parallel connections) is disabled;
- the thread pools of `huggingface_hub` and `datasets` become sequential;
- `socket.connect` refuses an outbound connection from a second thread
  (`ParallelDownloadError`), and logs it;
- `useml hub watch -- python train.py` watches the whole process tree from outside
  and kills it on a sustained violation.

`useml.hub.allow_parallel()` lifts it, explicitly.

## Datasets declared once

```python
from useml import datasets as ds

@ds.register
class Captions(ds.Driver):
    name = "captions"
    sources = [ds.Source("https://example.org/captions.zip", extract=True)]
    files = [ds.Files("annotations.csv", columns={"id": "image_id", "text": "caption",
                                                   "image": "file"}, paths={"image": "images"})]
    hash_splits = {"train": 0.8, "test": 0.2}
    key = "image"

train = ds.load("captions", "train")          # downloads, indexes and checks on first use
train[0]                                       # {"id": ..., "text": ..., "image": Path(...)}
```

Missing columns fail at preparation with the closest names; the index rebuilds itself when
the driver or its files change; splits are native or a stable salted hash.

## One cache store

```python
store = useml.store.open()                       # <root>/store
masks = store.declare("processing/masks", cost="expensive")
masks.put({"image": digest, "model": "sam-h"}, {"masks": array})
masks.get({"image": digest, "model": "sam-h"})
```

One SQLite index (WAL, safe across processes), large values in files written atomically,
one write per key. Spaces are declared with a cost (`cheap`, `expensive`, `precious`)
kept in the index, so cleaning needs no project code:

```bash
useml store spaces
useml store gc --max-size 450GB            # dry run: cheapest to recompute first
useml store gc --max-size 450GB --apply
```

`[store] max_size = "450GB"` in the config sets the default budget. Nothing is ever
evicted automatically, pinned entries and `precious` spaces are kept.

## Data processed once

```python
from useml import data

@data.processor("clip-image", version=1, batch_size=64, dtype="float16")
class ClipImage:
    def __init__(self, model="openai/clip-vit-base-patch32", device="cuda"): ...
    def load(self, path): ...             # per item, in worker threads
    def __call__(self, images): ...       # a batch → (B, D) array, or a dict of arrays

data.process(data.images("/data/photos"), ClipImage, name="photos-clip")
feats = data.load("photos-clip")            # one line; nothing is read until used
feats[0], feats[100:200], feats.numpy()
```

- One table per (processor, version, parameters, dtype), shared by every dataset: rows
  are found by a hash of their CONTENT, so an image two datasets share is computed once.
- Only missing rows are computed; when nothing is missing, the model is never loaded.
- Storage `float16` (default), `float32` or `int8` (one byte per number + a scale per row).
- Memory-mapped, append-only shards; one writer per table (a file lock); an interrupted
  run keeps its rows and resumes.
- The size to write is checked against the free disk and `[data] max_size` before writing.
- `process(..., detach=True)` runs in its own process: `job.follow()`, `useml data jobs`.
- Anything longer than ~2 s shows a tqdm bar or a "⏳ … / ✓ … (12.3s)" line.

```bash
useml data ls          # named datasets
useml data tables      # tables, size, who uses them
useml data rm photos-clip --table
```

## Configs in YAML, objects in one line

```toml
# pyproject.toml of your project
[tool.useml.config]
root = "configs"                          # configs/models/*.yaml, configs/scorers/*.yaml
packages = ["mypkg.models"]               # where `type:` names are looked up (no imports by string)
```

```yaml
# configs/models/resnet_ft.yaml
meta: {display_name: ResNet-50 (fine-tuned)}
type: Classifier
backbone: {ref: resnet50}                      # another config, built once and shared
optimizer: {type: SGD, lr: 0.01}
```

```python
from useml import config
model = config.load("resnet_ft")                                  # memoised by name
model = config.load("resnet_ft", optimizer={"lr": 0.1}) # one setting, merged
model.config.hash, model.config.provenance()                    # cite what ran
config.create("variant", model); config.diff("resnet_ft", model)   # and back to a file
```

Misspelled keys and types raise with the closest name and the place in the file.

## Experiences, and their outputs in one line

```python
from useml import exp, config, data

with exp.experience("vision/baseline") as e:          # nothing runs in the block
    trained = e.step(train, data.load("photos-clip"), config.load("classifier"),
                     epochs=10, name="model")
    e.step(report, trained, name="report")
# block end: declaration written, run detached, followed with a progress bar

model = exp.get("vision/baseline/model")             # later, anywhere: the trained model
exp.open("vision/baseline").results()                 # every step's numbers
```

- A step is an importable function. Arguments are recipes: outputs of earlier steps,
  `config.load(...)` objects, `data.load(name)` datasets, outputs of other experiences.
- Fingerprints chain: running the block again skips what is done; changing a line reruns
  only what depends on it. An interrupted run resumes.
- Outputs are saved by type: a model built from a config is kept as config + weights and
  comes back rebuilt; arrays as `.npz`; figures as `.png`; numbers as JSON.
- A step with a `ctx` parameter gets its folder, `ctx.progress(i, n)`, `ctx.checkpoints`
  and `ctx.should_stop()`.

```bash
useml exp ls
useml exp show vision/baseline
useml exp follow vision/baseline
useml exp stop vision/baseline
```

## A private dashboard, reachable from anywhere

```bash
useml monitor snapshot        # exactly what would leave this machine
useml monitor deploy          # a PRIVATE Hugging Face Space (once)
useml monitor start           # one agent per machine: outgoing connections only
```

Machines open no port: each agent keeps an outgoing HTTPS connection to the Space, which
can only ask it for a fresh snapshot (about a second after a click). `[monitor] share`
chooses what is sent; paths, hostname and user name are masked; code never leaves. The
setup view shows useml's state on each machine and writes the `useml settings set …`
command to run there to change a setting.

### Runs that survive, crashes that explain themselves

A detached run lives in a tmux session (`useml exp attach <name>`), or in its own session
without tmux: closing the laptop, the SSH connection or the notebook does not stop it.
A small supervisor process is its parent and records how it ended — exit code, signal
(SIGKILL ≈ out of RAM, SIGSEGV = native crash), peak RAM, kernel OOM lines, traceback,
last log lines:

```bash
useml exp why vision/baseline     # where it is, or why it died
useml exp stack vision/baseline   # the Python stack of a run that seems stuck
```

```toml
# ~/.config/useml/config.toml — optional: be told when a run ends or crashes
[exp]
notify = "your-command $USEML_EXP_NAME $USEML_EXP_STATUS"
```

### Bundles: archive the version that worked

```bash
useml exp bundle vision/baseline -o archive/baseline-final [--archive]
```

A standalone folder: the code exactly as it ran (commit + uncommitted diff + untracked
sources, recorded at every launch), a copy of useml, the configs, every step's output,
the rows of the datasets it read, `packages.txt`, and sha256 checksums. `python load.py`
/ `from load import get` reads an output; `python run.py --from-scratch` recomputes it.
