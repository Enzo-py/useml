# useml

**Stop coding plumbing, start training models.**

`useml` is a minimalist ML experiment versioning framework for PyTorch. It handles checkpointing, source archiving, and model isolation so you can focus on the architecture.

## Installation

```bash
pip install useml
```

## Core concepts

```
vault/
└── my-experiment/          ← project
    ├── snap_20240501_.../  ← snapshot (one per commit)
    │   ├── weights/        ← model .pth files
    │   ├── configs/        ← per-component YAML configs
    │   ├── source/         ← archived source code
    │   ├── manifest.yaml   ← component registry
    │   └── metadata.yaml   ← message + metrics
    └── snap_20240502_.../
```

## Basic usage

### Session setup

```python
import useml

useml.init("vault")          # connect to a vault directory
useml.new("my-experiment")   # create and focus a project
# or: useml.focus("my-experiment") to resume an existing one
```

### Track → Commit loop

```python
import torch.nn as nn
import useml

useml.init("vault")
useml.new("mnist-run", auto_focus=True)

model = nn.Linear(784, 10)
useml.track("classifier", model)

# ... train ...

snap = useml.commit("baseline", val_loss=0.42, accuracy=0.91)
```

### Multiple models, per-model configs

```python
useml.track("encoder", encoder, config=encoder_config)
useml.track("decoder", decoder, config=decoder_config)
useml.commit("joint training", val_loss=0.18)
```

### Config

```python
from useml import Config

config = Config(
    epochs=30,
    lr=1e-3,
    optimizer="adamw",          # "adam" | "adamw" | "sgd"
    loss="cross_entropy",       # str key, nn.Module class/instance, or callable
    batch_size=128,
    checkpoint_every=5,         # commit every N epochs during useml.train()
    device="auto",              # "auto" | "cpu" | "cuda" | "mps"
)
```

### Dashboard

```python
useml.show()
# --- UseML Dashboard ---
# Vault:   vault/
# Project: my-experiment (3 snapshots)
# Tracked: ['classifier']
# Latest:  baseline [snap_20240501_...]
```

## Level-0 — train in two lines

```python
import useml
from useml import Config, Model
import torch.nn as nn

class Net(Model):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

useml.init("vault")
useml.new("mnist-quick", auto_focus=True)

history = useml.train(Net, "mnist", config=Config(epochs=10))
# Built-in datasets: "mnist", "fashion_mnist", "cifar10", "cifar100"
# HuggingFace:       "hf:<dataset_name>"
# Custom:            any torch.utils.data.Dataset
```

### Custom step function

Override the default `criterion(model(x), y)` without subclassing:

```python
def ae_step(model, batch, device):
    x, _ = batch
    x = x.to(device)
    return F.mse_loss(model(x), x)   # target = input

useml.train(AutoEncoder, "mnist", config=config, step_fn=ae_step)
```

### Config custom fields

`Config` accepts any extra keyword argument as a custom field. Custom fields
are serialised to the snapshot YAML and forwarded to the model constructor
by `useml.load()`.

```python
config = Config(epochs=10, lr=1e-3, latent_dim=16, kl_weight=1e-3)
config.latent_dim   # 16
config.to_dict()    # includes latent_dim and kl_weight in snapshot YAML

# On load, useml calls VAE(latent_dim=16) automatically
model = useml.load("vae")
```

## Level-1 — DataBundle (versioned data contracts)

`DataBundle` pairs a raw dataset with a preprocessing transform and hashes
both at every snapshot. The raw data is stored once; only the transform code
and three hashes are embedded in the snapshot.

```python
from useml import DataBundle

bundle = DataBundle(
    name="polar_clouds",
    source=CartesianDataset(),      # raw data — any torch Dataset or built-in name
    transform=gridify,              # per-sample callable: (sample) -> sample
    version="v1",                   # optional label
)

history = useml.train(MyCNN, bundle, config=config)
```

Three hashes are recorded in `metadata.yaml` for every snapshot that uses the bundle:

| Hash | What it tracks |
|---|---|
| `source_hash` | Fingerprint of the raw dataset |
| `transform_hash` | sha256 of the transform function's source code + name |
| `cache_key` | sha256(source_hash + transform_hash) — used for disk cache |

The transform source code is also archived verbatim in `source/_bundle_<name>_<fn>.py`
inside the snapshot, so the exact preprocessing of any checkpoint is always recoverable.

### Factory transforms

When a transform is produced by a factory (different params, same algorithm),
set a descriptive `__name__` so the hashes and archived file names are distinct:

```python
def make_gridify(n_r_bins, n_theta_bins):
    def gridify(sample):
        ...
    gridify.__name__ = f"gridify_{n_r_bins}r_{n_theta_bins}t"
    return gridify

bundle_fine   = DataBundle("ds", raw_ds, transform=make_gridify(16, 32))
bundle_coarse = DataBundle("ds", raw_ds, transform=make_gridify(8, 16))
# → different transform_hash, different archived .py file
```

### Optional disk cache

For expensive transforms, set `cache=True` to materialise the processed dataset
on first run and reload it on subsequent runs:

```python
bundle = DataBundle("ds", raw_ds, transform=heavy_preprocess, cache=True)
# Saved to: <data_dir>/.bundle_cache/<cache_key>.pt
```

### Manual commit with bundle metadata

When calling `useml.train()`, bundle metadata flows into the snapshot automatically.
For manual workflows, pass it explicitly:

```python
useml.track("model", model, config=config)
useml.commit("run 1", bundle_meta=bundle.to_meta_dict())
```

## Loading models

```python
# Load latest weights (uses current code)
model = useml.load("classifier")

# Load with full code isolation (archived source + weights)
useml.mount("\\latest")
model = useml.load("classifier")

# Load from a specific snapshot
model = useml.load("classifier", _from="\\head~2")

# Import archived code directly
useml.mount("\\latest")
from useml.workdir.models import Net
```

## Snapshot tags

| Tag | Meaning |
|---|---|
| `\latest` | Most recent snapshot |
| `\head~N` | N snapshots before latest |
| `\current` / `\workdir` | Unmount (back to live code) |
| `snap_20240501_...` | Exact snapshot folder name |

## Custom loss

```python
# String key (built-in)
Config(loss="cross_entropy")   # also: "mse", "bce", "l1"

# Class
Config(loss=nn.HuberLoss)

# Instance
Config(loss=nn.HuberLoss(delta=0.5))

# Plain function
def focal_loss(pred, target): ...
Config(loss=focal_loss)
```

The loss source is automatically archived in the snapshot.

## Stashing (multi-project workflows)

```python
useml.focus("project-a")
useml.track("model", model_a)
useml.stash()                  # park project-a in RAM

useml.focus("project-b")
useml.track("model", model_b)
useml.commit("run b")

useml.focus("project-a")       # restore from stash
useml.commit("run a")
```

## Error codes

All exceptions raised by useml carry a `[UML-NNN]` code. See [ERRORS.md](ERRORS.md) for the full catalog with causes and fixes.

## API reference

| Function | Description |
|---|---|
| `useml.init(vault_path)` | Connect session to a vault directory |
| `useml.new(name)` | Create and focus a new project |
| `useml.focus(name)` | Resume an existing project |
| `useml.track(name, model, config?, optimizer?)` | Register a component |
| `useml.commit(message, **metrics)` | Save a snapshot |
| `useml.load(name, _from?)` | Restore a model with weights |
| `useml.mount(tag)` | Mount a snapshot for workdir imports |
| `useml.show()` | Print session dashboard |
| `useml.stash()` | Park current project state in RAM |
| `useml.projects()` | List all projects in the vault |
| `useml.train(model_cls, dataset, config?, step_fn?)` | Level-0 training entry point |
| `useml.debug_imports()` | Inspect workdir importable modules |

**Level-1**

| Class | Description |
|---|---|
| `useml.DataBundle(name, source, transform?, version?, cache?)` | Versioned data contract with hash tracking |
| `bundle.source_hash()` | Fingerprint of the raw dataset |
| `bundle.transform_hash()` | sha256 of the transform source + name |
| `bundle.cache_key()` | Combined hash used for disk cache |
| `bundle.to_meta_dict()` | Dict embedded in snapshot `metadata.yaml` under `data:` |
| `bundle.transform_source()` | Source code of the transform (for archival) |
