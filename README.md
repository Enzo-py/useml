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

## Level-0 API — train in two lines

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
| `useml.train(model_cls, dataset, config?)` | Level-0 training entry point |
| `useml.debug_imports()` | Inspect workdir importable modules |
