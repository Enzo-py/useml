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
    ├── versions.yaml       ← model version index
    ├── snap_20240501_.../  ← snapshot (one per checkpoint)
    │   ├── weights/        ← model .pth files
    │   ├── optimizers/     ← optimizer states (for resume)
    │   ├── configs/        ← per-component YAML configs
    │   ├── source/         ← archived source code
    │   ├── manifest.yaml   ← component registry
    │   └── metadata.yaml   ← message + metrics
    └── snap_20240502_.../
```

## Basic usage

```python
import useml
from useml import Config

useml.init("vault")
project = useml.new("mnist-cnn")      # returns a Project
# or: project = useml.focus("mnist-cnn")  to reopen existing
```

`useml.*` only exposes setup (`init` / `new` / `focus`) and the core classes
(`Config`, `Model`, `Loss`, `DataBundle`). Everything else — training, loading,
comparing — lives on the **project**.

---

## Level-0 — train in two lines

```python
from useml import Config

config = Config(
    epochs=10,
    lr=1e-3,
    loss="cross_entropy",
    checkpoint_strategy="best",
)

run = project.runs.new(ConvNet, "mnist", config=config)
history = run.train()
```

Built-in dataset strings: `"mnist"`, `"fashion_mnist"`, `"cifar10"`, `"cifar100"`,
`"hf:<dataset_name>"` (HuggingFace), or any `torch.utils.data.Dataset`.

### Pass your own loaders

```python
run = project.runs.new(ConvNet, config=config)
history = run.train(train_loader, val_loader)
```

### Custom step function

Override the default `criterion(model(x), y)` without subclassing:

```python
def ae_step(model, batch, device):
    x, _ = batch
    return F.mse_loss(model(x.to(device)), x.to(device))   # target = input

run = project.runs.new(AutoEncoder, "mnist", config=config)
history = run.train(step_fn=ae_step)
```

---

## Inspecting results

```python
project.show()
# --- Project: mnist-cnn ---
# Runs:    5
# Models:  convnet(5)
# Latest:  epoch 10 [snap_20240502_...]

# Leaderboard
print(project.runs.leaderboard("val_loss"))

# Best model
best = project.models["convnet"].best("val_loss")
model = best.load()
print(best.metrics)   # {"val_loss": 0.07, "epoch": 8}
```

---

## Model registry

After training, each saved checkpoint becomes a versioned model entry:

```python
reg = project.models["convnet"]   # ModelRegistry

reg["v1"].load()                  # load specific version
reg["v1"].metrics                 # {"val_loss": 0.12, "epoch": 5}
reg["v1"].config                  # {"lr": 0.001, "epochs": 10, ...}

reg.latest.load()                 # most recent checkpoint
reg.best("val_loss").load()       # best by metric (min by default)
reg.best("accuracy", mode="max")  # best by metric (max)

for v in reg:
    print(v.label, v.metrics)
```

---

## Runs view

```python
runs = project.runs

runs.latest                          # most recent RunRecord
runs.best("val_loss")                # best run by metric
list(runs)                           # all RunRecords, newest first
runs.filter(model="convnet")         # filter by component name
runs.leaderboard("val_loss", top=5)  # formatted table (str)
```

Each `RunRecord` lets you reload a model:

```python
record = project.runs.latest
model = record.load()               # single-model run → no name needed
model = record.load("encoder", model=Encoder())
```

---

## Config

`Config` accepts standard fields plus any custom keyword argument.
Custom fields are serialised to the snapshot YAML and forwarded to the model
constructor by `ModelVersion.load()`.

```python
config = Config(
    epochs=30,
    lr=1e-3,
    optimizer="adam",           # "adam" | "adamw" | "sgd"
    loss="cross_entropy",       # str | nn.Module class/instance | callable
    batch_size=128,
    val_split=0.1,
    seed=42,
    checkpoint_strategy="best", # "every_n" (default) | "best" | "last"
    checkpoint_metric="val_loss",
    checkpoint_mode="min",
    checkpoint_every=5,         # used when strategy="every_n"
    early_stop_patience=10,
    device="auto",              # "auto" | "cpu" | "cuda" | "mps"
    # Custom fields — stored in YAML and injected into model constructor:
    latent_dim=16,
    kl_weight=1e-3,
)

config.latent_dim   # 16
config.to_dict()    # all fields including custom ones
```

On reload, `useml` reads the snapshot YAML and calls `Model(latent_dim=16)`
automatically — no hardcoded defaults needed:

```python
model = project.models["vae"]["v1"].load()   # VAE(latent_dim=16) auto-injected
```

---

## Level-2 — scheduler, metrics, checkpoint strategies

### LR scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

run = project.runs.new(Net, dataset, config=config)
run.scheduler = CosineAnnealingLR(run.optimizer, T_max=config.epochs)
history = run.train()
```

Or pass a factory to `train()`:

```python
history = run.train(
    scheduler=lambda opt: CosineAnnealingLR(opt, T_max=config.epochs)
)
```

`ReduceLROnPlateau` is detected and receives the monitored metric automatically.
Set `run.scheduler_per_batch = True` for per-batch schedulers (`OneCycleLR`).

### Custom metrics

```python
def accuracy(model, batch, device):
    x, y = batch
    return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()

history = run.train(metrics={"accuracy": accuracy})
# history keys: train_loss, val_loss, train_accuracy, val_accuracy
```

Metrics are embedded in every checkpoint's `metadata.yaml`:

```yaml
metrics:
  val_loss: 0.07
  val_accuracy: 0.94
  epoch: 8
```

### Best-val checkpointing

```python
config = Config(
    checkpoint_strategy="best",
    checkpoint_metric="accuracy",
    checkpoint_mode="max",
)
```

`"last"` writes a single snapshot at the very end of training.

### Early stopping

```python
config = Config(
    early_stop_patience=5,
    early_stop_metric="val_loss",
    early_stop_mode="min",
)
```

---

## Level-2 — Pattern B (custom training loop)

For GANs, diffusion, multi-optimizer setups — anything that doesn't fit
the standard supervised loop:

```python
config = Config(
    epochs=50,
    checkpoint_strategy="best",
    checkpoint_metric="g_loss",
    checkpoint_mode="min",
)

run = project.runs.new(config=config)     # no model yet
run.register("G", Generator(), optimizer=opt_G)
run.register("D", Discriminator(), optimizer=opt_D)

for epoch in run.epochs(config.epochs):
    for batch in loader:
        # your adversarial step here
        g_loss = ...
        d_loss = ...
        run.update(n=batch_size, g_loss=g_loss, d_loss=d_loss)
    run.epoch_end(epoch)   # aggregate + log + checkpoint G and D atomically
```

`register()` returns `self` for chaining. All registered models are always
committed **atomically** — no risk of G and D from different epochs.

### on_epoch_end hook

```python
class MyRun(Run):
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        if epoch % 10 == 0:
            self.scheduler.base_lrs = [lr * 0.5 for lr in self.scheduler.base_lrs]
```

---

## Training resume

```python
# Pattern A
run = project.runs.new(Net, dataset, config=Config(epochs=20))
history = run.train(resume_from="\\latest")   # continues from saved epoch

# Pattern B
run = project.runs.new(config=config)
run.register("G", Generator(), optimizer=opt_G)
run.resume("\\latest")                         # loads weights + optimizer state

for epoch in run.epochs(config.epochs):        # starts from saved_epoch + 1
    ...
```

---

## Level-1 — DataBundle (versioned data contracts)

`DataBundle` pairs a dataset with a preprocessing transform and hashes both.
The transform source code is archived verbatim in every snapshot.

```python
from useml import DataBundle

bundle = DataBundle(
    name="polar_clouds",
    source=CartesianDataset(),      # raw data — any Dataset or built-in name
    transform=gridify,              # per-sample callable: sample -> sample
    version="v1",
    cache=True,                     # materialise on first run, reload after
)

run = project.runs.new(MyCNN, bundle, config=config)
history = run.train()
```

Three hashes are recorded in `metadata.yaml`:

| Hash | What it tracks |
|---|---|
| `source_hash` | Fingerprint of the raw dataset |
| `transform_hash` | sha256 of the transform function source + name |
| `cache_key` | sha256(source_hash + transform_hash) |

### Factory transforms

```python
def make_gridify(n_r, n_t):
    def gridify(sample): ...
    gridify.__name__ = f"gridify_{n_r}r_{n_t}t"
    return gridify

bundle_fine   = DataBundle("ds", raw, transform=make_gridify(16, 32))
bundle_coarse = DataBundle("ds", raw, transform=make_gridify(8, 16))
# → different transform_hash, different archived .py file
```

---

## Code isolation (workdir)

```python
from useml.session.manager import _session

_session.mount("\\latest")

from useml.workdir import mymodel   # loads archived source from snapshot
model = mymodel.MyModel()           # uses the code version from the checkpoint

_session.mount("\\workdir")         # unmount, back to live code
```

Snapshot tags:

| Tag | Meaning |
|---|---|
| `\latest` | Most recent snapshot |
| `\head~N` | N snapshots before latest |
| `\workdir` / `\current` | Unmount |
| `snap_20240501_...` | Exact folder name |

---

## Custom loss

```python
Config(loss="cross_entropy")          # built-in: "cross_entropy", "mse", "bce", "l1"
Config(loss=nn.HuberLoss)             # class
Config(loss=nn.HuberLoss(delta=0.5))  # instance
Config(loss=focal_loss)               # callable — source archived automatically
```

---

## Error codes

All exceptions carry a `[UML-NNN]` code. See [ERRORS.md](ERRORS.md) for the
full catalog with causes and fixes.

---

## API reference

### Setup

| | |
|---|---|
| `useml.init(vault_path)` | Connect to a vault directory |
| `useml.new(name)` → `Project` | Create a new project |
| `useml.focus(name)` → `Project` | Open an existing project |

### Project

| | |
|---|---|
| `project.runs` | `RunsView` — all runs, factory for new ones |
| `project.models` | `ModelsView` — versioned model registry |
| `project.log()` | List of snapshots, newest first |
| `project.show()` | Print project summary |

### RunsView / Run

| | |
|---|---|
| `project.runs.new(Model, dataset?, config=, name=)` | Create a training run |
| `run.train(loaders?, step_fn?, metrics?, scheduler?, resume_from?)` | Run Pattern A |
| `run.register(name, model, optimizer?)` | Register a model (Pattern B) |
| `run.epochs(n)` | Epoch iterator (Pattern B) |
| `run.update(n=, **metrics)` | Accumulate batch metrics (Pattern B) |
| `run.epoch_end(epoch)` | Aggregate + checkpoint (Pattern B) |
| `run.resume(tag)` | Load weights + optimizer states |
| `project.runs.latest` | Most recent `RunRecord` |
| `project.runs.best(metric, mode=)` | Best `RunRecord` |
| `project.runs.filter(model=)` | Filtered `RunsView` |
| `project.runs.leaderboard(metric, top=)` | Formatted table (str) |

### ModelsView / ModelRegistry / ModelVersion

| | |
|---|---|
| `project.models["name"]` | `ModelRegistry` for one component |
| `registry.latest` | Most recent `ModelVersion` |
| `registry.best(metric, mode=)` | Best `ModelVersion` |
| `registry["v1"]` | Specific version |
| `version.load(model=None)` | Restore weights (auto-instantiate if omitted) |
| `version.metrics` | `{"val_loss": 0.07, "epoch": 8, ...}` |
| `version.config` | Config dict from snapshot YAML |

### DataBundle

| | |
|---|---|
| `DataBundle(name, source, transform=, version=, cache=)` | Create a data contract |
| `bundle.source_hash()` | Raw dataset fingerprint |
| `bundle.transform_hash()` | Transform source hash |
| `bundle.cache_key()` | Combined hash |
| `bundle.to_meta_dict()` | Dict for snapshot `metadata.yaml` |
