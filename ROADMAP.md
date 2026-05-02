# useml — Architecture Roadmap

Each level is a **strict superset** of the one below it.
A user who starts at Level 0 can migrate to Level 1, 2, or 3 by adding pieces
without rewriting what already works. The goal is that the simplest thing that
works is always reachable in two lines, and every step up unlocks more control.

---

## Level 0 — `useml.train()` ✅ Implemented

> *"Train a model in two lines."*

```python
useml.init("vault")
useml.new("experiment-1")
history = useml.train(MyModel, "mnist", config=Config(epochs=10, lr=1e-3))
```

### What it provides
- Automatic dataset loading (built-ins, `hf:`, or custom `Dataset`)
- Train/val split, DataLoader construction, device placement
- Full training loop with per-epoch logging
- Periodic snapshot checkpointing to vault
- `step_fn` escape hatch for non-standard forward passes (AE, SSL, distillation)

### What it does NOT provide
- Control over the validation split (fixed ratio)
- LR scheduling
- Best-val checkpointing
- Callbacks / hooks
- Versioned data contracts

### Current limits
| Code | Limit | Severity |
|------|-------|----------|
| L-3 | No best-val checkpointing | ★★★☆☆ |
| L-4 | Two-optimizer training (GAN) requires manual loop | ★★★☆☆ |
| L-5 | No LR scheduler | ★★★☆☆ |
| L-6 | No training resume (optimizer state not saved) | ★★★☆☆ |
| L-7 | Metrics beyond val_loss require manual eval | ★★☆☆☆ |
| L-8 | Fixed val split — no pre-defined val set | ★★☆☆☆ |

### TODO
- [ ] Optimizer state serialized in snapshot (enables L-6 training resume)
- [ ] `Config.checkpoint_metric` actually used to trigger best-val save (L-3)

---

## Level 1 — `DataBundle` (versioned data contracts)

> *"Same two lines, but your data is now versioned alongside your model."*

Level 0 plus: a `DataBundle` wraps a dataset + its preprocessing transform into
a versioned, hashable object. The bundle's identity is the hash of its source
data + transform code. If nothing changed, loaders come from cache.

```python
# Everything from Level 0 still works as-is.
# Opt in to data versioning by wrapping your dataset:

bundle = useml.DataBundle(
    name="mnist_polar",
    source="mnist",                  # built-in, hf:, or Dataset
    transform=polar_gridify,         # any callable
    version="v1",                    # explicit label (optional)
)

# Pass bundle instead of a string — API is identical from here
history = useml.train(MyModel, bundle, config=Config(...))
```

### What it adds on top of Level 0
- **Code hash** of `transform` stored in every snapshot → alerts if transform
  changes between experiments
- **Data hash** of the processed tensors stored in snapshot manifest
- **Cache layer**: if `(source_hash, transform_hash)` is already on disk, skip
  reprocessing and serve from cache directly
- **Explicit data version** linked to each model snapshot: you can always tell
  *which data produced which weights*
- `DataBundle.diff(other)` — compare two bundles to see if source, transform,
  or output distribution changed

### What it does NOT provide
- Multi-stage pipelines (transform chained with another transform)
- Callback hooks during training

### TODO
- [ ] `DataBundle` class: `name`, `source`, `transform`, `version`
- [ ] `DataBundle._source_hash()` — hash of raw dataset (sample count + first N
  samples fingerprint)
- [ ] `DataBundle._transform_hash()` — `inspect.getsource(transform)` → md5
- [ ] `DataBundle._cache_key()` — `sha256(source_hash + transform_hash)`
- [ ] `DataBundle.loaders(config)` — returns `(train_loader, val_loader)`,
  processes and caches if needed
- [ ] `DataBundle.diff(other)` — prints source / transform / output diffs
- [ ] `run_training()` / `useml.train()` accept `DataBundle` in addition to
  current inputs (duck-typed: anything with `.loaders(config)`)
- [ ] Snapshot manifest extended: `data_bundle: {name, version, source_hash,
  transform_hash, cache_key}`
- [ ] `useml.show()` displays bundle info when available

---

## Level 2 — `Trainer` + Callbacks

> *"Same API, but the training loop is now programmable."*

Level 1 plus: the `Trainer` gains a callback system. Users can plug in
schedulers, best-val checkpointing, early stopping, or custom metric logging
without subclassing or rewriting the loop.

```python
# Everything from Level 0 & 1 still works as-is.
# Opt in to callbacks by using Trainer directly:

trainer = useml.Trainer(model, config)
trainer.add_callback(useml.callbacks.LRScheduler(scheduler))
trainer.add_callback(useml.callbacks.BestValCheckpoint(metric="val_loss"))
trainer.add_callback(useml.callbacks.EarlyStopping(patience=5))
trainer.add_callback(useml.callbacks.MetricLogger(["accuracy"]))

history = trainer.run(train_loader, val_loader)
```

`useml.train()` at Level 0 is still valid — it internally creates a `Trainer`
with no callbacks. Callbacks are purely additive.

### What it adds on top of Level 1
- **Callback protocol**: `on_epoch_end(trainer, epoch, metrics)` (and
  `on_train_begin`, `on_train_end`, `on_batch_end`)
- **`LRScheduler` callback**: wraps any `torch.optim.lr_scheduler`, calls
  `.step()` at the right moment
- **`BestValCheckpoint` callback**: saves a snapshot only when monitored metric
  improves (fixes L-3)
- **`EarlyStopping` callback**: stops training when metric stops improving
- **`MetricLogger` callback**: user provides `compute_fn(model, loader) -> dict`,
  result is appended to history and passed to `commit()` metrics
- **`TwoOptimizerTrainer`** subclass (or callback): handles GAN / multi-task
  setups with alternating optimizer steps (fixes L-4)
- `trainer.save_optimizer` flag: serializes optimizer state into snapshot,
  enabling `resume_from` (fixes L-6)

### What it does NOT provide
- Chained, hash-tracked multi-stage pipelines
- Automatic replay of only changed stages

### TODO
- [ ] Callback base class / protocol (`on_train_begin`, `on_epoch_end`,
  `on_batch_end`, `on_train_end`)
- [ ] `Trainer.add_callback(cb)` and `Trainer._fire(event, **kwargs)`
- [ ] `LRScheduler` callback (wraps `torch.optim.lr_scheduler.*`)
- [ ] `BestValCheckpoint(metric, mode)` callback — commits only on improvement
- [ ] `EarlyStopping(patience, metric, mode)` callback
- [ ] `MetricLogger(compute_fn)` callback — appends custom metrics to history
  and passes them to `commit()`
- [ ] `Trainer.save_optimizer: bool` flag — serializes optimizer `.state_dict()`
  into snapshot; `load()` restores it when present
- [ ] `TwoOptimizerTrainer(model_g, model_d, config)` — alternating step loop
- [ ] `useml.train()` passes `**callbacks` kwarg to Trainer so Level-0 users
  can still add a single callback in one line

---

## Level 3 — `Pipeline` (versioned multi-stage pipelines)

> *"The whole experiment — data prep, training, eval — is one versioned object."*

Level 2 plus: a `Pipeline` chains `Stage` objects (each a `DataBundle`
transform or a training step). Each stage knows its inputs and outputs by name.
The pipeline hashes every stage independently; on re-run, only stages whose
hash changed are re-executed. Results of unchanged stages are served from cache.

```python
# Everything from Levels 0, 1, 2 still works as-is.
# Opt in to full pipeline versioning:

pipeline = useml.Pipeline(name="polar-clf-v1")

pipeline.stage(
    "preprocess",
    fn=polar_gridify,
    inputs=["raw_mnist"],
    outputs=["polar_grids"],
)
pipeline.stage(
    "train",
    fn=useml.train,
    inputs=["polar_grids"],
    outputs=["model"],
    config=Config(epochs=20, n_r_bins=16, n_theta_bins=32),
    callbacks=[useml.callbacks.BestValCheckpoint()],
)
pipeline.stage(
    "eval",
    fn=evaluate_fn,
    inputs=["model", "polar_grids"],
    outputs=["metrics"],
)

results = pipeline.run()
# On re-run with only Config changed → preprocess is cached, train re-runs.
# On re-run with polar_gridify changed → preprocess + train + eval all re-run.
```

### What it adds on top of Level 2
- **`Stage`**: named computation unit with declared inputs/outputs + hash
  (code hash of `fn` + hashes of inputs)
- **Dependency graph**: stages form a DAG; outputs of one stage feed inputs of
  the next
- **Selective re-execution**: only stages with a changed hash are re-run;
  unchanged stages are skipped and their outputs restored from cache
- **Pipeline snapshot**: a top-level manifest records every stage's hash,
  config, and output path — full reproducibility in one object
- **`Pipeline.diff(other)`**: compare two pipeline runs stage-by-stage
- **`Pipeline.replay(from_stage)`**: force re-execution starting from a given
  stage (useful after data fixes)
- Cache acceleration: the more experiments share stages (e.g. same preprocessing
  with different training configs), the more cache hits → framework gets faster
  the more you use it

### TODO
- [ ] `Stage` dataclass: `name`, `fn`, `inputs`, `outputs`, `config`,
  `callbacks`
- [ ] `Stage._code_hash()` — `inspect.getsource(fn)` → sha256
- [ ] `Stage._input_hash(results_dict)` — hash of all input tensors/datasets
- [ ] `Stage._run_hash()` — sha256(code_hash + input_hash + config_hash)
- [ ] `Pipeline.stage(...)` — registers a stage and validates input names are
  available (from prior stages or initial inputs)
- [ ] `Pipeline.run(force=False)` — topological sort, hash check per stage,
  cache restore or re-execute
- [ ] `Pipeline._cache_dir` — per-project cache directory in vault
- [ ] `Pipeline.diff(other)` — stage-by-stage comparison of two runs
- [ ] `Pipeline.replay(from_stage)` — invalidates cache for `from_stage` and
  all dependents, then runs
- [ ] `Pipeline` manifest stored in vault snapshot: all stage hashes, configs,
  output refs
- [ ] `useml.show()` and CLI `useml log` display pipeline structure when present

---

## Level 4 — `Sweep` (experiment suite over a pipeline)

> *"One pipeline, run N times with different configs, ranked automatically."*

Level 3 plus: a `Sweep` is a parametric extension of a `Pipeline`. Instead of
running once with one `Config`, the sweep runs the same pipeline across a grid
or a search strategy and produces a ranked leaderboard. Each run inherits all
Level-3 caching: stages whose hash didn't change between configs are shared
across runs.

```python
# Everything from Levels 0–3 still works as-is.

sweep = useml.Sweep(
    pipeline=pipeline,
    space={
        "train.lr":         [1e-4, 1e-3, 1e-2],
        "train.latent_dim": [16, 32, 64],
        "preprocess.n_r_bins": [8, 16],
    },
    strategy="grid",                   # or "random", "bayesian"
    objective="eval.accuracy",         # which metric to rank by
    direction="maximize",
    budget=20,                         # max runs (for random/bayesian)
)

leaderboard = sweep.run(parallel=4)
# Returns a ranked list of (config, metrics, snapshot_tag).
# Cache hits are massive: only stages downstream of changed inputs re-run.
```

### What it adds on top of Level 3
- **`Sweep`**: parametric expansion of a pipeline over a config space
- **Search strategies**: grid, random, bayesian (via Optuna integration)
- **Cache reuse across runs**: if 20 runs share the same `preprocess` stage
  inputs, that stage runs once — preprocessing time amortizes to O(1)
- **Leaderboard**: ranked table of runs with full provenance (each row points
  to a Pipeline snapshot)
- **Parallel execution**: multiple runs scheduled concurrently
- **Pruning**: early-stop runs whose intermediate metric falls below the
  current best (median pruner / hyperband)
- **`Sweep.diff(top_n=5)`**: compare top-N runs — what config differences
  produced what metric differences
- **Resumable**: a sweep stores its state; interrupted sweeps resume from
  where they stopped

### What it does NOT provide
- Multi-user collaboration / shared registries
- Automated model promotion / serving
- Continuous (drift-triggered) retraining

### TODO
- [ ] `Sweep(pipeline, space, strategy, objective, direction, budget)`
- [ ] `space` syntax: `"stage.config_key": [values]` or distribution objects
- [ ] `Sweep._enumerate(strategy)` — produces config iterables
- [ ] `Sweep.run(parallel=N)` — schedules pipeline runs, collects metrics
- [ ] `Sweep.leaderboard` — ranked DataFrame-like view
- [ ] Optuna adapter for bayesian / hyperband / median pruning
- [ ] `Sweep` state persisted in vault → resume after crash
- [ ] CLI: `useml sweep run`, `useml sweep status`, `useml sweep top`

---

## Level 5 — `Workspace` (collaborative registry + lifecycle)

> *"Your local vault becomes a shared, production-grade ML workspace."*

Level 4 plus: a `Workspace` is a multi-user vault with a registry, promotion
flow, lineage tracking, and serving hooks. Snapshots can be tagged, promoted
across environments, and served behind an inference endpoint. Drift signals
can trigger automatic re-runs of pipelines or sweeps. This is where useml
crosses from "experiment tracking" into "ML platform".

```python
# Everything from Levels 0–4 still works as-is.

workspace = useml.Workspace(remote="s3://team-vault")  # or git-backed

# Promote a snapshot through environments
project = workspace.project("polar-clf")
project.snapshot("\\latest").promote("staging")
project.snapshot("v2-best").promote("production")

# Serve the production snapshot behind an inference endpoint
endpoint = workspace.serve(
    project="polar-clf",
    tag="production",
    entrypoint=lambda model, x: model(x).softmax(-1),
)
endpoint.url   # → "https://workspace.local/polar-clf/production"

# Schedule continuous retraining when drift is detected
workspace.watch(
    pipeline=pipeline,
    trigger=useml.triggers.DataDrift(bundle="mnist_polar", threshold=0.05),
    on_trigger=useml.actions.Sweep(space={"train.lr": [1e-4, 1e-3]}),
)
```

### What it adds on top of Level 4
- **Remote vault**: backend-agnostic storage (S3, GCS, git LFS, custom)
- **Multi-user lineage**: every snapshot records `author`, `git_commit`,
  `pipeline_version`; full who-did-what audit trail
- **Promotion flow**: tag snapshots `staging` / `production` / `archived`;
  promotion is itself an audited event
- **Model registry view**: project-level dashboard listing all "live" model
  versions across environments (orthogonal to commits — see Side Track A)
- **Serving**: spin up a minimal inference server from any snapshot tag;
  optionally containerize it
- **Drift triggers**: data-drift / metric-drift watchers that fire pipelines
  or sweeps automatically
- **Access control**: per-project read/write/promote roles
- **Reproducibility export**: bundle a snapshot + its full lineage (data,
  code, config, environment) into a single archive

### TODO
- [ ] `Workspace` class — wraps `Vault` with a remote backend abstraction
- [ ] Storage adapters: local, S3, GCS, git
- [ ] Snapshot tags: `\\staging`, `\\production`, `\\archived`,
  custom user tags
- [ ] `snapshot.promote(tag)` — atomic move + audit log entry
- [ ] `workspace.serve(project, tag, entrypoint)` — minimal FastAPI endpoint
- [ ] `useml.triggers.*` — `DataDrift`, `MetricDrift`, `Schedule`
- [ ] `useml.actions.*` — `Pipeline`, `Sweep`, `Notify`
- [ ] `workspace.watch(pipeline, trigger, on_trigger)`
- [ ] Lineage manifest: who, when, which git commit, which env
- [ ] CLI: `useml promote`, `useml serve`, `useml watch`
- [ ] Reproducibility archive: `useml export <snapshot> --bundle`

---

## Side Track A — Project-as-workspace view *(orthogonal to levels)*

Currently the user thinks **commit by commit**: `useml.load(name,
_from="\\head~3")` requires knowing which snapshot has which weights.

The proposed change is to add a **project-level view** where models and
datasets are first-class versioned entities *inside* a project. Snapshots
remain the underlying storage mechanism, but the API surface presents a
flatter, more familiar mental model — closer to a model registry.

```python
project = useml.focus("polar-clf")

# Model versions
project.models                      # {"NetA": [v1, v2, v3], "AutoEnc": [v1]}
project.models["NetA"]["v2"].load()
project.models["NetA"].latest()     # → v3
project.models["NetA"].best("val_loss")
project.models["NetA"]["v2"].promote("production")

# Dataset versions
project.datasets                    # {"mnist_polar": [v1, v2]}
project.datasets["mnist_polar"]["v1"]
project.datasets["mnist_polar"].latest().bundle    # DataBundle instance

# Experiments
project.experiments                 # high-level list with metrics summary
project.experiments["sweep-2026-04"].leaderboard

# Snapshots are still accessible — they're now an implementation detail
project.snapshots                   # raw chronological list (rare to use)
```

### Why this matters
- **Discoverability**: `project.models` is more intuitive than scrolling
  `useml log` looking for the right snapshot tag
- **Collaboration**: a teammate sees "Model NetA has versions v1, v2, v3" and
  can pick one without learning your commit conventions
- **Decoupling**: the same snapshot can be tagged as `NetA v2` and `AutoEnc
  v1` independently — versioning is per-component, not global
- **Compatibility**: existing snapshot/commit API stays — this is a *view*
  built on top, not a replacement

### Mapping (snapshots → versions)
- A model "version" is a label attached to one snapshot's component weights
- Versions are auto-incremented per-component on `commit()` if no explicit
  version is provided (`v1`, `v2`, …)
- A user can also tag explicitly: `useml.commit("...", versions={"NetA":
  "v2-pretrained"})`
- The vault stores a per-project `versions.yaml` mapping
  `(component, version) → snapshot_tag`

### TODO
- [ ] Per-project `versions.yaml` index in vault
- [ ] Auto-increment version on `commit()` (configurable: per component or
  per project)
- [ ] `Project.models`, `Project.datasets`, `Project.experiments` views
- [ ] `ModelVersion.load()`, `.config`, `.metrics`, `.snapshot_tag`,
  `.promote(tag)`
- [ ] `Project.experiments` aggregates Sweep results (Level 4)
- [ ] CLI: `useml models list`, `useml models load <name> <version>`
- [ ] `useml.show()` updated to display project view by default
- [ ] Backward-compat: existing `useml.load(name, _from=tag)` still works

---

## Migration Path

```
Level 0   →  Level 1   : wrap dataset in DataBundle — no other change
Level 1   →  Level 2   : use Trainer directly + add_callback — run() unchanged
Level 2   →  Level 3   : wrap Trainer call in a Pipeline stage — same config
Level 3   →  Level 4   : wrap Pipeline in a Sweep — same stages
Level 4   →  Level 5   : point Workspace at a remote — same sweeps + promotion
```

A project can sit at any level permanently; upgrading is always opt-in and
backward-compatible. Level N internally uses Level N-1 infrastructure all the
way down to Level 0.

**Side Track A** (project-view) is orthogonal — it can be added at any level
and benefits all levels above where it's introduced.

---

## Priority Order (suggested)

| Priority | Item | Addresses |
|----------|------|-----------|
| 1 | Callback system + `LRScheduler` + `BestValCheckpoint` | L-3, L-5 |
| 2 | `EarlyStopping` + `MetricLogger` | L-3, L-7 |
| 3 | `save_optimizer` + `resume_from` | L-6 |
| 4 | `DataBundle` + cache layer | L-8, data versioning |
| 5 | **Side Track A** — project-view (`Project.models`, versions) | UX, discoverability |
| 6 | `TwoOptimizerTrainer` | L-4 |
| 7 | `Pipeline` + DAG execution | full pipeline versioning |
| 8 | `Sweep` + leaderboard | hyperparam search |
| 9 | `Workspace` + remote vault | collaboration, production |
