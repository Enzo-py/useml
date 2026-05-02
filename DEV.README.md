# useml — Developer Guide

## Setup

```bash
git clone https://github.com/Enzo-py/useml
cd useml
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -vv          # full suite with details
pytest -x -q        # stop on first failure, minimal output
pytest tests/test_session.py   # single file
```

All tests must pass before merging.

## Project structure

```
useml/
├── __init__.py           # public API surface (init, track, commit, …)
├── errors.py             # error catalog — all UML-NNN codes, causes, fixes
├── imports.py            # snapshot module loader + ImportManager
│
├── session/
│   ├── manager.py        # Session singleton + focus/stash/track/commit/mount/load
│   └── component.py      # Component dataclass (model + config + optimizer)
│
├── vault/
│   ├── core.py           # Vault — root directory, project registry
│   ├── project.py        # Project — list of snapshots, commit()
│   ├── snapshot.py       # Snapshot — save/load weights, configs, source
│   └── code_extractor.py # Source archiving (local files + notebook cells)
│
├── template/
│   ├── config.py         # Config dataclass (hyperparams + loss introspection)
│   ├── model.py          # Model base class (nn.Module + forward contract)
│   ├── loss.py           # Loss base class
│   ├── trainer.py        # Trainer + _build_loss / _build_optimizer + run_training()
│   └── dataset.py        # load_dataset() — torchvision / HuggingFace / custom
│
└── workdir/
    ├── __init__.py       # installs import hook, exposes mount state
    └── _hook.py          # WorkdirImportFinder + WorkdirLoader
```

## Architecture

### Vault hierarchy

```
Vault (root dir)
└── Project (one per experiment name)
    └── Snapshot (one per commit, folder: snap_YYYYMMDD_HHMMSS_ffffff)
        ├── weights/      .pth per component
        ├── configs/      .yaml per component (from Config.to_dict())
        ├── optimizers/   .pth per component (optional)
        ├── source/       archived Python source tree
        ├── manifest.yaml component registry (class, module, hashes)
        └── metadata.yaml commit message + user metrics
```

### Session singleton

`useml.session.manager._session` is the global `Session` instance. The public API in `useml/__init__.py` is a thin proxy over it. Tests reset it via the `fresh_session` fixture in `conftest.py`.

### workdir import hook

`useml.workdir._hook.WorkdirImportFinder` intercepts any `import useml.workdir.*` and redirects it to the currently mounted snapshot's `source/` directory. The hook is installed as a `sys.meta_path` finder when `useml.workdir` is first imported.

### Error catalog

All custom exceptions live in `useml/errors.py`. Each has:
- A unique `UML-NNN` code (class attribute)
- An entry in `_CATALOG` with `message`, `cause`, and `fix`
- Dual inheritance: `UseMlError` + the matching Python built-in (`ValueError`, `FileNotFoundError`, etc.)

To regenerate `ERRORS.md` after adding a new error:
```bash
python3 -c "from useml.errors import generate_errors_md; open('ERRORS.md','w').write(generate_errors_md())"
```

Code ranges:
- `UML-1xx` — Session (vault connection, focus, dirty state)
- `UML-2xx` — Vault / Project
- `UML-3xx` — Snapshot (overwrite, missing, weights)
- `UML-4xx` — Import / workdir
- `UML-5xx` — Training (loss, optimizer)
- `UML-6xx` — Dataset

## Adding a new error

1. Add an `_ErrorDef` entry to `_CATALOG` in `errors.py`
2. Add a concrete exception class that sets `code` and inherits the right Python base
3. Replace the generic `raise ValueError/RuntimeError/...` at the raise site
4. Regenerate `ERRORS.md`

## Build & publish

Bump version in `pyproject.toml`, then:

```bash
rm -rf dist/ build/ *.egg-info
python3 -m build
twine check dist/*
twine upload dist/*
```
