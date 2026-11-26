# Reference

## Machine config: `~/.config/useml/config.toml`

(`$USEML_CONFIG` names another file; `$XDG_CONFIG_HOME` is respected.)

```toml
root = "/data/me/useml"            # the cache root of this machine

[env]                              # caches placed elsewhere than under the root
HF_HOME = "/data/shared/hf"

[hub]
parallel = false                   # true: allow parallel downloads

[store]
root = "/data/me/useml/store"      # default: <root>/store
max_size = "450GB"                 # the budget of `useml store gc`

[data]
root = "/data/me/useml/data"       # default: <root>/data
max_size = "300GB"                 # refuse to write a table beyond this

[datasets]
root = "/data/me/datasets"         # default: <root>/datasets

[monitor]
space = "<you>/useml-monitor"      # see monitor.md for every key
machine = "gpu-a"
share = ["status", "steps", "progress", "errors", "metrics"]

[exp]
root = "/data/me/experiments"      # used when the project does not set one
notify = "my-command"              # run when an experience ends or crashes
```

Sizes accept `B`, `KB`, `MB`, `GB`, `TB` (binary units): `"450GB"`, `"1.5 TB"`.

Edit it with `useml settings` rather than by hand: comments and order are kept, the
previous file is saved as `config.toml.bak`, unknown keys and wrong types are refused.

```bash
useml settings show                            # every known key, its value and meaning
useml settings get monitor.share
useml settings set monitor.share '["status", "steps", "errors"]'
useml settings set env.HF_HOME /data/shared/hf
useml settings unset store.max_size
```

Values are TOML (`50`, `true`, `"text"`, `["a", "b"]`); a bare word is a string.

## Project settings

Either `useml.toml` at the project root:

```toml
[config]
root = "configs"
packages = ["mypkg"]
kinds = {model = "models"}
default_kind = "model"
pythonpath = ["."]

[datasets]
packages = ["mypkg.drivers"]       # where dataset drivers are registered
root = "data"                      # optional

[exp]
root = "experiences"
```

or the same tables under `[tool.useml.config]` and `[tool.useml.exp]` in
`pyproject.toml`. Paths are relative to the file. The nearest file above the current
folder wins.

## Environment variables

| Variable | Effect |
|---|---|
| `USEML_ROOT` | the cache root (overrides `root`) |
| `USEML_CONFIG` | the machine config file |
| `USEML_NO_AUTO=1` | `import useml` changes nothing (call `paths.apply()` and `hub.apply_configured_policy()` yourself) |
| `USEML_STORE_ROOT` | the store folder |
| `USEML_DATA_ROOT` | the processed-data folder |
| `USEML_DATASETS_ROOT` | the datasets folder |
| `USEML_EXP_ROOT` | the experiences folder |
| `USEML_CONFIG_ROOT` | the config folder (no type package) |
| `USEML_NO_TMUX=1` | never start runs in tmux |

## Commands

| Command | What it does |
|---|---|
| `useml init [--root DIR] [--store DIR] [--yes] [--no-machine] [--no-project]` | set up the machine and the project; adds, never overwrites |
| `useml check [--measure]` | check that everything is in place |
| `useml docs [topic] [--path]` | read this documentation |
| `useml settings show` / `get <key>` / `set <key> <value>` / `unset <key>` | read and edit the machine config, keeping comments |
| `useml paths` | every cache variable, before and after |
| `useml hub ls [--refresh] [--json]` | recorded downloads |
| `useml hub get <uri> [--dest DIR] [--extract] [--refresh]` | download, print the path |
| `useml hub status` | which layers of the download policy hold |
| `useml hub scan [--move] [--apply]` | caches left in the home folder |
| `useml hub watch [--pid N] [--max-connections N] [--interval S] [--sustained N] [--no-kill] [-- command]` | the external network monitor |
| `useml store info` / `spaces` / `usage [--days N]` | store size, spaces, daily hits |
| `useml store check [--deep] [--repair]` | inconsistencies |
| `useml store gc [--max-size S] [--older-than DAYS] [--space S] [--include-pinned] [--include-precious] [--apply]` | clean |
| `useml store move <dest> [--delete-source]` | move the store |
| `useml store unpin <owner>` | release pins |
| `useml data ls` / `tables` / `info <name>` / `jobs` / `follow <job>` / `rm <name> [--table]` | processed datasets |
| `useml datasets ls` / `info <name> [--split S]` / `prepare <name> [--force]` / `check <name>` | datasets and their drivers |
| `useml monitor snapshot [--light]` / `serve [--port N]` / `deploy [--space ID]` / `start` / `stop` / `status` | the private dashboard and its agent |
| `useml config ls` / `describe <name>` / `show <name> [k=v ...]` / `find <hash>` | configs |
| `useml exp ls` / `show` / `follow` / `log [-n N]` / `run [--no-follow]` / `stop [--now]` / `why` / `attach` / `stack` `<name>` | experiences |
| `useml exp bundle <name> [-o DIR] [--data auto\|rows\|none] [--archive] [--checkpoints] [--partial]` | standalone archive |
| `useml exp verify <bundle>` | check a bundle's checksums |

`python -m useml ...` works the same as `useml ...`.

## Python API at a glance

```python
import useml
from useml import paths, hub, store, data, config, exp
from useml import datasets as ds

paths.root(); paths.home(); paths.apply(); paths.diagnose()
hub.get(uri, dest=None, allow=None, ignore=None, extract_archive=False, refresh=False)
hub.adopt(uri, path); hub.ls(); hub.scan(); hub.move(path, apply=False)
hub.sequential(); hub.allow_parallel(); hub.status()

cache = store.open(); space = cache.declare("family/name", cost="expensive")
space.put(key, value); space.get(key); space.put_file(key, write, extension=".png")

data.processor(name, version=1, batch_size=32, dtype="float16", ignore=("device", "batch_size"), workers=4, load=None)
data.process(source, processor, name=None, batch_size=None, detach=False, dry_run=False, **params)
data.load(name); data.ls(); data.tables(); data.rm(name, table=False); data.jobs()

ds.load(name, split=None); ds.prepare(name, force=False); ds.ls(); ds.register(DriverClass)
config.load(name, kind=None, reload=False, overrides=None, **overrides)
config.load_raw(...); config.describe(name); config.find(hash)
config.create(name, obj); config.get_config(obj); config.diff(a, b)
config.use(root, packages=[...]); config.register(cls)

exp.experience(name, note="", run=True, detach=True, follow=True, gpu=None)
exp.get("experience/step"); exp.open(name); exp.ls(); exp.running()
exp.bundle(name, destination=None, data="auto", checkpoints=False, archive=False)
```
