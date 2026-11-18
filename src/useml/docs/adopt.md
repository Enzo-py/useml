# Adopting useml in an existing project

`useml init` looks at the machine and the project, then proposes what to add. By default
it writes nothing: it prints a plan and asks for confirmation.

```bash
cd my-project
useml init                  # print the plan, ask
useml init --yes            # write without asking
useml init --no-machine     # only the project files
useml init --no-project     # only the machine config
```

## What it never does

- overwrite a value that is already set;
- move or delete a file, a cache or a dataset;
- touch an experience written by another tool.

Every file it changes is first copied to `<file>.useml-backup-<date>`. Running it twice
adds nothing the second time.

## What it looks for

**Machine** (`~/.config/useml/config.toml`):

- a cache root: `--root`, or a `.cache_root` file at the project root (first line: the
  root; lines `VARIABLE=path`: caches placed elsewhere, for example `HF_HOME=/data/hf`);
- an existing cache store: `--store`, a `.store_root` file, or
  `.cache/store/index.sqlite` — it is used in place, nothing is copied;
- the download policy: `[hub] parallel = false` is added if absent.

**Project** (`pyproject.toml` under `[tool.useml]`, or a new `useml.toml`):

- a folder of YAML configs (`configs/`, `config/` or `conf/`) whose files have a `type:`
  key. Its sub-folders become kinds (`configs/models` → kind `model`);
- the packages that define those types, found by scanning the project for
  `class Name` lines. Types it cannot find are reported, never guessed;
- an `experiences/` folder.

**Git** (`.gitignore`): rules for what runs write (step outputs, logs, code snapshots,
bundles), unless git already ignores them. It also warns about leftover merge-conflict
markers in `.gitignore`.

## Example plan

```text
• create ~/.config/useml/config.toml   (machine config)
    + root = "/data/me/useml"
    + [hub]
    + parallel = false
• add to pyproject.toml   (project settings)
    + [tool.useml.config]
    + root = "configs"
    + kinds = {model = "models", metric = "metrics"}
    + default_kind = "model"
    + packages = ["mypkg"]
    + pythonpath = ["."]
    +
    + [tool.useml.exp]
    + root = "experiences"
  note: configs: configs/ with kinds ['model', 'metric']; 12 types used, defined in ['mypkg']
```

## After init

```bash
useml check
```

`useml check` prints one line per item, `✓` (fine), `⚠` (works, could be better) or `✗`
(broken):

- machine config, cache root (writable, free space), download policy;
- caches still in the home folder (`--measure` gives their size);
- the cache store;
- every config file reads and every `type:` resolves;
- experiences, including those that died without a word;
- tmux, and whether the machine kills processes at logout.

It exits with code 1 if any line is `✗`.

## Moving caches out of the home folder

```bash
useml hub scan                    # what is in the home, how big, where it would go
useml hub scan --move             # the plan
useml hub scan --move --apply     # do it (never overwrites; conflicts stay in place)
```

## Experiences from another tool

If `experiences/` already holds experiments written by another system, useml leaves them
alone. useml marks its own declarations (`"format": "useml-exp/1"`); others are not
listed, not opened, and never overwritten.
