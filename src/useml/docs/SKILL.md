---
name: useml
description: Use when a project uses (or should use) the `useml` Python package — caches and downloads (useml.paths, useml.hub), the cache store (useml.store), dataset drivers (useml.datasets), datasets processed once and loaded by name (useml.data), YAML configs (useml.config), experiences run in the background (useml.exp), `useml init` / `useml check`, or a run that crashed. Read before writing code that calls useml or before diagnosing a useml run.
---

# Working with useml

useml's full documentation is installed with it. Read the relevant guide before writing
code, instead of guessing the API:

```bash
useml docs              # list the guides and their folder
useml docs exp          # print one guide: quickstart, adopt, paths-and-hub, store, data, datasets,
                        # config, exp, monitor, troubleshooting, reference, faq
```

## First steps in a project

0. A project's methods, datasets and focus live in `[project]` of useml.toml
   (`useml docs projects`); declare experiences with `method=` so their runs are grouped,
   and keep step arguments plain (named values, configs with overrides): they become the
   factors of the ablation map.
1. `useml check` — see what is set up and what is broken; each problem comes with the
   command that fixes it (show it to the user before running it). Change machine settings with
   `useml settings set <key> <value>` (keeps comments, refuses typos), not by rewriting the
   TOML file. If useml is not set up, propose
   `useml init` (it prints a plan and writes nothing until confirmed; it only adds, never
   overwrites). Do not run `useml init --yes` without the user's agreement.
2. `import useml` must come before torch / transformers / open_clip in entry points.

## Rules that prevent the common mistakes

- **Step functions of an experience live in a module** (a `.py` file of the project),
  never in a notebook cell or inside another function.
- **Arguments of a step must be describable**: plain values, references returned by
  earlier `e.step(...)`, objects from `config.load(name, **overrides)`, datasets from
  `data.load(name)`, outputs from `exp.get(...)`. Never pass an object built by hand or
  modified after `config.load`; use overrides instead.
- **The `with exp.experience(...)` block runs nothing.** Do not use a step's result inside
  the block (no `if result > x:`); read results after the run with `exp.get(...)`.
- **Process datasets with `useml.data`, outside experiences**, give them a name
  (`data.process(..., name="...")`), and load them in experiences with `data.load(name)`.
  Do not re-embed data inside an experience step.
- **Datasets are declared by drivers** (`useml docs datasets`): load them with
  `datasets.load(name, split)`; never hard-code their download or parsing in a step. A
  filtered dataset cannot be passed to an experience: declare a split instead.
- **Processor rows have a fixed shape.** Pool or pad variable-length outputs.
- **Bump a processor's `version`** when its code changes what it computes.
- **Declare a store space before writing to it**: `store.open().declare("family/name",
  cost="cheap" | "expensive" | "precious")`.
- **Downloads go through `useml.hub.get(...)`** and are sequential by default. Do not add
  thread pools for downloads; do not call `hub.allow_parallel()` unless the user asks.
- **The monitor sends data to a Hugging Face Space** (`useml docs monitor`). Before `useml monitor deploy` or
  `start`, show the user `useml monitor snapshot` and their `[monitor] share`; never widen
  `share` or remove `redact` entries on your own.
- **Never delete** caches, tables, experiences or store entries without the user's
  explicit agreement. Cleaning commands are dry runs until `--apply`.

## Diagnosing a run

```bash
useml exp ls
useml exp why <name>      # cause of a crash: signal, OOM, traceback, last log lines
useml exp stack <name>    # where a running run is (seems stuck)
useml exp log <name> -n 80
```

`SIGKILL` usually means the machine ran out of RAM; `SIGSEGV`/`SIGABRT` a native crash
(the Python stack is in run.log); `died` means the run and its supervisor vanished
(reboot, logout kill: see `loginctl enable-linger`). After a fix, `useml exp run <name>`
resumes; finished steps are skipped.

## Privacy

When writing examples, issues or anything public about a project that uses useml, use
neutral names; never copy private project names, paths, hostnames or emails.
