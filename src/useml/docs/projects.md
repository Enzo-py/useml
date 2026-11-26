# Projects: methods, datasets, every run kept

A project groups the experiments of one research question. It answers, on every
computer: what are we trying (the **methods**), on what data (the **datasets**), what are
we working on now (the **focus**), and what did each attempt give (the **runs**).

Four words, with an example:

- **method**: one way of solving the problem, e.g. `wide` = "wider blocks, same budget".
- **run**: one execution of an experience, e.g. `wide/w4-mix` trained once.
- **factor**: a choice that can differ between runs, e.g. `train.width = 4` or
  `train.augment = "mix"`. Factors are read from the declaration: the steps' plain
  arguments, the config each step reads and its overrides, the datasets, the tags.
- **result**: a number a run produced, e.g. `evaluate.accuracy = 0.83`. Results are read
  from the steps' JSON outputs, plus the last and best value of each curve
  (`train.loss.last`, `train.loss.best`).

## Start from what exists

```bash
useml project init          # reads the experiences that exist, prints what it would add
useml project init --yes    # writes it
```

It proposes a method per first part of the experience names (`wide/w4`, `wide/w8` →
method `wide`, matched by `wide/*`), or the method an experience declares; the datasets
the experiences read; and as focus, the method of the experience changed last. It writes
in `useml.toml`, or under `[tool.useml.project]` when the project's useml settings are in
pyproject.toml. It only ADDS: nothing declared is changed, comments stay, the previous
file is kept as `.bak`. Experiences whose names have no group are listed, to name them
or to match them with a method.

## Change it with commands

```bash
useml project describe "Image classifiers that keep their accuracy on corrupted inputs."
useml project method add distill -d "Distil the wide model into a small one" -s idea -e "distill/*"
useml project method set wide -s active          # change only what is given
useml project method rm baseline                 # its runs stay; they become unassigned
useml project focus wide
useml project dataset add cifar10 cifar10-c
useml project dataset rm cifar10-c
```

Each edits one line (or one table) of the file, like `useml settings`, and refuses what
would be wrong: an unknown method as focus, a method that exists already, a name with
spaces. The dashboard's **Add a component** writes these commands for you.

## The declaration, written by hand

```toml
# useml.toml (or the same tables under [tool.useml.project] in pyproject.toml)
[project]
name = "robust-vision"
description = "Image classifiers that keep their accuracy on corrupted inputs."
focus = "wide"                         # the method you work on now
datasets = ["cifar10", "cifar10-c"]    # more are found in the runs

[project.methods.baseline]
description = "ResNet-style network, standard training."
status = "reference"                   # free text: idea, active, reference, dropped…
experiences = ["baseline/*"]           # these experiences try this method

[project.methods.wide]
description = "Wider blocks for the same compute budget."
status = "active"
```

The declaration lives with the code, so every machine that clones the repository knows
the same project. Commit it like code.

An experience belongs to a method in one of two ways:

```python
with exp.experience("wide/w4-mix", method="wide", tags={"variant": "mix"}) as e:
    trained = e.step(train, lr=3e-3, width=4, augment="mix", epochs=12, name="train")
    e.step(evaluate, trained, corruption="noise", name="evaluate")
```

or by matching the `experiences` globs of a method. Neither `method` nor `tags` changes the
fingerprint: they describe the run, they do not change what it computes.

## On one machine

```bash
useml project show                 # methods (focus first), datasets, runs per method
useml project runs                 # every ended run: method, version, first results
useml project export               # Markdown: datasets, methods, a table of runs
useml project export --format json
```

## On every machine: the Projects view of the dashboard

With the [monitor](monitor.md), each agent sends the records of finished runs, and the
dashboard keeps them. The **Projects** tab shows:

- **running now**: the runs of the project going on at this moment, with their progress,
  the last values of their curves and their last log line, updated at every refresh;
- **the methods**, the focus first: number of runs, code versions, best value of the main
  result, last run;
- **the datasets** and which methods used them;
- for a method, **its versions**: a version is the code of the steps (a hash). Two runs
  of the same version ran the same code and differ only by their factors. The page says
  which step's code changed from one version to the next, and the git commit (`*` when
  there were uncommitted changes);
- **the ablation map**: a row per run, the results first (green: better, red: worse,
  within each column; ↑ or ↓ says which way is better, click to switch), then the factors
  that change between runs. The factors that never change are listed once below. Choose
  a reference run (the `ref` column) and every other row highlights what it changed;
- **correlations**: for each factor and each result, `r` (from −1 to +1) for numbers —
  green when more of the factor goes with a better result — and `η` (from 0 to 1) for
  choices among names, how much the choice explains the result. A click draws the runs.
  With few runs, read it as a hint, not a proof;
- **curves** of the runs side by side, the running ones included (thicker, marked
  "live", updated at every refresh): a new attempt is compared with the previous ones while
  it trains. A running run also shows its current values in the ablation map;
- a run's page: results, factors, steps, configs, curves, and **New alternative**: the
  run's declaration as Python, to edit (one value) and run on a machine. It becomes a new
  row of the map;
- **Add a component**: the page writes the command that adds a method, a dataset or sets
  the focus, and the code to start it. **Export** gives Markdown or JSON.

The page writes code and settings; it never sends anything to a machine.

**An empty Projects tab** says why, machine by machine, with the fix to run there: an
older useml (projects need 1.3, curves read from tqdm and printed lines need 1.4), the
group `projects` not shared, or experiences that belong to no declared project
(`useml project init` in the repository they run from).

## Where the history is kept

A record never holds code: names (`module:function`), numbers, curves (at most 120 points
each), configs (YAML), and texts masked like everything the agent sends (paths, host,
user). It is sent only when the group `projects` is shared (`[monitor] share`, on by
default).

- **Hugging Face Space**: in a private dataset, `<you>/useml-monitor-history`, created by
  `useml monitor deploy`. The Space writes there with its own token, which you create
  once:
  1. at https://huggingface.co/settings/tokens, a **Fine-grained** token with "Write
     access to contents" on that dataset only;
  2. in the Space's settings (Variables and secrets), a **secret** named
     `USEML_HISTORY_TOKEN` holding it.

  The agents keep their read-only tokens. Without this secret, the dashboard keeps runs
  in memory until it restarts, and says "history not kept".
- **`useml monitor serve`**: in `<useml home>/monitor/history/` on that machine.

An agent marks a run as kept only when the dashboard answers that it stored it; it never
sends the same run twice.
