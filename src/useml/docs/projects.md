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
- for a method, **its evolution**: every run in order with its main result, the line of
  the best so far, and the list of each new best — by how much it beat the previous one,
  after how many other attempts, what it changed (factors `old → new`, the steps whose
  code changed) and its note. On a method's page, experience names are shown without the
  method (`wide/v2/w8` → `v2/w8`);
- for a method, **its versions**: a version is the code of the steps (a hash). Two runs
  of the same version ran the same code and differ only by their factors. The page says
  which step's code changed from one version to the next, and the git commit (`*` when
  there were uncommitted changes);
- **the ablation table**, on a method's page, laid out like a paper's: the components that
  change (✓ / × for yes / no, – when left out), then the scores grouped by benchmark and
  metric; **bold** is the best of a column, underlined the second; the best configuration
  is highlighted; rows come in blocks by their main component. Chips choose the
  benchmarks and metrics shown (by default: the main metric on every benchmark that has
  it). **Copy LaTeX** gives a booktabs table, **Copy Markdown** a Markdown one. A row per run. Runs whose names differ by
  their first part under the method (`v2/…`, `sweep/…`) are **families**, compared
  apart (chips to switch): they rarely share results or settings. The results come first
  (green: better, red: worse, (green: better, red: worse,
  within each column; ↑ or ↓ says which way is better, click to switch), then the factors
  that change: at most 8 columns, those where two runs set different values first (a
  setting present in some runs only is behind "N more factors"), identical columns merged,
  long names shortened (the whole name on hover). The factors that never change are listed
  once below. Choose
  a reference run (the `ref` column) and every other row highlights what it changed;
- **correlations**: for each factor and each result, `r` (from −1 to +1) for numbers —
  green when more of the factor goes with a better result — and `η` (from 0 to 1) for
  choices among names, how much the choice explains the result. A click draws the runs.
  With few runs, read it as a hint, not a proof;
- **benchmarks**: where scores were measured. A score `report.<benchmark>.<metric>` names
  its benchmark in its key; a score `report.metrics.<metric>` takes it from the run's name
  (`baselines/<dataset>/<task>/<model>` → benchmark `<dataset>/<task>`). Each benchmark has
  a **sheet**: one row per model (each experience name) at its best run, its metrics,
  blocks by method — the baselines and the best version of each model side by side, with
  the same exports. A dataset links to the benchmarks named after it;
- **curves** of the runs side by side, the running ones included (thicker, marked
  "live", updated at every refresh): a new attempt is compared with the previous ones while
  it trains. A running run also shows its current values in the ablation map;
- a run's page: results (a report is a table, grouped, with a filter when long), factors,
  steps, configs, curves, and **New alternative**: the
  run's declaration as Python, to edit (one value) and run on a machine. It becomes a new
  row of the map;
- **Add a component**: the page writes the command that adds a method, a dataset or sets
  the focus, and the code to start it. **Export** gives Markdown or JSON.

The page writes code and settings; it never sends anything to a machine.

**An empty Projects tab** says why, machine by machine, with the fix to run there: an
older useml (projects need 1.3, curves read from tqdm and printed lines need 1.4), the
group `projects` not shared, or experiences that belong to no declared project
(`useml project init` in the repository they run from).

## Asking about it in a few words

`useml info project <name>` (and `-m <method>`) gives the same in text, short enough for
an AI assistant: [info.md](info.md). A method's main result can be declared:
`useml project method set wide --result evaluate.accuracy`.

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
sends the same run twice — unless asked: `useml monitor resend`, on a machine, sends all
its runs again and the dashboard replaces the records that changed (for example, after an
update of useml that reads more of each run).

**What a record holds of a report**: every number of the steps' JSON outputs, whatever
their size (up to 64 MB; lists longer than 20 are left out). A curve has a "best" only
when its name says which way is better (`loss`, `error`… lower; `acc`, `f1`… higher);
for other names (`assign`), its first, last, lowest and highest values.
