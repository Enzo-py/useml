# Experiences: `useml.exp`

Declare an experiment as a script; run it in the background; read its outputs in one line.

## Declare

```python
from useml import config, data, exp
from my_project import steps

with exp.experience("vision/baseline") as e:
    model = e.step(steps.train, data.load("photos-clip"), config.load("classifier"),
                   epochs=10, name="model")
    e.step(steps.evaluate, model, name="report")
```

**The block runs nothing.** Each `e.step(fn, *args, **kwargs)` records a call and returns
a reference. When the block ends, the declaration is written, then run.

`exp.experience(name, note="", run=True, detach=True, follow=True, gpu=None)`:

| Option | Effect |
|---|---|
| `run=False` | only write the declaration; `e.run()` later |
| `detach=False` | run in this process (debugging) |
| `follow=False` | do not show the progress bar |
| `gpu=1` or `gpu=[0, 1]` | sets `CUDA_VISIBLE_DEVICES` for the run |

## What a step can receive

A step function must be **importable**: defined in a module, not in a notebook cell or
inside another function. Its arguments must be **describable**, because the run happens
in another process:

| Argument | Rebuilt in the run as |
|---|---|
| plain values, lists, dicts, `Path` | themselves |
| the reference returned by an earlier `e.step` | that step's output |
| an object from `config.load(name, **overrides)` | the same config, built again |
| `data.load(name)` | the same processed dataset |
| `datasets.load(name, split)` (and `.field(...)`) | the same dataset and split |
| an output of another experience (`exp.get(...)`) | that output, loaded from its folder |

Anything else raises **on the line that declares it**, with what to do instead. An object
loaded by `config.load` and then modified by hand is refused too: pass the change as an
override.

## Runs that survive

Detached runs start in a **tmux** session `useml-<name>` when tmux exists, otherwise in
their own session. Closing the laptop, the SSH connection or the notebook does not stop
them.

A small **supervisor** process is the parent of the run. When the run ends, it records
how: exit code or signal, peak RAM, kernel OOM messages, Python traceback, last log
lines. See [troubleshooting.md](troubleshooting.md).

```bash
useml exp ls
useml exp show vision/baseline       # steps: done, duration, output kind
useml exp follow vision/baseline     # progress bar and log
useml exp attach vision/baseline     # the tmux terminal
useml exp why vision/baseline        # where it is, or why it died
useml exp stop vision/baseline       # after the current step
useml exp stop vision/baseline --now
```

Get told when a run ends or crashes:

```toml
# ~/.config/useml/config.toml
[exp]
notify = "my-notify-command $USEML_EXP_NAME $USEML_EXP_STATUS"
```

The command receives `USEML_EXP_NAME`, `USEML_EXP_STATUS` and `USEML_EXP_MESSAGE`.

## Resuming and changing

- Each step has a **fingerprint**: its function, its arguments, and the fingerprints of
  the steps it reads. Running the block again **skips** steps already done.
- **Changing a line reruns only the steps that depend on it.**
- A step whose declaration changed starts from scratch; its former folder is set aside as
  `steps/<name>.stale-<date>`, never deleted.
- Editing a step's code does **not** rerun it (that would waste hours for a renamed
  variable); a warning says the code changed.
- A run that failed or was stopped resumes with `e.run()` (or `useml exp run <name>`).

## Read the outputs

```python
model = exp.get("vision/baseline/model")       # or exp.get("vision/baseline", "model")
exp.open("vision/baseline").results()          # every JSON output, no model loaded
exp.open("vision/baseline").path("model")      # the step's folder
```

What a step returns is saved by type:

| Returned value | Saved as | `get` returns |
|---|---|---|
| a torch module built by `config.load` | its config + `weights.pt` | the module, rebuilt and with its weights |
| an object with `save(dir)` and a `load(dir)` classmethod | its own format | `Class.load(dir)` |
| another torch module | `module.pt` | the module |
| a numpy array or dict of arrays | `output.npz` | arrays |
| a matplotlib figure (also inside dicts) | `.png` | the path of the image |
| JSON-able values | `output.json` | the values |
| anything else | `output.pkl` (with a warning) | the object |

`exp.register_saver(kind, predicate, save, load)` teaches it another type.

## Inside a step: `ctx`

A step function with a parameter named `ctx` receives a context:

```python
def train(dataset, model, epochs: int, *, ctx):
    start = ctx.checkpoints.latest_step() or 0
    for epoch in range(start, epochs):
        ...
        ctx.progress(epoch + 1, epochs, "training")          # shown by follow
        ctx.checkpoints.save({"model": model.state_dict()}, step=epoch + 1,
                             metrics={"val_loss": val_loss})
        if ctx.should_stop():                                # `useml exp stop`
            break
    return model
```

- `ctx.dir`: the step's folder, kept after an interruption;
- `ctx.log(message)`;
- `ctx.metric(step, **values)`: values drawn as curves by the monitor (optional: see
  below);
- `ctx.checkpoints`: `save`, `latest`, `latest_step`, `best(metric, mode="min")`; keeps
  the last 3 and the best of each metric; writes atomically.

## Curves and progress without `ctx`

Code that never calls `ctx` still gets live curves and progress. While a step runs, useml
reads what it already shows:

- a **tqdm bar**: the outermost bar is the step's progress (`n / total`, its
  description); the numbers given to `bar.set_postfix(loss=…, acc=…)` become points;
- **printed lines** holding `name=number` pairs (`epoch=3 loss=0.23`), or two or more
  `name: number` pairs (`Epoch 3/10 - loss: 0.2310 - accuracy: 0.9120`) become points.
  `epoch`, `step`, `iter`, `iteration` or `global_step` on the line is the x of the point;
  a single `seed: 42` is not taken.

A step that calls `ctx.metric` or `ctx.progress` itself keeps only what it reports.
`USEML_NO_AUTO_METRICS=1` turns the reading off. A run in this process (`detach=False`, a
notebook) also writes what it prints to `run.log`, so the dashboard and
`useml exp log` show it.

## On disk

```text
experiences/vision/baseline/
    experience.json       the declaration and fingerprints
    state.json            status, step, progress, pids, exit
    run.log               everything the run printed
    crash.txt             why it failed (if it did)
    code/<date>/          the code of each launch: commit, uncommitted diff, untracked
                          sources, installed packages
    steps/<name>/         what each step wrote, done.json, its output
```

## Bundles: archive the version that worked

```bash
useml exp bundle vision/baseline -o archive/baseline-final
useml exp bundle vision/baseline --archive          # a .tar.gz
useml exp verify archive/baseline-final             # every file against its checksum
```

A bundle is a standalone folder:

- `code/`: the project code **exactly as it ran** (commit + uncommitted changes +
  untracked sources);
- `useml/`: a copy of useml, so nothing has to be installed from PyPI;
- `experiences/`: the declaration and every step's output (checkpoints with
  `--checkpoints`);
- `data/`: the rows of the processed datasets the steps read (`--data auto` copies them
  up to 2 GB; `rows` always; `none` never);
- `environment/packages.txt`: every installed package and its version;
- `manifest.json`: sha256 of every file.

```python
from load import get           # from inside the bundle folder
model = get("model")
```

```bash
python run.py --from-scratch   # recompute every step from the bundle
```
