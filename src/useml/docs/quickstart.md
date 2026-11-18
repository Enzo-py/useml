# Quickstart

## 1. Install

```bash
pip install "useml[all]"
```

`[all]` adds the optional download backends (`huggingface_hub`, `gdown`) and `psutil` for
the external network monitor. The core needs only numpy, tqdm and PyYAML. Python 3.10 or
newer.

## 2. Set up the machine and the project

From the root of your project:

```bash
useml init                      # shows what it would write; asks before writing
useml init --root /data/me/useml --yes    # on a server: put every cache on a big disk
useml check                     # checks that everything resolves
```

`useml init` only adds what is missing and keeps a backup of every file it changes. See
[adopt.md](adopt.md) for what it looks at.

## 3. Import useml first

```python
import useml          # before torch, transformers, open_clip...
```

Importing useml points every library cache under the machine's root (if one is set) and
makes downloads sequential. Nothing else happens at import.

## 4. A first experiment

Step functions must live in a module (a `.py` file), not in a notebook cell, because the
experience runs in its own process:

```python
# my_project/steps.py
import numpy as np

def make_numbers(n: int):
    return np.arange(n, dtype=np.float32)

def summarize(values, label: str = "run"):
    return {"label": label, "mean": float(values.mean()), "count": int(len(values))}
```

```python
# in a notebook or a script
from useml import exp
from my_project import steps

with exp.experience("demo/first") as e:
    numbers = e.step(steps.make_numbers, 1000, name="numbers")
    e.step(steps.summarize, numbers, label="first try", name="summary")
# When the block ends, the experience starts in the background and a progress bar
# follows it. Closing the notebook does not stop it.

exp.get("demo/first/summary")       # {'label': 'first try', 'mean': 499.5, 'count': 1000}
```

Run the same block again: nothing is recomputed. Change `label`: only `summary` runs again.

## 5. Where to go next

- Process a dataset once and reuse it everywhere: [data.md](data.md).
- Describe models in YAML and load them in one line: [config.md](config.md).
- Everything about experiences (GPUs, checkpoints, bundles): [exp.md](exp.md).
