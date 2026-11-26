# The monitor: your experiments from anywhere

A private dashboard, hosted in a Hugging Face Space, fed by a small agent on each machine.
Nothing to open on the machines: no port, no firewall rule, no VPN.

```text
your phone / laptop ── HTTPS ──▶ private Space ◀── HTTPS (outgoing) ── useml agent (each machine)
```

## How it works, in four terms

- The **dashboard** is a web page served by a private Space. Hugging Face only lets you (and
  your organisation) open it.
- The **agent** is one process per machine. It keeps asking the dashboard "anything for me?"
  through an outgoing connection, like a browser would.
- A **snapshot** is what the agent sends: the state of the machine's experiences and
  resources, filtered by your settings.
- **Refresh**: the page asks every agent for a snapshot now. The agent is already
  waiting, so the answer comes back in about a second plus the network time.

The page refreshes **by itself every 10 s** (5 s to 1 min, or off: the menu next to the
button). These automatic refreshes ask for a light snapshot: status, progress, steps,
curves, log and machine resources, without results, figures, configs and setup. The
**Refresh** button (or the `R` key) asks for everything. The page pauses while its tab
is hidden. Without any page open, the agent still sends a light snapshot when something
changes (at most every 10 s) and a heartbeat every 60 s. A machine that stops talking
for 3 minutes is shown as **no signal**: you see it even if the whole machine died.

## Reading the dashboard

- **The counters** at the top: running, need attention (failed, died, or no signal),
  done, machines online. A click filters the list below. The browser tab shows the same
  (`▶ 2 · ⚠ 1`).
- **In progress**: one card per running experience, with the current step, its progress
  (`i / n`, the label your code gives), the last values of its curves, and the last log
  line. Failures of the last 24 h and **pinned** experiences (☆) stay there too.
- **Machines**: GPU use and memory, RAM, disk, load. Bars turn orange above 75 % and red
  above 90 %.
- **All experiences**: filters, and a search on names, steps and machines (`/` to type).
- A click on an experience opens its panel: why it stopped (the step is marked), steps,
  curves (hover for values), results and figures, configs, log. Its address
  (`#/exp/<machine>/<name>`) can be bookmarked or sent to your phone.

**Naming a machine.** The pencil next to a machine renames it in this browser at once,
and gives the command that names it everywhere:

```bash
useml settings set monitor.label 'GPU server'
```

`label` is only the name shown; `machine` stays the machine's identifier. A machine that
was shut down for good can be removed with **Forget** (it only leaves the dashboard's
memory; a running agent comes back).

## Set up

```bash
pip install -U "useml[monitor]"
hf auth login                      # once per machine (a write token to create the Space)
useml monitor snapshot             # read what would leave this machine
useml monitor deploy               # creates <you>/useml-monitor, PRIVATE
useml monitor start                # the agent, in tmux or detached; survives logout
```

`deploy` refuses to upload into an existing Space that is public. It records the Space in
the machine config (`[monitor] space`). It also creates the private dataset where the
dashboard keeps the history of projects, and prints the two steps that let the Space write
there (a token you create, a secret you add); `--no-history` skips it. See
[projects.md](projects.md). On other machines, add the same line, log in to
Hugging Face, and run `useml monitor start`.

```bash
useml monitor status               # agent running? which dashboard? what is shared?
useml monitor stop
```

The agent's log is `<useml home>/logs/monitor.log`.

## Choose what leaves the machine

```toml
# ~/.config/useml/config.toml
[monitor]
space = "<you>/useml-monitor"
machine = "gpu-a"                  # identifier; default: a hash, never the hostname
label = "GPU server"               # the name shown (any text); default: the identifier
share = ["status", "steps", "progress", "errors", "log", "metrics",
         "results", "figures", "configs", "data", "machine", "setup", "projects"]
experiences = ["*"]                # e.g. ["vision/*"] to report only some
roots = []                         # experiences folders (default: the current project's)
log_lines = 30
redact = []                        # more strings to mask, e.g. a project name
heartbeat = 60
min_interval = 10
```

| Group | What it sends |
|---|---|
| `status` | name, status, steps done / total, current step, last update (always on) |
| `steps` | each step: name, function as `module:name` (never its code), done, duration, output kind |
| `progress` | the current step's `i / n` and label |
| `errors` | why a run stopped: the cause (signal, OOM…) and the end of the Python error |
| `log` | the last `log_lines` lines of `run.log` |
| `metrics` | curves: checkpoint metrics, `ctx.metric(...)` values, and what tqdm bars and printed lines show (see below) |
| `results` | JSON outputs of finished steps (up to 64 KB each) |
| `figures` | PNG figures of finished steps (up to 6, 400 KB each) |
| `configs` | configs the steps read: name, hash, overrides, YAML |
| `data` | names of the datasets and processed data the steps read |
| `machine` | GPUs (name, use, memory), RAM, free disk, load |
| `projects` | the project and method of each experience and its factors; when a run ends, its record (factors, numbers, curves, configs), kept in the dashboard's private dataset — see [projects.md](projects.md) |
| `setup` | useml itself: version, settings (masked; `exp.notify` shown as `<set>`), the checks of `useml check` with their fixes, store size and budget, processed tables, datasets, downloads, caches left in the home, download policy — sent on Refresh only |

**Never sent**, whatever the settings: the code snapshots, the installed packages,
environment variables, tokens. There is no group for them.

**Masked in every text** (log, errors, configs, results): the home folder becomes `~`,
project folders `<project>`, useml and data folders `<useml>` / `<data>`, the hostname
`<host>`, your user name `<user>`, and every string of `redact` `<redacted>`.

`useml monitor snapshot` prints the exact JSON that would be sent. The log is the field to
remove first on a sensitive project: it holds whatever your code prints.

## useml setup, and changing a setting

Each machine card has a **useml setup** button, which also says the machine's health
(`✓ setup OK`, `! 2 to look at`, `✕ 1 to fix`). It opens:

- **Health**: the result of `useml check` on that machine — config keys, folders (writable,
  free space, not left in the home), caches, store budget, datasets, experiences that
  died, tmux — each problem with the command that fixes it, and a copy button;
- **State**: the version, the store and its budget, processed tables, datasets,
  downloads, caches left in the home folder, the download policy.

Below, the **settings** of the machine. Click one (or type a key), give a new value: the
page writes the command to run **on the machine**, with a copy button:

```bash
useml settings set monitor.log_lines 50
```

The dashboard never changes anything by itself: it only composes the command. Nothing is
sent to the machine.

## Projects

The **Projects** tab groups runs by project and method, keeps them after machines are
off, and shows versions, an ablation map, correlations and curves side by side:
[projects.md](projects.md).

## Curves while a step runs

Nothing to add when the step already shows its numbers: the postfix of a tqdm bar
(`bar.set_postfix(loss=loss)`) and printed lines such as `epoch=3 loss=0.23` or
`Epoch 3/10 - loss: 0.2310 - accuracy: 0.9120` become curves, and the outermost tqdm bar
is the progress ([exp.md](exp.md), "Curves and progress without ctx"). To choose exactly
what is drawn:

```python
def train(dataset, model, epochs: int, *, ctx):
    for epoch in range(epochs):
        ...
        ctx.metric(epoch, loss=loss, val_accuracy=accuracy)   # only these are drawn
```

Runs started before useml 1.4 have no curves unless they called `ctx.metric`: the
numbers were never written down.

## Security

- **Who can open the dashboard**: Hugging Face checks it — the Space is private.
- **Who can feed it**: agents send their Hugging Face token; the dashboard checks once
  with Hugging Face that it belongs to the Space's owner (or the owner's organisation),
  then keeps only a hash of it in memory for 10 minutes.
- **Changing a setting**: the page shows a command; you run it on the machine. No action
  travels from the dashboard to a machine.
- **What the page keeps**: names given with the pencil, pins and the refresh interval stay
  in your browser (local storage); nothing about them is sent anywhere.
- **What the dashboard can make an agent do**: only send a snapshot (full or light). There is no command
  to run, stop or change anything: even someone controlling the Space could not act on
  your machines.
- **What is stored**: the live state is not: it lives in the Space's memory, and agents
  send it again within a minute after a restart. The records of finished runs of a
  project are, in a private dataset that only the Space writes to, with a token limited
  to it ([projects.md](projects.md)).
- **Downloads policy**: the agent is its own process with a single connection; it never
  competes with an experiment's downloads.

## Without Hugging Face

```bash
useml monitor serve                # the same dashboard on 127.0.0.1, with this machine's agent
ssh -L 8765:127.0.0.1:8765 <server>   # then open the printed URL on your laptop
```

`serve` prints a URL with a one-time token; nothing leaves the machine.
