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
- **Refresh**: the button asks every agent for a snapshot now. The agent is already
  waiting, so the answer comes back in about a second plus the network time.

Without a click, the agent sends a light snapshot when something changes (at most every
10 s) and a heartbeat every 60 s. Ticking **auto** refreshes every 15 s. A machine that
stops talking for 3 minutes is shown as **silent**: you see it even if the whole machine
died.

## Set up

```bash
pip install -U "useml[monitor]"
hf auth login                      # once per machine (a write token to create the Space)
useml monitor snapshot             # read what would leave this machine
useml monitor deploy               # creates <you>/useml-monitor, PRIVATE
useml monitor start                # the agent, in tmux or detached; survives logout
```

`deploy` refuses to upload into an existing Space that is public. It records the Space in
the machine config (`[monitor] space`). On other machines, add the same line, log in to
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
machine = "gpu-a"                  # the name shown; default: a hash, never the hostname
share = ["status", "steps", "progress", "errors", "log", "metrics",
         "results", "figures", "configs", "data", "machine", "setup"]
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
| `metrics` | curves: checkpoint metrics and `ctx.metric(...)` values |
| `results` | JSON outputs of finished steps (up to 64 KB each) |
| `figures` | PNG figures of finished steps (up to 6, 400 KB each) |
| `configs` | configs the steps read: name, hash, overrides, YAML |
| `data` | names of the datasets and processed data the steps read |
| `machine` | GPUs (name, use, memory), RAM, free disk, load |
| `setup` | useml itself: version, settings (masked; `exp.notify` shown as `<set>`), store size and budget, processed tables, datasets, downloads, caches left in the home, download policy, suggested commands — sent on Refresh only |

**Never sent**, whatever the settings: the code snapshots, the installed packages,
environment variables, tokens. There is no group for them.

**Masked in every text** (log, errors, configs, results): the home folder becomes `~`,
project folders `<project>`, useml and data folders `<useml>` / `<data>`, the hostname
`<host>`, your user name `<user>`, and every string of `redact` `<redacted>`.

`useml monitor snapshot` prints the exact JSON that would be sent. The log is the field to
remove first on a sensitive project: it holds whatever your code prints.

## useml setup, and changing a setting

Each machine card has a **useml setup** button: the version, the store and its budget,
processed tables, datasets, downloads, caches left in the home folder, the download
policy, and suggested commands (for example `useml hub scan --move --apply` when caches
are still in the home).

Below, the **settings** of the machine. Click one (or type a key), give a new value: the
page writes the command to run **on the machine**, with a copy button:

```bash
useml settings set monitor.log_lines 50
```

The dashboard never changes anything by itself: it only composes the command. Nothing is
sent to the machine.

## Curves while a step runs

```python
def train(dataset, model, epochs: int, *, ctx):
    for epoch in range(epochs):
        ...
        ctx.metric(epoch, loss=loss, val_accuracy=accuracy)   # drawn by the dashboard
```

## Security

- **Who can open the dashboard**: Hugging Face checks it — the Space is private.
- **Who can feed it**: agents send their Hugging Face token; the dashboard checks once
  with Hugging Face that it belongs to the Space's owner (or the owner's organisation),
  then keeps only a hash of it in memory for 10 minutes.
- **Changing a setting**: the page shows a command; you run it on the machine. No action
  travels from the dashboard to a machine.
- **What the dashboard can make an agent do**: only send a snapshot. There is no command
  to run, stop or change anything: even someone controlling the Space could not act on
  your machines.
- **What is stored**: nothing on disk. Snapshots live in the Space's memory; after a
  restart, agents send them again within a minute.
- **Downloads policy**: the agent is its own process with a single connection; it never
  competes with an experiment's downloads.

## Without Hugging Face

```bash
useml monitor serve                # the same dashboard on 127.0.0.1, with this machine's agent
ssh -L 8765:127.0.0.1:8765 <server>   # then open the printed URL on your laptop
```

`serve` prints a URL with a one-time token; nothing leaves the machine.
