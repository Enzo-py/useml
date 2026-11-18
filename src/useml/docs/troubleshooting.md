# Troubleshooting

## A run failed: why?

```bash
useml exp why <name>
```

It prints the cause in words, the step, the peak RAM, the Python traceback and the last
log lines. The same text is in `experiences/<name>/crash.txt`.

| What `why` says | What happened | What to do |
|---|---|---|
| `SIGKILL` … OOM killer | the machine ran out of RAM; the kernel killed the run | read less at once: `feats.batches(...)` instead of `feats.numpy()`, smaller batches, `dtype="int8"` tables |
| `SIGSEGV` / `SIGABRT` | native code crashed (CUDA, a C extension); the Python stack is in `run.log` just above | look at the last Python frame of the stack; often a CUDA or library version mismatch |
| `SIGTERM` | stopped by `useml exp stop --now`, a scheduler, or a shutdown | resume with `useml exp run <name>` |
| GPU ran out of memory | `torch.cuda.OutOfMemoryError` | lower the batch size, or pick a free card (`gpu=`) |
| a Python exception | the traceback is shown | fix it, run again: finished steps are skipped |
| `died` … disappeared without recording anything | the run **and** its supervisor vanished: reboot, logout kill, an admin or scheduler | see "Runs die when I disconnect" below |

Run again after fixing: `useml exp run <name>`. Steps already done are not recomputed.

## A run seems stuck

```bash
useml exp stack <name>
```

It prints the Python stack of every thread of the running process, without stopping it.
`useml exp why <name>` also shows when the log was last written.

## Runs die when I disconnect

`useml check` tells you if the machine kills a user's processes at logout (systemd
`KillUserProcesses=yes` without lingering). tmux cannot help there. Ask once for
lingering:

```bash
loginctl enable-linger $USER
```

Without tmux, runs still survive a disconnection; you just cannot reattach a terminal
(`useml exp follow <name>` shows the log instead).

## `follow` shows nothing new

`follow` reads `run.log` and `state.json`. If the log does not move, use
`useml exp stack <name>` to see where the run is.

## "is defined in a notebook cell"

The run happens in another process, which must import the step function. Move it into a
`.py` module of the project and import it from there.

## "cannot be rebuilt by the experience"

The argument is an object built by hand. Load it with `config.load(name, **overrides)`,
pass a processed dataset with `data.load(name)`, or pass plain values.

## "was modified after loading"

An object from `config.load` was changed by hand (`model.lr = 0.1`). The run would rebuild
it from the file and lose the change. Pass it as an override:
`config.load("resnet_ft", optimizer={"lr": 0.1})`.

## "unknown type" / "does not take"

The YAML names a type that the config packages do not export, or a parameter the class
does not accept. The message gives the closest names. Add the package to
`[tool.useml.config] packages`, or export the class from the package's `__init__.py`.
`useml check` lists every unresolved type at once.

## "refused connection … parallel downloading"

A second thread tried to download while another one was. Either the code downloads in
parallel on purpose (then allow it: `useml.hub.allow_parallel()`), or a library opened a
thread pool. The full stack is in `<root>/logs/network_guard.log`.

## The disk fills up

```bash
useml store info                 # store size, budget, free disk
useml data tables                # processed tables and who uses them
useml hub ls                     # downloads
useml hub scan                   # caches left in the home folder
```

Remove a processed dataset and its table: `useml data rm <name> --table`. Clean the store
down to a budget: `useml store gc --max-size 300GB --apply`.
