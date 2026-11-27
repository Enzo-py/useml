# Asking about runs and projects (you, or an AI assistant)

`useml info` answers "what is going on, and what did each attempt give?" in a few lines.
It is written to be read by a person and by an AI assistant at the lowest cost in tokens
(a token is a piece of a word: an assistant pays, and is limited, by the tokens it reads).
So you can tell an assistant "look at the project and propose the next run" without
pasting anything.

## The views

```bash
useml info                              # start here
useml info project robust-vision        # a project
useml info project robust-vision -m wide    # one method: its runs table
useml info run wide/w4-mix              # one finished run (an id prefix works)
useml info run 35a28e5f --code          # + its declaration as Python
useml info exp gpu-a:wide/w4-mix        # an experience as its machine reports it now, + log
useml info machine gpu-a                # resources, and useml's health checks there
```

What each one says:

- **overview** (about 300 tokens): each machine (GPU use, free disk, useml version and
  health), every experience running now (progress, last values of its curves), the
  failures of the last 24 hours with their cause, and the projects (focus, runs kept).
- **project**: its methods, each ranked by its own main result with its best run, the
  runs going on, and the latest runs. Repeats of an identical run are counted (`×3`),
  not listed.
- **project -m <method>**: with several families of runs (`v2/…`, `sweep/…`), the
  most recent one that has results (`--family <name>` for another); the evolution (each new best of the main result, its gain,
  what it changed), then the runs table — results first, then only the factors that
  differ between runs (the ones that never change are listed once, at most 10 columns:
  `--factors N` for more), the strongest links between a factor and a result, and the
  code versions with the steps that changed.
- **run**: factors, results, curves as `first → last (best @x)`, steps, data, configs,
  error. `--code` adds the declaration (to write an alternative), `--configs` the YAML.
- **exp**: status, progress, steps, error, curves and the end of the log, live.

Every view ends with the command that goes further. `--json` gives the same data
unformatted. Numbers have 4 significant digits; times are relative ("3h ago").

## Where the data comes from

From your private dashboard when the machine knows it (`[monitor] space`), with this
machine's Hugging Face token — the same reading the agent does: every machine, every run
kept, from anywhere. `--local` reads this machine only (its experiences, its project's
ended runs), with no dashboard.

Everything is **read only**: no view starts, stops or changes anything.

## The main result of a method

Runs are ranked by one result. Declare it, so that you and the assistant agree:

```bash
useml project method set wide --result evaluate.accuracy
```

Without it, useml guesses — a quality measure (accuracy, f1, auc, loss…), never a count
(`n_test`) or a label (`runner_version`) — and says `[guessed]`.

## For an AI assistant: MCP

MCP (Model Context Protocol) is how an assistant calls tools on your computer. useml is
an MCP server with five read-only tools — `useml_overview`, `useml_project`, `useml_run`,
`useml_experience`, `useml_machine` — returning the texts above.

```bash
claude mcp add useml -- useml mcp       # Claude Code, once
```

Give the full path of `useml` if it is not on the assistant's PATH (`which useml`). Other
assistants take the same command (`useml mcp`, standard input and output) in their MCP
settings. Without MCP, an assistant that can run commands uses `useml info` directly.

What leaves your computer: what the assistant reads is sent to it, like anything you
paste. The views hold what the dashboard holds (masked paths, no code), nothing more.
