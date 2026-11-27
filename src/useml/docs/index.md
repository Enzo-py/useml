# useml documentation

useml is a toolkit for machine-learning projects that run on big machines. It solves
nine problems that grow with every project:

| Problem | useml answer | Guide |
|---|---|---|
| Caches (Hugging Face, torch, pip…) fill the home folder and spread everywhere | one cache root per machine | [paths-and-hub.md](paths-and-hub.md) |
| Downloads go through many libraries, some in parallel | one gate, sequential by default | [paths-and-hub.md](paths-and-hub.md) |
| Every dataset has its own download and parsing code | drivers: declared once, loaded by name | [datasets.md](datasets.md) |
| Results are recomputed because nothing remembers them | a key → value cache with a disk budget | [store.md](store.md) |
| Datasets are re-embedded inside every experiment | data processed once, loaded by name | [data.md](data.md) |
| Checking runs means logging into each server | a private dashboard, reachable from anywhere | [monitor.md](monitor.md) |
| Nobody remembers which attempt gave which number | projects: methods, datasets, every run kept with its factors and results | [projects.md](projects.md) |
| Explaining the context again to an AI assistant | `useml info` and an MCP server: every run and project in few tokens | [info.md](info.md) |
| Models are hard to rebuild, experiments die silently | YAML configs and supervised experiments | [config.md](config.md), [exp.md](exp.md) |

Each module works on its own: a project can use only `useml.hub`, or only `useml.exp`.

## Start here

1. [quickstart.md](quickstart.md) — install, set up a project, run a first experiment.
2. [adopt.md](adopt.md) — bring useml into an existing project without losing anything.
3. [troubleshooting.md](troubleshooting.md) — a run crashed, hangs, or will not start.

## Reference

- [reference.md](reference.md) — every setting, environment variable and command.
- [faq.md](faq.md) — short answers to common questions.
- [SKILL.md](SKILL.md) — instructions for an AI coding assistant working with useml.

## Read the docs from a terminal

The documentation is installed with the package:

```bash
useml docs            # list the guides and where they are
useml docs data       # print one guide
useml docs --path     # the folder holding them
```
