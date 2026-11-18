# The cache store: `useml.store`

A key → value cache shared by every process of the machine: an SQLite index, large values
in files.

```python
from useml import store

cache = store.open()                                   # <root>/store by default
scores = cache.declare("results/scores", cost="precious")

key = {"model": "resnet_ft", "dataset": "cifar10", "metric": "accuracy"}
if (value := scores.get(key)) is None:
    value = {"accuracy": 0.91}                         # the expensive computation
    scores.put(key, value)
```

## Keys

A key is a dict, a list or a string. It is written as canonical JSON (same fields, same
text, whatever their order), then hashed. The plain key is kept in the index, so
`entries()` can show it.

## Spaces and costs

Every entry belongs to a declared **space**, named `<family>/<name>`
(`results/scores`, `processing/masks`). Writing to an undeclared space raises.

The **cost** says what an entry costs to recompute, and so what the cleaner removes first:

| Cost | Meaning | Removed by `gc` |
|---|---|---|
| `cheap` | seconds to recompute | first |
| `expensive` | minutes or hours | next |
| `precious` | must not be lost (reported numbers) | only with `--include-precious` |

The declaration is stored in the index, so the cleaner works without importing your code.

## Values

| Value | Stored as |
|---|---|
| JSON under 64 KB | inside the index |
| larger JSON | a `.json` file |
| bytes | a `.bin` file |
| a numpy array, or a dict of arrays | `.npz` — compressed for integers and booleans, raw for floats |
| your own format | `put_file(space, key, write)`: `write(path)` fills a temporary file |

## Guarantees

- **One write per key**: `put` on an existing key does nothing unless `overwrite=True`.
- **Atomic**: a file is written under a temporary name, then renamed. A crash leaves the
  old value or nothing.
- **Several processes**: SQLite in WAL mode; parallel runs share the index safely.
- **A missing or unreadable file is a miss**, and its entry is removed.

## Pins

```python
scores.pin(key, "my-experiment")        # gc keeps this entry
cache.unpin_owner("my-experiment")
```

## Managed folders

For data too big for one index row per item (memory-mapped shards…), a space can own a
folder: `cache.managed("processing/features")` returns `<store>/managed/<space>/`. It is
counted in sizes and moved with the store.

## Cleaning and maintenance

```toml
# ~/.config/useml/config.toml
[store]
max_size = "450GB"          # the budget `gc` aims for
```

```bash
useml store info                          # size, budget, free disk
useml store spaces                        # per space: entries, size, cost, last access
useml store gc                            # dry run down to the budget
useml store gc --max-size 300GB --apply   # really delete
useml store gc --older-than 30 --apply    # entries unread for 30 days
useml store check --repair                # missing files, orphans, leftovers
useml store move /new/disk/store          # copy, verify, then switch
```

Nothing is ever deleted automatically.
