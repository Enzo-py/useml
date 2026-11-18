# Configs: `useml.config`

Describe objects in YAML; build them by name in one line.

## Set up the library

In the project's `pyproject.toml` (or `[config]` in a `useml.toml`):

```toml
[tool.useml.config]
root = "configs"                              # relative to this file
packages = ["mypkg.models", "mypkg.metrics"]  # where `type:` names are looked up
kinds = {model = "models", metric = "metrics"}  # default: every sub-folder is a kind
default_kind = "model"
pythonpath = ["."]                            # added to sys.path (packages not installed)
```

`useml init` writes this for you. Without settings, a `configs/` folder above the current
directory is used, with no type package (register classes with `@config.register`).

## A config file

```yaml
# configs/models/resnet_ft.yaml
meta:
  display_name: "ResNet-50 (fine-tuned)"
  short_name: "ResNet-FT"
type: Classifier
backbone: {ref: resnet50}             # another config, built once and shared
head: {type: LinearHead, dropout: 0.1}
optimizer: {type: SGD, lr: 0.01}
```

- A node is a value, a list, a mapping, `{type: Name, ...}` (built), or
  `{ref: other_config, ...overrides}`.
- **What is left out keeps its default.**
- **What is misspelled raises**, with the closest name and the place in the file:
  `head.LinearHead does not take 'dropuot' (close to dropout)`.
- `type:` names come only from the declared packages and registered classes: a YAML file
  can never import arbitrary code.
- `meta` accepts `display_name`, `short_name`, `note`, `cite`, `tags`.

## Load

```python
from useml import config

model = config.load("resnet_ft")                               # memoised by name
model = config.load("resnet_ft", reload=True)                  # rebuild after editing
model = config.load("resnet_ft", optimizer={"lr": 0.1})        # one setting, merged
model = config.load("resnet_ft", overrides={"optimizer.lr": 0.1})   # dotted path
model = config.load("resnet_ft", head={"type": "MLPHead", "layers": 2})  # node replaced
scorer = config.load("accuracy", model="resnet_ft")            # fills the object slot
```

- **Memoised**: two loads of the same name return the same object. With overrides there is
  no memo.
- **Merge rule**: a dict without `type:`/`ref:` merges into the node; anything else
  replaces it.
- **The object slot**: `model=` fills the parameter that holds an object, whatever its
  name (`encoder`, `classifier`…). It is inferred: the first positional parameter declared
  as an object in the file, annotated with a non-scalar type, or defaulting to `None`
  (`device`, `dtype` excluded). A string is a config name.

## Cite what ran

```python
model.config.hash          # 8 hex characters; ignores formatting and `meta`
model.config.cited_name    # "resnet_ft", or "resnet_ft+ovr" with overrides
model.config.label         # "ResNet-FT [lr=0.1]"
model.config.provenance()  # "ResNet-50 (fine-tuned)  cfg:1a2b3c4d  configs/models/resnet_ft.yaml  code:abc1234"
config.find("1a2b3c4d")    # git versions of the configs that produce this hash
```

Store the provenance line **next to** the numbers it describes.

## From an object back to a file

```python
config.create("small", MyModel(hidden=64), meta={"display_name": "Small model"})
print(config.get_config(model))          # the YAML of a live object
config.diff("resnet_ft", model)          # {"optimizer.lr": (0.01, 0.1)}
```

Arguments are read back from the object's attributes (`self.hidden` or `self._hidden`).
`create` rebuilds the file and checks it gives the same object, or removes it and raises.

```bash
useml config ls
useml config describe resnet_ft
useml config show resnet_ft optimizer.lr=0.1     # resolved YAML and hash
useml config find 1a2b3c4d
```
