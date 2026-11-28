"""The architecture of a model a step returns: its blocks, as a tree.

When a step's output is a PyTorch module (or holds some, in a dict, list or tuple, or as
attributes of a plain object), the run writes `steps/<step>/architecture.json`:

    {"model": {"class": "mypkg.nets.Encoder", "params": 1234567, "trainable": 1234567,
               "extra": "", "children": [
                   {"name": "embed", "class": "torch.nn.Embedding", "params": 65536,
                    "extra": "1024, 64", "children": []},
                   {"name": "layers", "class": "torch.nn.ModuleList", "repeat": 12, ...}]}}

- `extra` is what PyTorch prints for the block (`in_features=256, out_features=10`);
- children that follow each other with the same structure are one entry with `repeat`
  (`layers.0` … `layers.11` → `layers` ×12);
- depth and size are bounded (MAX_DEPTH levels, MAX_CHILDREN per block): a tree to read,
  not a dump.

Nothing is imported: without `torch` already loaded by the step, nothing is written. The
weights are never read — only class names, shapes and counts. Class names are code names:
they go to the dashboard with the run's record (`[monitor] share` group "projects").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_DEPTH = 6
MAX_CHILDREN = 40
MAX_BYTES = 256 * 1024


def _modules_in(value, name: str = "model", depth: int = 0):
    """(name, module) for the modules in a step's output."""
    torch = sys.modules.get("torch")
    if torch is None or depth > 2:
        return
    if isinstance(value, torch.nn.Module):
        yield name, value
    elif isinstance(value, dict):
        for key, item in list(value.items())[:20]:
            yield from _modules_in(item, str(key), depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:20]):
            yield from _modules_in(item, f"{name}[{index}]", depth + 1)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for key, item in list(vars(value).items())[:20]:
            if not key.startswith("_"):
                yield from _modules_in(item, key, depth + 1)


def _class(module) -> str:
    kind = type(module)
    return f"{kind.__module__}.{kind.__qualname__}"


def _signature(node: dict) -> str:
    """What makes two blocks the same, to fold repeats."""
    return json.dumps({k: v for k, v in node.items() if k != "name"}, sort_keys=True)


def describe(module, depth: int = 0) -> dict:
    """The block tree of one module."""
    params = trainable = 0
    for parameter in module.parameters():
        count = parameter.numel()
        params += count
        if parameter.requires_grad:
            trainable += count
    try:
        extra = module.extra_repr()
    except Exception:                                        # noqa: BLE001 - user code
        extra = ""
    node = {"class": _class(module), "params": params, "trainable": trainable,
            "extra": extra[:160]}
    children = []
    if depth < MAX_DEPTH:
        for child_name, child in list(module.named_children())[:MAX_CHILDREN]:
            children.append({"name": child_name, **describe(child, depth + 1)})
        hidden = len(list(module.named_children())) - len(children)
        if hidden > 0:
            children.append({"name": f"+{hidden} more", "class": "", "params": 0,
                             "trainable": 0, "extra": "", "children": []})
    node["children"] = _fold(children)
    return node


def _fold(children: list) -> list:
    """Neighbours with the same structure are one entry with `repeat`: `0` … `11` (a
    ModuleList) → `0–11` ×12; `block1`, `block2` → `block1…block2` ×2."""
    out: list = []
    for child in children:
        previous = out[-1] if out else None
        if previous is not None and _signature(previous["one"]) == _signature(child):
            previous["count"] += 1
            previous["last"] = child["name"]
        else:
            out.append({"one": child, "count": 1, "last": child["name"]})
    folded = []
    for group in out:
        node = dict(group["one"])
        if group["count"] > 1:
            first, last = node["name"], group["last"]
            node["name"] = f"{first}–{last}" if first.isdigit() and last.isdigit() else f"{first}…{last}"
            node["repeat"] = group["count"]
        folded.append(node)
    return folded


def capture(value, folder: Path) -> Path | None:
    """Write `architecture.json` in `folder` for the modules of `value`; None if none."""
    try:
        found = {name: describe(module) for name, module in _modules_in(value)}
    except Exception:                                        # noqa: BLE001 - never fail a step
        return None
    if not found:
        return None
    text = json.dumps(found)
    if len(text) > MAX_BYTES:                            # a huge tree: keep the top levels
        for tree in found.values():
            _prune(tree, 3)
        text = json.dumps(found)
    path = Path(folder) / "architecture.json"
    path.write_text(text)
    return path


def _prune(node: dict, depth: int) -> None:
    if depth <= 0:
        node["children"] = []
        return
    for child in node.get("children", []):
        _prune(child, depth - 1)


def read(folder: Path) -> dict | None:
    path = Path(folder) / "architecture.json"
    if not path.exists() or path.stat().st_size > MAX_BYTES * 2:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
