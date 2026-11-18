"""What a step returns is SAVED by type, and LOADED back by `exp.get` — in one line.

    a torch module built from a config   config recipe + `weights.pt` (state_dict): loading
                                          rebuilds it from its config and loads the weights
    an object with save(dir) / cls.load(dir)   its own format, under `output/`
    a torch module (no config)           `module.pt` (the whole module, pickled)
    a numpy array / dict of arrays       `output.npz`
    a matplotlib figure                  `output.png` (and a dict of figures: one png each)
    JSON-able values                     `output.json`
    anything else                        `output.pkl`, with a warning in the log

Every loaded object remembers where it comes from (`_useml_origin`): passed to ANOTHER
experience, it is declared as {"$output": …} and reloaded there, never pickled across.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from ._refs import recipe as _recipe

_SAVERS: list = []                 # (predicate, save, load, kind) registered by projects


def register_saver(kind: str, predicate, save, load) -> None:
    """Teach experiences a type: `predicate(value)`, `save(value, folder)`,
    `load(folder)`. Registered savers are tried first."""
    _SAVERS.insert(0, (kind, predicate, save, load))


def _is_figure(value) -> bool:
    return type(value).__name__ == "Figure" and hasattr(value, "savefig")


def _is_module(value) -> bool:
    return hasattr(value, "state_dict") and hasattr(value, "load_state_dict") and \
        type(value).__module__.split(".")[0] != "builtins"


def _config_recipe(value):
    try:
        recipe = _recipe(value, None, where="output")
    except Exception:                                     # noqa: BLE001
        return None
    return recipe if isinstance(recipe, dict) and "$config" in recipe else None


def save(value, folder: Path) -> dict:
    """Write `value` under `folder`; return the metadata `load` needs."""
    folder.mkdir(parents=True, exist_ok=True)
    for kind, predicate, saver, _ in _SAVERS:
        if predicate(value):
            saver(value, folder / "output")
            return {"kind": kind}
    if value is None:
        return {"kind": "none"}
    if _is_module(value):
        config = _config_recipe(value)
        import torch

        if config is not None:
            torch.save(value.state_dict(), folder / "weights.pt")
            return {"kind": "config_module", "config": config["$config"]}
        torch.save(value, folder / "module.pt")
        return {"kind": "torch"}
    if hasattr(value, "save") and hasattr(type(value), "load"):
        value.save(folder / "output")
        cls = type(value)
        return {"kind": "custom", "class": f"{cls.__module__}:{cls.__qualname__}"}
    try:
        import numpy as np

        if isinstance(value, np.ndarray) or (
                isinstance(value, dict) and value and
                all(isinstance(item, np.ndarray) for item in value.values())):
            arrays = value if isinstance(value, dict) else {"__array__": value}
            np.savez(folder / "output.npz", **arrays)
            return {"kind": "npz"}
    except ImportError:                                     # pragma: no cover
        pass
    if _is_figure(value):
        value.savefig(folder / "output.png", bbox_inches="tight", dpi=150)
        return {"kind": "figure", "path": "output.png"}
    with_figures = _replace_figures(value, folder)
    try:
        text = json.dumps(with_figures, indent=1, ensure_ascii=False)
    except (TypeError, ValueError):
        with open(folder / "output.pkl", "wb") as file:
            pickle.dump(value, file)
        print(f"[experience] output of type {type(value).__name__} pickled: it may not load "
              f"again after the code changes. Return plain values, arrays, or an object "
              f"with save()/load().", flush=True)
        return {"kind": "pickle"}
    (folder / "output.json").write_text(text)
    return {"kind": "json"}


def _replace_figures(value, folder: Path, prefix: str = "figure"):
    if _is_figure(value):
        path = folder / f"{prefix}.png"
        value.savefig(path, bbox_inches="tight", dpi=150)
        return {"$figure": path.name}
    if isinstance(value, dict):
        return {key: _replace_figures(item, folder, f"{prefix}-{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_figures(item, folder, f"{prefix}-{index}")
                for index, item in enumerate(value)]
    return value


def load(folder: Path, meta: dict):
    kind = meta.get("kind")
    for registered, _, _, loader in _SAVERS:
        if registered == kind:
            return loader(folder / "output")
    if kind == "none":
        return None
    if kind == "config_module":
        import torch

        from .. import config as configs

        spec = meta["config"]
        # A FRESH build, never the memoised one: loading trained weights into the object
        # `config.load(name)` returns would change it for everyone holding it.
        model = configs.load_raw(spec["name"], spec.get("kind"),
                                 overrides=spec.get("overrides")).build()
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu"))
        return model
    if kind == "torch":
        import torch

        return torch.load(folder / "module.pt", map_location="cpu", weights_only=False)
    if kind == "custom":
        from ._refs import resolve_function

        return resolve_function(meta["class"]).load(folder / "output")
    if kind == "npz":
        import numpy as np

        with np.load(folder / "output.npz", allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        return values["__array__"] if list(values) == ["__array__"] else values
    if kind == "figure":
        return folder / meta["path"]
    if kind == "json":
        return _restore_figures(json.loads((folder / "output.json").read_text()), folder)
    if kind == "pickle":
        with open(folder / "output.pkl", "rb") as file:
            return pickle.load(file)
    raise ValueError(f"unknown output kind {kind!r} in {folder}")


def _restore_figures(value, folder: Path):
    if isinstance(value, dict):
        if set(value) == {"$figure"}:
            return folder / value["$figure"]
        return {key: _restore_figures(item, folder) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_figures(item, folder) for item in value]
    return value
