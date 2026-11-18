"""Arguments as RECIPES: what lets another process rebuild them.

An experience runs in its own process. What a step receives must therefore be described,
not passed. At declaration, every argument becomes JSON:

    {"$step": "train"}                          the output of an earlier step
    {"$config": {"name", "kind", "overrides", "hash"}}   an object from useml.config
    {"$data": {"name", "table", "rows", "digests"}}      a processed dataset (useml.data)
    {"$output": {"experience", "step", "fingerprint"}}   a step of ANOTHER experience
    {"$path": "/abs/path"}                      a path
    dicts, lists and plain values               walked / passed as they are

ERRORS FALL ON THE LINE THAT CAUSES THEM, not ten minutes later in run.log: an object
that cannot be described raises when it is declared, saying what to do instead.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path


class RefError(ValueError):
    """An argument or a step an experience cannot describe."""


class StepRef:
    """The output of a declared step. Pass it to later steps; after the run, `.get()`."""

    def __init__(self, builder, function: str) -> None:
        self._builder = builder
        self.function = function
        self.name: str | None = None

    def get(self):
        return self._builder.get(self)

    def __repr__(self) -> str:
        return f"<step {self.name or '(named when the block ends)'} = {self.function}(…)>"


def importable(fn, *, what: str = "exp.step") -> str:
    """`module:qualname` of a function another process can import, or RefError."""
    module = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None)
    if not callable(fn) or not module or not name:
        raise RefError(f"{what}: {fn!r} is not a function.")
    if module == "__main__" or "<" in name:
        raise RefError(f"{what}: {name!r} is defined in a notebook cell, a script's __main__ "
                       f"or inside another function. The experience runs in another process "
                       f"and cannot import it: move it into a module of the project.")
    target = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part, None)
    if target is not fn and getattr(target, "target", None) is not fn:
        raise RefError(f"{what}: {module}.{name} does not resolve back to this function.")
    return f"{module}:{name}"


def resolve_function(path: str):
    module_name, _, name = path.partition(":")
    target = importlib.import_module(module_name)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def code_hash(fn) -> str:
    try:
        return hashlib.blake2b(inspect.getsource(fn).encode(), digest_size=6).hexdigest()
    except (OSError, TypeError):
        return ""


# ── declaration: value → recipe ───────────────────────────────────────────────

def recipe(value, builder, *, where: str):
    """A Python value → its JSON recipe (StepRefs stay objects until names are fixed)."""
    if isinstance(value, StepRef):
        if value._builder is not builder:
            raise RefError(f"{where}: this step belongs to another experience; pass "
                           f"its output (`useml.exp.get(...)`) instead.")
        return value
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return {"$path": str(value.expanduser().resolve())}
    if isinstance(value, (list, tuple)):
        return [recipe(item, builder, where=f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RefError(f"{where}: dict keys must be strings to be declared.")
        if any(key.startswith("$") for key in value):
            raise RefError(f"{where}: keys starting with '$' are reserved.")
        return {key: recipe(item, builder, where=f"{where}.{key}") for key, item in value.items()}
    origin = origin_of(value)
    if isinstance(origin, dict):
        return {"$output": dict(origin)}
    processed = _processed_recipe(value, where)
    if processed is not None:
        return processed
    configured = _config_recipe(value, where)
    if configured is not None:
        return configured
    raise RefError(f"{where}: a {type(value).__name__} built in this process cannot be "
                   f"rebuilt by the experience. Pass instead: an object loaded with "
                   f"useml.config.load(name, **overrides), a processed dataset "
                   f"useml.data.load(name), the output of a step, or plain values.")


def _processed_recipe(value, where):
    try:
        from ..data import Processed
    except ImportError:                          # pragma: no cover - numpy missing
        return None
    if not isinstance(value, Processed):
        return None
    if not value.name:
        raise RefError(f"{where}: this processed dataset has no name. Give it one: "
                       f"useml.data.process(..., name='…'), then pass data.load('…').")
    from ..data import record

    entry = record(value.name)
    digest = hashlib.blake2b(value.digests.tobytes(), digest_size=8).hexdigest()
    return {"$data": {"name": value.name, "table": entry["table"], "rows": len(value),
                      "digests": digest}}


def _config_recipe(value, where):
    from ..config import Config

    config = getattr(value, "config", None)
    if not isinstance(config, Config):
        return None
    try:
        json.dumps(config.overrides)
    except TypeError:
        raise RefError(f"{where}: {config.name!r} was loaded with overrides holding live "
                       f"objects; only plain values can be declared.") from None
    before = getattr(value, "_useml_snapshot", None)
    if before is not None and config.library is not None:
        after = config.library.snapshot(value)
        if after is not None and after != before:
            from ..config._write import _flatten

            flat_before, flat_after = _flatten(before), _flatten(after)
            changed = [f"{key}: {flat_before.get(key)!r} → {flat_after.get(key)!r}"
                       for key in sorted(flat_before.keys() | flat_after.keys())
                       if flat_before.get(key) != flat_after.get(key)]
            raise RefError(f"{where}: {config.name!r} was modified after loading "
                           f"({', '.join(changed[:4])}). The experience rebuilds it from its "
                           f"file, so pass the change as an override: "
                           f"useml.config.load({config.name!r}, …).")
    return {"$config": {"name": config.name, "kind": config.kind,
                        "overrides": dict(config.overrides), "hash": config.hash}}


#: id(value) → (weak or strong reference, origin), for outputs that cannot hold an
#: attribute (numpy arrays, dicts, lists).
_ORIGINS: dict = {}


def set_origin(value, origin: dict) -> None:
    """Remember which experience step `value` was loaded from."""
    import weakref

    try:
        value._useml_origin = origin
        return
    except (AttributeError, TypeError):
        pass
    if value is None or isinstance(value, (str, bool, int, float)):
        return                                   # plain values are declared as they are
    key = id(value)
    try:
        reference = weakref.ref(value, lambda _, key=key: _ORIGINS.pop(key, None))
    except TypeError:                            # dict, list: no weak reference
        reference = (lambda value=value: value)
    _ORIGINS[key] = (reference, origin)


def origin_of(value):
    origin = getattr(value, "_useml_origin", None)
    if isinstance(origin, dict):
        return origin
    entry = _ORIGINS.get(id(value))
    if entry is not None and entry[0]() is value:
        return entry[1]
    return None


def freeze(value):
    """Replace StepRefs by {"$step": name} once names are fixed."""
    if isinstance(value, StepRef):
        return {"$step": value.name}
    if isinstance(value, list):
        return [freeze(item) for item in value]
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    return value


def steps_read(value) -> list:
    if isinstance(value, dict):
        if set(value) == {"$step"}:
            return [value["$step"]]
        return [name for item in value.values() for name in steps_read(item)]
    if isinstance(value, list):
        return [name for item in value for name in steps_read(item)]
    return []


# ── execution: recipe → value ─────────────────────────────────────────────────

def rebuild(value, outputs):
    """A recipe → the object, in the running experience. `outputs(name)` gives a step's
    output."""
    if isinstance(value, dict):
        if set(value) == {"$step"}:
            return outputs(value["$step"])
        if set(value) == {"$config"}:
            from .. import config

            spec = value["$config"]
            return config.load(spec["name"], spec.get("kind"), overrides=spec.get("overrides"))
        if set(value) == {"$data"}:
            from .. import data

            return data.load(value["$data"]["name"])
        if set(value) == {"$output"}:
            from ._experience import open_experience

            spec = value["$output"]
            return open_experience(spec["experience"]).get(spec["step"])
        if set(value) == {"$path"}:
            return Path(value["$path"])
        return {key: rebuild(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [rebuild(item, outputs) for item in value]
    return value
