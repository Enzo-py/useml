"""An instance → its config file: the inverse of `load`. And comparing two configs.

    lib.create("small", MyModel(hidden=64), meta={"display_name": "Small model"})
    print(lib.get_config(model))               # the YAML one would write
    lib.diff("resnet_ft", loaded_object)     # {"head.dropout": (0.1, 0.3)}

PYTHON DOES NOT KEEP A CONSTRUCTOR'S ARGUMENTS: they are READ BACK on the object — for
each parameter, the attribute of the same name (`self.max_new_tokens`), else its private
version (`self._max_new_tokens`). When that convention is not kept, it SAYS so instead of
guessing: a wrong config written silently is what a config system exists to prevent.
`**force` gives a value by hand.

ROOT DEFAULTS ARE WRITTEN (`defaults=True`): a default changed in the code must not
silently change a config already cited. In SUB-COMPONENTS only what differs is written.
A shared object is written ONCE (the parent injects it again). A `torch.device` or dtype
whose default is None is left out (resolved on the machine that runs).

VERIFIED BY ROUND TRIP: the written file is built again and must read back like the
original, or the file is removed and it raises.
"""
from __future__ import annotations

import inspect
import warnings
from pathlib import Path

from ._config import Config, ordered
from ._types import ConfigError

_ABSENT = object()


def _type_name(library, obj, *, where: str) -> str:
    name = library.namespace.name_of(type(obj))
    if name is None:
        cls = type(obj)
        raise ConfigError(f"{where}{cls.__module__}.{cls.__name__} is not in the library's "
                          f"namespace: the written file would not read back. Export it from "
                          f"one of {library.namespace.packages} or register it.")
    return name


def _to_yaml(library, value, *, where: str, seen: set, refs: bool = True):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        out = [_to_yaml(library, v, where=f"{where}[{i}].", seen=seen, refs=refs)
               for i, v in enumerate(value)]
        return _ABSENT if any(item is _ABSENT for item in out) else out
    if isinstance(value, dict):
        out = {str(k): _to_yaml(library, v, where=f"{where}{k}.", seen=seen, refs=refs)
               for k, v in value.items()}
        return _ABSENT if any(item is _ABSENT for item in out.values()) else out
    config = getattr(value, "config", None)
    if refs and isinstance(config, Config) and not config.overrides:
        return {"ref": config.name} if config.kind == library.default_kind else \
            {"ref": config.name, "kind": config.kind}
    try:
        return {"type": _type_name(library, value, where=where),
                **_arguments(library, value, where=where, seen=seen, refs=refs)}
    except ConfigError:
        return _ABSENT


def _arguments(library, instance, *, where: str = "", force: dict | None = None,
               seen: set | None = None, defaults: bool = False, refs: bool = True) -> dict:
    force = dict(force or {})
    seen = set() if seen is None else seen
    seen.add(id(instance))
    cls = type(instance)
    signature = inspect.signature(cls)
    params = [p for p in signature.parameters.values()
              if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    unknown = set(force) - {p.name for p in params}
    if unknown and not any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        raise ConfigError(f"{where}{cls.__name__} does not take {sorted(unknown)}.")
    out, skipped = {key: force[key] for key in sorted(unknown)}, []
    for p in params:
        if p.name in force:
            if force[p.name] is None and p.default is None:
                continue
            out[p.name] = force[p.name]
            continue
        value = getattr(instance, p.name, _ABSENT)
        if value is _ABSENT:
            value = getattr(instance, f"_{p.name}", _ABSENT)
        if value is _ABSENT:
            raise ConfigError(f"{where}{cls.__name__}: `{p.name}` cannot be found on the "
                              f"instance (neither `.{p.name}` nor `._{p.name}`), so its value "
                              f"cannot be read back. Give it by hand: `{p.name}=…`.")
        if p.default is not p.empty and _equal(value, p.default) and (
                not defaults or p.default is None):
            continue
        if id(value) in seen and not isinstance(value, (str, int, float, bool)):
            skipped.append(p.name)
            continue
        written = _to_yaml(library, value, where=f"{where}{p.name}.", seen=seen, refs=refs)
        if written is _ABSENT:
            if p.default is None and _machine_value(value):
                skipped.append(p.name)
                continue
            raise ConfigError(f"{where}{cls.__name__}: `{p.name}` is a "
                              f"{type(value).__name__}, which cannot be written to YAML. "
                              f"Give it by hand: `{p.name}=…` (a {{'ref': …}} works).")
        out[p.name] = written
    if skipped:
        warnings.warn(f"{where}{cls.__name__}: left at their default (resolved on the "
                      f"machine that runs, or written above): {', '.join(skipped)}.",
                      stacklevel=3)
    return out


def _machine_value(value) -> bool:
    kind = type(value)
    return kind.__module__.split(".")[0] == "torch" and kind.__name__ in ("device", "dtype")


def _equal(a, b) -> bool:
    try:
        return bool(a == b) and type(a) is type(b)
    except Exception:                                            # noqa: BLE001
        return False


def get_config(library, instance, *, defaults: bool = True, refs: bool = True,
               **force) -> Config:
    """The config of a LIVE instance, read on the object: it shows changes made after
    loading. Nothing is written."""
    source = getattr(instance, "config", None)
    source = source if isinstance(source, Config) else None
    kind = source.kind if source else (library.kind_of(instance) if library.kind_of
                                       else library.default_kind)
    return Config(name=source.name if source else type(instance).__name__, kind=kind,
                  path=None, type=_type_name(library, instance, where=""),
                  kwargs=_arguments(library, instance, force=force, defaults=defaults,
                                    refs=refs),
                  meta=dict(source.meta) if source else {}, library=library)


def create(library, name: str, instance, *, kind: str | None = None,
           meta: dict | None = None, overwrite: bool = False, verify: bool = True,
           defaults: bool = True, **force) -> Config:
    """Write `<root>/<kind>/<name>.yaml` from `instance`; return the config READ BACK.
    Never overwrites without `overwrite=True`: a config is cited by name in results."""
    kind = kind or (library.kind_of(instance) if library.kind_of else library.default_kind)
    if kind is None:
        raise ConfigError("no kind: pass kind=… (the library has no kind folder yet)")
    folder = library.kinds.get(kind, library.root / f"{kind}s")
    path = folder / f"{name}.yaml"
    if path.exists() and not overwrite:
        raise ConfigError(f"{path} already exists. `overwrite=True` replaces it.")
    meta = dict(meta or {})
    unknown = [key for key in meta if key not in library.meta_fields]
    if unknown:
        raise ConfigError(f"`meta` does not know {unknown}. Accepted fields: "
                          f"{', '.join(library.meta_fields)}.")
    read = get_config(library, instance, defaults=defaults, **force)
    read.meta = meta
    text = f"# Written by useml.config.create from an instance of {read.type}.\n" + read.to_yaml()
    former = path.read_text() if path.exists() else None
    folder.mkdir(parents=True, exist_ok=True)
    if library._kinds is not None and kind not in library._kinds:
        library._kinds[kind] = folder
    path.write_text(text)
    try:
        config = library.load_raw(name, kind)
        if verify:
            _verify(library, instance, config.build(), force)
    except Exception:
        if former is None:
            path.unlink()
        else:
            path.write_text(former)
        raise
    return config


def _verify(library, original, rebuilt, force: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = _arguments(library, original, force=force)
        after = _arguments(library, rebuilt, force=force)
    differences = {key: (before.get(key), after.get(key)) for key in before.keys() | after.keys()
                   if before.get(key) != after.get(key)}
    if differences:
        detail = "; ".join(f"{k}: {a!r} → {b!r}" for k, (a, b) in sorted(differences.items()))
        raise ConfigError(f"the file does not rebuild the same {type(original).__name__} "
                          f"(original → read back): {detail}. Give these arguments by hand.")


# ── comparing ─────────────────────────────────────────────────────────────────

def _as_config(library, value, *, build: bool) -> Config:
    if isinstance(value, str):
        value = library.load_raw(value)
    if isinstance(value, Config) and not build:
        return value
    obj = value.build() if isinstance(value, Config) else value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_config(library, obj, refs=False)


def _complete(library, node, *, where: str = ""):
    """The tree with ALL its defaults, and `{ref: …}` unrolled: compare what would be
    built, not how it is written."""
    if isinstance(node, list):
        return [_complete(library, v, where=f"{where}[{i}].") for i, v in enumerate(node)]
    if not isinstance(node, dict):
        return node
    if "ref" in node:
        config = library.load_raw(str(node["ref"]), node.get("kind"))
        return _complete(library, {"type": config.type, **config.kwargs}, where=where)
    out = {k: _complete(library, v, where=f"{where}{k}.") for k, v in node.items()}
    if "type" in out:
        try:
            params = inspect.signature(library.namespace.resolve(out["type"])).parameters.values()
        except (ConfigError, TypeError, ValueError):
            return out
        for p in params:
            if p.name in out or p.default is p.empty or p.default is None:
                continue
            value = _to_yaml(library, p.default, where=f"{where}{p.name}.", seen=set())
            if value is not _ABSENT:
                out[p.name] = _complete(library, value, where=f"{where}{p.name}.")
    return out


def _flatten(node, prefix: str = "") -> dict:
    if isinstance(node, dict) and node:
        out = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): node}


def diff(library, a, b, *, build: bool = True) -> dict:
    """`{path: (value_a, value_b)}` where two configs differ; empty if they describe the
    same object. Defaults are completed on both sides, refs unrolled, `meta` ignored. A
    file is BUILT to be compared (`build=False` compares declarations as written)."""
    config_a, config_b = _as_config(library, a, build=build), _as_config(library, b, build=build)
    flat_a = _flatten(_complete(library, {"type": config_a.type, **ordered(config_a.effective)}))
    flat_b = _flatten(_complete(library, {"type": config_b.type, **ordered(config_b.effective)}))
    return {key: (flat_a.get(key), flat_b.get(key)) for key in sorted(flat_a.keys() | flat_b.keys())
            if flat_a.get(key) != flat_b.get(key)}
