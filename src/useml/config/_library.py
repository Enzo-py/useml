"""A config library: a folder of YAML files per kind, a namespace of types, a memo.

ONE FILE DESCRIBES A TREE OF CONSTRUCTORS. A node is a scalar, a list, a mapping, or
`{type: Name, **kwargs}` — built recursively — or `{ref: other_config, **overrides}`.

    meta:
      display_name: "ResNet-50 (fine-tuned)"
    type: Classifier
    clip: {ref: clip_l}
    head: {type: LinearHead, dropout: 0.1}

WHAT IS LEFT OUT STAYS AT ITS DEFAULT: a three-line file works, one writes only what
differs. WHAT IS MISSPELLED RAISES, with the closest name: `alpha_fuqw` never silently
does nothing.

`load(name)` IS MEMOISED BY NAME (no overrides): two calls in a notebook return the SAME
object, so a CLIP is loaded once. `reload=True` rebuilds after editing the file. With
overrides there is no memo: two different variants must be two objects.
"""
from __future__ import annotations

import difflib
import inspect
from pathlib import Path

from ._config import META, TYPE, Config, expand_dotted, merge
from ._types import ConfigError, Namespace, instantiate, parameters

DEFAULT_META_FIELDS = ("display_name", "short_name", "note", "cite", "tags")
_SCALARS = (str, int, float, bool)
_SCALAR_NAMES = {"str", "int", "float", "bool"}
_MACHINE_PARAMETERS = {"device", "dtype", "torch_dtype"}


class Library:
    def __init__(self, root, *, kinds: dict | None = None, packages=(),
                 default_kind: str | None = None, meta_fields=DEFAULT_META_FIELDS,
                 slot_alias: str | None = "model", on_build=None, kind_of=None) -> None:
        """`root` — the configs folder. `kinds` — {kind: sub-folder}; by default every
        sub-folder of `root` is a kind. `packages` — where `type:` names are looked up.
        `default_kind` — the kind a bare name or `{ref: …}` means first. `slot_alias` —
        the override name that fills a config's OBJECT slot, whatever that parameter is
        called (`load("accuracy", model="resnet_ft")` fills `encoder`). `on_build(obj,
        config)` — runs on each built object. `kind_of(instance)` — the kind to write an
        instance as (`create`)."""
        self.root = Path(root).expanduser().resolve()
        self._kinds = {kind: self.root / folder for kind, folder in kinds.items()} \
            if kinds else None
        self.namespace = Namespace(packages)
        self._default_kind = default_kind
        self.meta_fields = tuple(meta_fields)
        self.slot_alias = slot_alias
        self.on_build = on_build
        self.kind_of = kind_of
        self._memo: dict = {}

    def __repr__(self) -> str:
        return f"Library({str(self.root)!r}, kinds={list(self.kinds)})"

    # ── kinds and files ──────────────────────────────────────────────────────

    @property
    def kinds(self) -> dict:
        if self._kinds is not None:
            return self._kinds
        if not self.root.is_dir():
            return {}
        return {path.name: path for path in sorted(self.root.iterdir())
                if path.is_dir() and not path.name.startswith((".", "_"))}

    @property
    def default_kind(self) -> str | None:
        return self._default_kind or next(iter(self.kinds), None)

    def register(self, target=None, *, name: str | None = None):
        """`@library.register` or `library.register(cls)`: make a class nameable."""
        if target is None:
            return lambda cls: self.namespace.register(cls, name)
        return self.namespace.register(target, name)

    def _file(self, name: str, kind: str) -> Path | None:
        folder = self.kinds.get(kind)
        if folder is None:
            raise ConfigError(f"unknown kind {kind!r}. Known: {', '.join(self.kinds) or 'none'} "
                              f"(folders of {self.root}).")
        for suffix in (".yaml", ".yml"):
            if (folder / f"{name}{suffix}").exists():
                return folder / f"{name}{suffix}"
        return None

    def resolve_kind(self, name: str, kind: str | None = None) -> str:
        """The kind holding `name`: the one given; else the default kind if it has it;
        else the only kind that has it."""
        if kind is not None:
            if self._file(name, kind) is None:
                close = difflib.get_close_matches(name, self.names(kind), n=3, cutoff=0.5)
                hint = f" Did you mean {', '.join(close)}?" if close else ""
                raise ConfigError(f"no {kind} config named {name!r} in "
                                  f"{self.kinds[kind]}.{hint}")
            return kind
        if self.default_kind and self._file(name, self.default_kind):
            return self.default_kind
        holders = [candidate for candidate in self.kinds if self._file(name, candidate)]
        if len(holders) == 1:
            return holders[0]
        if len(holders) > 1:
            raise ConfigError(f"{name!r} exists as {' and '.join(holders)}: say which "
                              f"with kind=…")
        everything = [n for k in self.kinds for n in self.names(k)]
        close = difflib.get_close_matches(name, everything, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ConfigError(f"no config named {name!r} in {self.root}.{hint}")

    def names(self, kind: str | None = None) -> list:
        kinds = [kind] if kind else list(self.kinds)
        out = set()
        for each in kinds:
            folder = self.kinds.get(each)
            if folder and folder.is_dir():
                out.update(path.stem for path in folder.iterdir()
                           if path.suffix in (".yaml", ".yml"))
        return sorted(out)

    def ls(self) -> dict:
        return {kind: self.names(kind) for kind in self.kinds}

    # ── reading ──────────────────────────────────────────────────────────────

    def load_raw(self, name: str, kind: str | None = None, *, overrides: dict | None = None,
                 **more) -> Config:
        """The config read and checked, WITHOUT building anything (cite it, describe it)."""
        import yaml

        kind = self.resolve_kind(name, kind)
        path = self._file(name, kind)
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as error:
            raise ConfigError(f"[{path}] invalid YAML: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigError(f"[{path}] the root must be a mapping, not {type(raw).__name__}.")
        if TYPE not in raw:
            raise ConfigError(f"[{path}] `type:` is missing: the class to build, for example "
                              f"`type: Classifier`.")
        meta = raw.get(META) or {}
        if not isinstance(meta, dict):
            raise ConfigError(f"[{path}] `meta:` must be a mapping.")
        unknown = [key for key in meta if key not in self.meta_fields]
        if unknown:
            raise ConfigError(f"[{path}] `meta:` does not know {unknown}. Accepted fields: "
                              f"{', '.join(self.meta_fields)}.")
        config = Config(name=name, kind=kind, path=path, type=str(raw[TYPE]),
                        kwargs={k: v for k, v in raw.items() if k not in (META, TYPE)},
                        meta=meta, library=self)
        merged = expand_dotted({**(overrides or {}), **more})
        if merged:
            config.overrides = self._resolve_overrides(config, merged)
        return config

    def describe(self, name: str, kind: str | None = None) -> str:
        from ._git import relative

        config = self.load_raw(name, kind)
        lines = [f"{config.name}  ({config.display_name})  [{config.kind}]",
                 f"  file      {relative(config.path)}",
                 f"  hash      cfg:{config.hash}",
                 f"  type      {config.type}"]
        if config.meta.get("note"):
            lines.append(f"  note      {config.meta['note']}")
        slot = self.object_slot(config)
        if slot and self.slot_alias:
            lines.append(f"  {self.slot_alias}=    fills `{slot}` (inferred)")
        for key, value in config.kwargs.items():
            shown = value.get(TYPE, value.get("ref", value)) if isinstance(value, dict) else value
            lines.append(f"  {key:9s} {shown}")
        return "\n".join(lines)

    # ── overrides and the object slot ────────────────────────────────────────

    def object_slot(self, config: Config) -> str | None:
        """The parameter `model=` fills, INFERRED: the FIRST positional parameter that
        cannot be a setting, with a POSITIVE sign — the file declares it as an object
        (`{ref:}` / `{type:}`), or its annotation is present and not a scalar, or its
        default is None (« an object, built later »). A parameter the file fills with a
        scalar is never it. None when nothing qualifies: the caller then says which
        parameters exist instead of guessing."""
        try:
            values = inspect.signature(self.namespace.resolve(config.type)).parameters.values()
        except (ConfigError, TypeError, ValueError):
            return None
        for parameter in values:
            if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD:
                continue
            if parameter.name in _MACHINE_PARAMETERS:
                continue                  # `device=None` means « this machine », not an object
            declared = config.kwargs.get(parameter.name)
            if isinstance(declared, dict) and ("ref" in declared or "type" in declared):
                return parameter.name
            if isinstance(declared, _SCALARS):
                continue
            if parameter.default is not parameter.empty and \
                    isinstance(parameter.default, _SCALARS):
                continue
            if _scalar_annotation(parameter.annotation):
                continue
            if parameter.annotation is not parameter.empty or (
                    parameter.default is not parameter.empty and parameter.default is None):
                return parameter.name
        return None

    def _resolve_overrides(self, config: Config, overrides: dict) -> dict:
        out = dict(overrides)
        alias = self.slot_alias
        if not alias or alias not in out:
            return out
        accepted = parameters(self.namespace.resolve(config.type, where=f"[{config.path}] "))
        if accepted and alias in accepted:
            return out                              # a real parameter of that name
        slot = self.object_slot(config)
        if slot is None:
            raise ConfigError(f"[{config.path}] `{alias}=` finds no slot on {config.type}: "
                              f"none of its positional parameters can hold an object. "
                              f"Override the intended parameter directly. Parameters: "
                              f"{', '.join(sorted(accepted or [])) or 'none'}.")
        if slot in out:
            raise ConfigError(f"[{config.path}] `{alias}=` and `{slot}=` name the same slot "
                              f"and were both passed. Keep one.")
        value = out.pop(alias)
        out[slot] = self.load(value) if isinstance(value, str) else value
        return out

    # ── building ─────────────────────────────────────────────────────────────

    def build_node(self, node, *, where: str = ""):
        """A config node → the value given to the parent constructor. A dict with `type:`
        is BUILT; everything else passes through, recursively."""
        if isinstance(node, dict):
            if "ref" in node:
                if "type" in node:
                    raise ConfigError(f"{where}a reference cannot also declare `type:`; "
                                      f"drop one of the two.")
                overrides = {k: v for k, v in node.items() if k not in ("ref", "kind")}
                return self.load(str(node["ref"]), node.get("kind"), overrides=overrides)
            if TYPE not in node:
                return {k: self.build_node(v, where=f"{where}{k}.") for k, v in node.items()}
            cls = self.namespace.resolve(node[TYPE], where=where)
            kwargs = {k: self.build_node(v, where=f"{where}{k}.") for k, v in node.items()
                      if k != TYPE}
            return instantiate(cls, kwargs, where=where)
        if isinstance(node, (list, tuple)):
            return [self.build_node(v, where=f"{where}[{i}].") for i, v in enumerate(node)]
        return node

    def build_config(self, config: Config):
        where = f"[{config.path}] " if config.path else ""
        cls = self.namespace.resolve(config.type, where=where)
        kwargs = {k: self.build_node(v, where=f"{where}{k}.")
                  for k, v in config.effective.items()}
        return self._attach(instantiate(cls, kwargs, where=where), config)

    def _attach(self, obj, config: Config):
        """The object knows where it comes from (`obj.config`), so it can write its
        provenance next to its numbers. An object with `__slots__` cannot; not an error."""
        try:
            obj.config = config
            obj._useml_snapshot = self.snapshot(obj)
        except AttributeError:
            pass
        if self.on_build is not None:
            self.on_build(obj, config)
        return obj

    def snapshot(self, obj) -> dict | None:
        """The object's arguments READ BACK on it (see `_write`), or None if they cannot
        be. Taken right after building: comparing a later read with it says whether the
        object was changed by hand — without the false differences a comparison with the
        FILE shows for values the constructor resolves."""
        import warnings

        from ._config import canonical
        from ._write import get_config

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return canonical(get_config(self, obj, refs=False).kwargs)
        except Exception:                                        # noqa: BLE001
            return None

    def load(self, name: str, kind: str | None = None, *, reload: bool = False,
             overrides: dict | None = None, **more):
        """The object described by `<root>/<kind>/<name>.yaml`.

            model = lib.load("resnet_ft")
            model = lib.load("resnet_ft", optimizer={"lr": 0.1})   # merged
            model = lib.load("resnet_ft", overrides={"optimizer.lr": 0.1})
            sc   = lib.load("accuracy", model="resnet_ft")                 # the object slot
        """
        merged = {**(overrides or {}), **more}
        if merged:
            from .. import _progress

            config = self.load_raw(name, kind, overrides=merged)
            with _progress.step(f"building {config.label}"):
                return config.build()
        key = (self.resolve_kind(name, kind), name)
        if reload:
            self._memo.pop(key, None)
        if key not in self._memo:
            from .. import _progress

            config = self.load_raw(name, key[0])
            with _progress.step(f"building {config.label}"):
                self._memo[key] = config.build()
        return self._memo[key]

    def clear(self) -> None:
        """Forget every memoised object."""
        self._memo.clear()

    def find(self, config_hash: str, kind: str | None = None) -> list[dict]:
        from ._git import find

        return find(self, config_hash, kind)

    def get_config(self, instance, **options) -> Config:
        """The config of a live instance (see `_write.get_config`)."""
        from ._write import get_config

        return get_config(self, instance, **options)

    def create(self, name: str, instance, **options) -> Config:
        """Write an instance as a config file (see `_write.create`)."""
        from ._write import create

        return create(self, name, instance, **options)

    def diff(self, a, b, *, build: bool = True) -> dict:
        """Where two configs differ (see `_write.diff`)."""
        from ._write import diff

        return diff(self, a, b, build=build)


def _scalar_annotation(annotation) -> bool:
    """READ AS TEXT: with `from __future__ import annotations` they are strings, and
    `annotation in (str, int)` would be false everywhere — silently. `"int | None"` counts
    as a scalar: a setting or nothing is still a setting."""
    if annotation is inspect.Parameter.empty:
        return False
    text = annotation if isinstance(annotation, str) else getattr(
        annotation, "__name__", str(annotation))
    parts = {part.strip() for part in str(text).replace("Optional[", "").rstrip("]").split("|")}
    return bool(parts & _SCALAR_NAMES)


__all__ = ["Library", "ConfigError", "merge"]
