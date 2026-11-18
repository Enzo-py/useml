"""useml.config — objects described in YAML, built by name in one line.

    from useml import config

    model = config.load("resnet_ft")                                   # memoised by name
    model = config.load("resnet_ft", optimizer={"lr": 0.1}) # one setting, merged
    sc   = config.load("accuracy", model="resnet_ft")               # fills the object slot
    config.ls(); print(config.describe("resnet_ft")); config.find("a3f9c21e")
    config.create("small", MyModel(hidden=64))            # the inverse
    config.diff("resnet_ft", model)

THE LIBRARY (where the files are, where the types come from) is found once:

  1. `config.use(root, packages=[...], ...)` — explicit;
  2. else `[tool.useml.config]` in the nearest `pyproject.toml` above the current folder:

        [tool.useml.config]
        root = "configs"                          # relative to the pyproject
        packages = ["mymodels", "mymetrics"]      # where `type:` names are looked up
        kinds = {model = "models", scorer = "scorers"}   # default: every sub-folder
        default_kind = "model"
        pythonpath = ["."]                        # added to sys.path (packages not installed)

  3. else a `configs/` folder above the current folder, with no type package (register
     classes with `@config.register`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ._config import Config, expand_dotted, merge, parse_override, summarize_overrides
from ._library import DEFAULT_META_FIELDS, Library
from ._types import ConfigError
from . import _write

_DEFAULT: Library | None = None
_PENDING_REGISTRATIONS: list = []


def use(root=None, **options) -> Library:
    """Set (and return) the default library. Without `root`, discover it again."""
    global _DEFAULT
    _DEFAULT = Library(root, **options) if root is not None else _discover()
    for target, name in _PENDING_REGISTRATIONS:
        _DEFAULT.register(target, name=name)
    return _DEFAULT


def library() -> Library:
    """The default library (discovered on first use)."""
    return _DEFAULT if _DEFAULT is not None else use()


def _discover() -> Library:
    explicit = os.environ.get("USEML_CONFIG_ROOT")
    if explicit:
        return Library(explicit)
    from .._project import section

    folder, settings = section("config")
    if settings is not None:
        settings = dict(settings)
        for entry in settings.pop("pythonpath", []):
            path = str((folder / entry).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        root = folder / settings.pop("root", "configs")
        return Library(root, **settings)
    start = Path.cwd().resolve()
    for folder in (start, *start.parents):
        if (folder / "configs").is_dir():
            return Library(folder / "configs")
    raise ConfigError(f"no config library found above {start}: add [tool.useml.config] to "
                      f"your pyproject.toml, create a `configs/` folder, or call "
                      f"useml.config.use(root).")


def register(target=None, *, name: str | None = None):
    """`@config.register`: make a class nameable as `type:` in the default library (also
    one discovered later)."""
    def wrap(cls):
        _PENDING_REGISTRATIONS.append((cls, name))
        if _DEFAULT is not None:
            _DEFAULT.register(cls, name=name)
        return cls
    return wrap(target) if target is not None else wrap


def load(name: str, kind: str | None = None, *, reload: bool = False,
         overrides: dict | None = None, **more):
    return library().load(name, kind, reload=reload, overrides=overrides, **more)


def load_raw(name: str, kind: str | None = None, *, overrides: dict | None = None, **more):
    return library().load_raw(name, kind, overrides=overrides, **more)


def ls() -> dict:
    return library().ls()


def describe(name: str, kind: str | None = None) -> str:
    return library().describe(name, kind)


def find(config_hash: str, kind: str | None = None) -> list[dict]:
    return library().find(config_hash, kind)


def get_config(instance, **options) -> Config:
    return _write.get_config(library(), instance, **options)


def create(name: str, instance, **options) -> Config:
    return _write.create(library(), name, instance, **options)


def diff(a, b, *, build: bool = True) -> dict:
    return _write.diff(library(), a, b, build=build)


load.__doc__ = Library.load.__doc__
__all__ = ["load", "load_raw", "ls", "describe", "find", "get_config", "create", "diff",
           "use", "library", "register", "Library", "Config", "ConfigError", "merge",
           "expand_dotted", "parse_override", "summarize_overrides", "DEFAULT_META_FIELDS"]
