"""The closed namespace of `type:` names, and the errors that say what to fix.

NO IMPORT BY STRING, and never `eval`. A config file is content: it is read, copied from
one notebook to another, and giving it the right to import an arbitrary path would make
every YAML executable code. A name resolved in a CLOSED namespace fails readably; an
import path fails silently. The namespace is:

  • every public name exported by the library's `packages` (imported lazily: loading
    `mymodels` costs torch, and `import useml` must cost nothing);
  • every class registered with `@useml.config.register`.
"""
from __future__ import annotations

import difflib
import importlib
import inspect


class ConfigError(ValueError):
    """A config that cannot be read or built. Always carries the file and the place in
    the tree: an error without its address forces reading the whole file again."""


class Namespace:
    def __init__(self, packages=(), registered: dict | None = None) -> None:
        self.packages = tuple(packages)
        self.registered = dict(registered or {})
        self._names: dict | None = None

    def register(self, target, name: str | None = None):
        self.registered[name or target.__name__] = target
        if self._names is not None:
            self._names[name or target.__name__] = target
        return target

    def names(self) -> dict:
        if self._names is None:
            names: dict = {}
            for package in self.packages:
                try:
                    module = importlib.import_module(package)
                except ImportError as error:
                    raise ConfigError(f"cannot import the type package {package!r}: "
                                      f"{error}") from error
                for name in getattr(module, "__all__", None) or dir(module):
                    if not name.startswith("_"):
                        names.setdefault(name, getattr(module, name))
            names.update(self.registered)
            self._names = names
        return self._names

    def resolve(self, name: str, *, where: str = ""):
        """`"LinearHead"` → the class. Raises with the closest names."""
        if name in self.registered:
            return self.registered[name]
        names = self.names()
        if name in names:
            return names[name]
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        sources = " and ".join(self.packages) or "nothing"
        raise ConfigError(f"{where}unknown type: {name!r}.{hint} Types come from "
                          f"{sources}, plus classes registered with "
                          f"@useml.config.register.")

    def name_of(self, cls) -> str | None:
        """The name under which `cls` reads back, or None if it does not."""
        for name, value in self.registered.items():
            if value is cls:
                return name
        return cls.__name__ if self.names().get(cls.__name__) is cls else None


def parameters(cls) -> set | None:
    """Named constructor parameters, or None if the signature cannot be read."""
    try:
        values = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return None
    return {p.name for p in values if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def accepts_anything(cls) -> bool:
    try:
        return any(p.kind is p.VAR_KEYWORD for p in inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return False


def instantiate(cls, kwargs: dict, *, where: str):
    """Build, translating the two errors that really happen. An UNKNOWN KEY is caught
    BEFORE the call: Python's TypeError names the parameter but not the file nor the place
    in the tree, and suggests nothing — and a typo in a setting name is exactly what a
    config system must catch."""
    accepted = parameters(cls)
    if accepted is not None:
        unknown = [key for key in kwargs if key not in accepted]
        if unknown and not accepts_anything(cls):
            details = "; ".join(
                f"{key!r}" + (f" (close to {', '.join(close)})" if (close := difflib.
                              get_close_matches(key, accepted, n=2, cutoff=0.6)) else "")
                for key in unknown)
            raise ConfigError(f"{where}{cls.__name__} does not take {details}. Accepted "
                              f"parameters: {', '.join(sorted(accepted))}.")
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except Exception as error:                                   # noqa: BLE001
        raise ConfigError(f"{where}{cls.__name__}(…) raised {type(error).__name__}: "
                          f"{error}") from error
