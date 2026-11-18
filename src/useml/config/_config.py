"""`Config`: a file read and resolved — what is needed to build, and to cite oneself.

READING A CONFIG LOADS NO WEIGHT. `.build()` builds; components keep their own lazy loading.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

META = "meta"
TYPE = "type"


@dataclass
class Config:
    name: str
    kind: str
    path: Path | None          # None: read back from an instance, never written
    type: str
    kwargs: dict
    meta: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    library: object = field(default=None, repr=False, compare=False)

    @property
    def display_name(self) -> str:
        """What goes in a table. PRESENTATION ONLY: never part of a cache key, or renaming
        a column would invalidate every result already computed."""
        return self.meta.get("display_name") or self.name

    @property
    def short_name(self) -> str:
        return self.meta.get("short_name") or self.display_name

    @property
    def effective(self) -> dict:
        """Constructor arguments with the overrides MERGED into the file's nodes."""
        out = dict(self.kwargs)
        for key, value in self.overrides.items():
            out[key] = merge(self.kwargs[key], value) if key in self.kwargs else value
        return out

    @property
    def hash(self) -> str:
        """Fingerprint of the RESOLVED declaration, overrides included. Reordering keys or
        reformatting the YAML changes nothing; `meta` is excluded (a display name does not
        make the model). Overrides count: a config loaded with `model=other` is not the
        file's config, and its citation must not pretend it is."""
        payload = json.dumps({"type": self.type, "kwargs": canonical(self.effective)},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    config_hash = hash

    @property
    def cited_name(self) -> str:
        """`+ovr` as soon as an override is in play: the file is no longer what runs."""
        return f"{self.name}+ovr" if self.overrides else self.name

    @property
    def label(self) -> str:
        """`short_name`, followed by what the overrides change: « ResNet-50 [lr=0.1] »."""
        if not self.overrides:
            return self.short_name
        return f"{self.short_name} [{summarize_overrides(self.overrides)}]"

    def provenance(self) -> str:
        """The line to store NEXT TO the numbers it describes (LaTeX, markdown comment).
        Recomputed at render time it would describe the config loaded then."""
        from ._git import git_head, relative

        where = relative(self.path) if self.path else "(not saved)"
        return f"{self.display_name}  cfg:{self.hash}  {where}  code:{git_head(self.path) or '?'}"

    def build(self):
        """Build the object (not memoised: `load` memoises)."""
        if self.library is None:
            raise RuntimeError("this Config has no library to resolve its types")
        return self.library.build_config(self)

    def to_dict(self) -> dict:
        out = {META: dict(self.meta)} if self.meta else {}
        out.update({TYPE: self.type, **ordered(self.effective)})
        return out

    def to_yaml(self) -> str:
        import yaml

        text = ""
        if self.meta:
            text = yaml.safe_dump({META: self.meta}, sort_keys=False, allow_unicode=True) + "\n"
        return text + yaml.safe_dump({TYPE: self.type, **ordered(self.effective)},
                                     sort_keys=False, allow_unicode=True)

    __str__ = to_yaml

    def __repr__(self) -> str:
        return f"Config({self.cited_name!r}, {self.kind}, {self.type}, cfg:{self.hash})"


def merge(base, override):
    """An override applied to a node. A DICT WITHOUT `type:` NOR `ref:` MERGES into the
    node, recursively: only what is named changes. EVERYTHING ELSE REPLACES: a scalar, a
    built object, a list, and a dict carrying its own `type:` or `ref:` (another node,
    declared whole)."""
    if isinstance(base, dict) and isinstance(override, dict) and \
            "type" not in override and "ref" not in override:
        out = dict(base)
        for key, value in override.items():
            out[key] = merge(base[key], value) if key in base else value
        return out
    return override


def expand_dotted(overrides: dict) -> dict:
    """{"optimizer.lr": 0.1} → {"optimizer": {"lr": 0.1}}."""
    out: dict = {}
    for key, value in overrides.items():
        parts = str(key).split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {key!r} conflicts with another one")
        node[parts[-1]] = merge(node.get(parts[-1]), value) if parts[-1] in node else value
    return out


def parse_override(text: str) -> tuple[str, object]:
    """"optimizer.lr=0.1" → ("optimizer.lr", 0.1); the value is read
    as YAML, so `3`, `0.5`, `true`, `[1, 2]` and `{type: X}` keep their type."""
    import yaml

    if "=" not in text:
        raise ValueError(f"override {text!r}: expected key=value")
    key, _, value = text.partition("=")
    return key.strip(), yaml.safe_load(value) if value.strip() else ""


def ordered(value):
    """Like `canonical`, keeping the key order (for display)."""
    if isinstance(value, dict):
        return {key: ordered(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ordered(item) for item in value]
    return canonical(value)


def canonical(value):
    """A value → a canonical form, serialisable and STABLE from one run to the next. An
    object's repr often holds its memory address; an object override is therefore
    represented by its type, and by the hash of ITS config when it has one."""
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    config = getattr(value, "config", None)
    return {"__obj__": type(value).__name__,
            "cfg": config.hash if isinstance(config, Config) else None}


def summarize_overrides(overrides: dict, limit: int = 60) -> str:
    """{"optimizer": {"lr": 0.1}} → « lr=0.1 »: the LEAF only (the whole
    path when two leaves share a name); a replaced node reads « matcher=Matcher(alpha=0.4) »."""
    leaves: list = []

    def show(value) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, (int, str)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ",".join(show(item) for item in value)
        if isinstance(value, dict):
            head = value.get("type") or value.get("ref") or ""
            rest = " ".join(f"{key}={show(item)}" for key, item in value.items()
                            if key not in ("type", "ref"))
            return f"{head}({rest})" if rest else str(head)
        return type(value).__name__

    def walk(node: dict, path: tuple) -> None:
        for key, value in node.items():
            here = (*path, str(key))
            if isinstance(value, dict) and not ({"type", "ref"} & set(value)):
                walk(value, here)
            else:
                leaves.append((here, value))

    walk(overrides, ())
    lasts = [path[-1] for path, _ in leaves]
    text = " ".join(f"{path[-1] if lasts.count(path[-1]) == 1 else '.'.join(path)}={show(v)}"
                    for path, v in leaves)
    return text if len(text) <= limit else text[:limit - 1] + "…"
