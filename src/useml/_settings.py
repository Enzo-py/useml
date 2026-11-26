"""`useml settings`: read and edit the machine config without losing anything.

    useml settings show                      # the file, and every known key with its value
    useml settings get monitor.log_lines
    useml settings set monitor.log_lines 50
    useml settings set monitor.share '["status", "steps", "errors"]'
    useml settings set env.HF_HOME /data/shared/hf
    useml settings unset store.max_size

EDITED LINE BY LINE, NOT REWRITTEN: comments, order and spacing of the file stay as they
are; only the line of the key changes (or is added to its table, or the table is added at
the end). The previous file is kept as `config.toml.bak`. The result is parsed again
before it replaces the file: an edit that would break it is refused.

ONLY KNOWN KEYS (`SCHEMA`), checked for their type, so a typo never silently does
nothing. `env.<VARIABLE>` accepts any variable name. Values are TOML: `50`, `true`,
`"text"`, `["a", "b"]`; a bare word is taken as a string.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

SCHEMA = {
    "root": ("path", "the cache root of this machine"),
    "env.*": ("path", "a cache placed elsewhere than under the root (e.g. env.HF_HOME)"),
    "hub.parallel": ("bool", "allow parallel downloads"),
    "store.root": ("path", "the cache store folder"),
    "store.max_size": ("size", "the budget of `useml store gc` (e.g. \"450GB\")"),
    "data.root": ("path", "the processed-data folder"),
    "data.max_size": ("size", "refuse to write a processed table beyond this"),
    "datasets.root": ("path", "the datasets folder"),
    "exp.root": ("path", "experiences folder when the project sets none"),
    "exp.notify": ("string", "command run when an experience ends or crashes"),
    "monitor.space": ("string", "the private Space of the dashboard (owner/name)"),
    "monitor.url": ("string", "the dashboard URL, if not a Space"),
    "monitor.machine": ("string", "the name shown for this machine"),
    "monitor.share": ("list", "groups sent to the dashboard"),
    "monitor.experiences": ("list", "globs of experience names reported"),
    "monitor.roots": ("list", "experiences folders reported"),
    "monitor.log_lines": ("int", "log lines sent"),
    "monitor.redact": ("list", "more strings masked"),
    "monitor.heartbeat": ("number", "seconds between unrequested snapshots"),
    "monitor.min_interval": ("number", "minimum seconds between pushes on change"),
}


class SettingsError(ValueError):
    pass


def _toml_loads(text: str) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                         # pragma: no cover
        import tomli as tomllib
    return tomllib.loads(text)


def describe_key(key: str) -> tuple[str, str]:
    if key in SCHEMA:
        return SCHEMA[key]
    if key.startswith("env.") and re.fullmatch(r"env\.[A-Za-z_][A-Za-z0-9_]*", key):
        return SCHEMA["env.*"]
    import difflib

    close = difflib.get_close_matches(key, [k for k in SCHEMA if k != "env.*"], n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise SettingsError(f"unknown setting {key!r}.{hint} `useml settings show` lists them.")


def parse_value(key: str, text: str):
    """A command-line value → a checked Python value."""
    kind, _ = describe_key(key)
    try:
        value = _toml_loads(f"v = {text}")["v"]
    except Exception:                                            # noqa: BLE001
        value = text                                             # a bare word: a string
    if kind == "bool" and not isinstance(value, bool):
        raise SettingsError(f"{key} is true or false, not {text!r}")
    if kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
        raise SettingsError(f"{key} is a whole number, not {text!r}")
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise SettingsError(f"{key} is a number, not {text!r}")
    if kind == "list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise SettingsError(f"{key} is a list of strings, e.g. '[\"a\", \"b\"]'")
        if key == "monitor.share":
            from .monitor._share import GROUPS

            unknown = [v for v in value if v not in GROUPS]
            if unknown:
                raise SettingsError(f"monitor.share: unknown groups {unknown}; known: "
                                    f"{list(GROUPS)}")
    if kind in ("path", "string", "size") and not isinstance(value, str):
        value = str(value)
    if kind == "size":
        from .store import parse_size

        parse_size(value)                                        # raises if unreadable
    return value


def format_value(value) -> str:
    """A Python value → its TOML text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


# ── the line editor ───────────────────────────────────────────────────────────

_HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def _bracket_balance(text: str) -> int:
    depth, quote = 0, None
    for index, char in enumerate(text):
        if quote:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "#":
            break
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
    return depth


def _locate(lines: list[str], table: str, key: str):
    """(start, end) line span of `key` in `table` (end exclusive), the index after which a
    new key of that table goes, and whether the table exists."""
    current, table_found, insert_at = "", table == "", None
    if table == "":
        insert_at = 0
    index = 0
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    while index < len(lines):
        line = lines[index]
        header = _HEADER.match(line)
        if header:
            current = header.group(1).strip()
            if current == table:
                table_found, insert_at = True, index + 1
            index += 1
            continue
        span = 1
        if "=" in line and _bracket_balance(line.split("=", 1)[1]) > 0:
            depth = _bracket_balance(line.split("=", 1)[1])
            while depth > 0 and index + span < len(lines):
                depth += _bracket_balance(lines[index + span])
                span += 1
        if current == table:
            if pattern.match(line):
                return (index, index + span), insert_at, table_found
            if line.strip() and not line.strip().startswith("#"):
                insert_at = index + span
        index += span
    return None, insert_at, table_found


def _split(key: str) -> tuple[str, str]:
    if "." not in key:
        return "", key
    table, name = key.split(".", 1)
    return table, name


def edit(key: str, value=None, *, remove: bool = False, path: Path | None = None) -> Path:
    """Set (or remove) `key` in the machine config, keeping everything else."""
    from . import paths

    describe_key(key)
    path = Path(path or paths.config_path())
    original = path.read_text() if path.exists() else ""
    lines = original.splitlines()
    table, name = _split(key)
    span, insert_at, table_found = _locate(lines, table, name)
    new_line = f"{name} = {format_value(value)}" if not remove else None
    if span is not None:
        lines[span[0]:span[1]] = [new_line] if new_line else []
    elif remove:
        return path                                      # nothing to remove
    elif table_found or table == "":
        lines.insert(insert_at or 0, new_line)
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines += [f"[{table}]", new_line]
    text = "\n".join(lines) + "\n"
    try:
        _toml_loads(text)
    except Exception as error:                           # noqa: BLE001
        raise SettingsError(f"this edit would break {path} ({error}); nothing changed") \
            from None
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
    return path


def get(key: str):
    from . import paths

    describe_key(key)
    table, name = _split(key)
    config = paths.config()
    return (config if table == "" else config.get(table, {}) or {}).get(name)


def listing() -> list[dict]:
    """Every known key, its kind, meaning and current value (env.* expanded)."""
    from . import paths

    config = paths.config()
    out = []
    for key, (kind, meaning) in SCHEMA.items():
        if key == "env.*":
            for variable, value in (config.get("env") or {}).items():
                out.append({"key": f"env.{variable}", "kind": kind, "meaning": meaning,
                            "value": value})
            out.append({"key": "env.<VARIABLE>", "kind": kind, "meaning": meaning,
                        "value": None})
            continue
        table, name = _split(key)
        value = (config if table == "" else config.get(table, {}) or {}).get(name)
        out.append({"key": key, "kind": kind, "meaning": meaning, "value": value})
    return out
