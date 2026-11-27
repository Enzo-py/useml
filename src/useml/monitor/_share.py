"""What the agent may send: the `[monitor]` table of the machine config, and the masking.

    # ~/.config/useml/config.toml
    [monitor]
    space = "<user>/useml-monitor"         # set by `useml monitor deploy`
    machine = "gpu-a"                      # the name shown; never the real hostname
    share = ["status", "steps", "progress", "errors", "log", "metrics",
             "results", "figures", "configs", "data", "machine"]
    experiences = ["*"]                    # globs of experience names to report
    roots = []                             # experiences folders (default: this project's)
    log_lines = 30
    redact = []                            # more strings replaced by <redacted>
    heartbeat = 60                         # seconds between unrequested snapshots
    min_interval = 10                      # at most one push per 10 s when things change

`useml monitor snapshot` prints exactly what would be sent with these settings.

NEVER SENT, whatever the settings: the code snapshots of experiences, installed packages,
environment variables, tokens. There is no group for them.

MASKING, applied to every text of the snapshot (log lines, errors, configs, results):
the home folder → `~`, the project folders → `<project>`, the useml and data roots →
`<useml>` / `<data>`, the hostname → `<host>`, the user name (as a word) → `<user>`, and
every string of `redact` → `<redacted>`.
"""
from __future__ import annotations

import fnmatch
import getpass
import hashlib
import os
import platform
import re
from pathlib import Path

GROUPS = {
    "status": "name, status, steps done / total, current step, last update",
    "steps": "each step: name, function (module:name, never its code), done, duration, output kind",
    "progress": "the current step's progress (i / n and its label)",
    "errors": "why a run stopped: signal or cause, and the end of the Python error",
    "log": "the last lines of run.log",
    "metrics": "metric curves (checkpoint metrics and ctx.metric values)",
    "results": "JSON outputs of finished steps (up to 64 KB each)",
    "figures": "PNG figures of finished steps (up to 6 per experience, 400 KB each)",
    "configs": "the configs a step reads: name, hash, overrides, YAML",
    "data": "names of the datasets and processed data a step reads",
    "machine": "GPUs (name, use, memory), RAM, free disk, load",
    "setup": "useml itself: version, settings (masked), store, processed data, datasets, "
             "downloads, caches left in the home, download policy, suggested commands",
    "projects": "the project and method of each experience, its factors (plain arguments, "
                "configs and overrides, datasets, tags), the project's declaration; and when "
                "a run ends, its record (factors, numbers, curves, configs), which the "
                "dashboard keeps in its private dataset",
}
DEFAULT_SHARE = list(GROUPS)


def settings() -> dict:
    from .. import paths

    table = dict(paths.config().get("monitor", {}) or {})
    share = table.get("share", DEFAULT_SHARE)
    unknown = [group for group in share if group not in GROUPS]
    if unknown:
        raise ValueError(f"[monitor] share: unknown groups {unknown}; known: {list(GROUPS)}")
    if share and "status" not in share:
        share = ["status", *share]           # nothing can be shown without it
    return {
        "space": table.get("space"),
        "url": table.get("url"),
        "machine": str(table.get("machine") or default_alias()),
        "label": str(table["label"]) if table.get("label") else None,
        "share": list(share),
        "experiences": list(table.get("experiences", ["*"])),
        "roots": [Path(os.path.expandvars(str(r))).expanduser() for r in table.get("roots", [])],
        "log_lines": int(table.get("log_lines", 30)),
        "redact": [str(item) for item in table.get("redact", []) if str(item)],
        "heartbeat": float(table.get("heartbeat", 60)),
        "min_interval": float(table.get("min_interval", 10)),
    }


def machine_id() -> str:
    """An identifier of this computer that survives network changes and reboots: the
    platform UUID (macOS), /etc/machine-id (Linux), the MachineGuid (Windows); else the
    hostname without its domain (macOS adds `.local` on some networks)."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                                 capture_output=True, text=True, timeout=5).stdout
            found = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
            if found:
                return found.group(1)
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            if os.path.isfile(path):
                text = open(path).read().strip()
                if text:
                    return text
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Cryptography") as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0])
    except Exception:                                            # noqa: BLE001
        pass
    return platform.node().split(".")[0].lower()


def default_alias() -> str:
    """A stable name that reveals nothing: a hash of `machine_id()` (never sent as is).
    `[monitor] machine` replaces it."""
    return "machine-" + hashlib.sha256(f"useml|{machine_id()}".encode()).hexdigest()[:6]


def wanted(name: str, patterns: list) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


class Masker:
    """Replaces identifying strings in every text of a snapshot."""

    def __init__(self, *, projects=(), extra=()) -> None:
        from .. import paths

        pairs = []
        for folder in projects:
            pairs.append((str(Path(folder).resolve()), "<project>"))
            pairs.append((str(Path(folder)), "<project>"))
        for label, folder in (("<useml>", paths.home()), ("<data>", _data_root()),
                              ("<datasets>", _datasets_root())):
            if folder:
                pairs.append((str(Path(folder).expanduser().resolve()), label))
        home = str(Path.home())
        pairs += [(str(Path(home).resolve()), "~"), (home, "~")]
        host = platform.node()
        if host:
            pairs.append((host, "<host>"))
            if "." in host:
                pairs.append((host.split(".")[0], "<host>"))
        pairs += [(item, "<redacted>") for item in extra]
        # Longest first: a project inside the home becomes <project>, not ~/…
        self.pairs = sorted({(a, b) for a, b in pairs if a and len(a) > 2},
                            key=lambda pair: -len(pair[0]))
        try:
            user = getpass.getuser()
        except Exception:                                        # noqa: BLE001
            user = ""
        self.user = re.compile(rf"\b{re.escape(user)}\b") if len(user) > 2 else None

    def text(self, value: str) -> str:
        for needle, replacement in self.pairs:
            if needle in value:
                value = value.replace(needle, replacement)
        if self.user is not None:
            value = self.user.sub("<user>", value)
        return value

    def apply(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            return {self.apply(key) if isinstance(key, str) else key: self.apply(item)
                    for key, item in value.items()}
        return value


def _data_root():
    try:
        from ..data._catalog import data_root

        return data_root()
    except Exception:                                            # noqa: BLE001
        return None


def _datasets_root():
    try:
        from ..datasets import root

        return root()
    except Exception:                                            # noqa: BLE001
        return None
