"""useml — one root for every ML cache, one gate for every download.

IMPORTING useml HAS TWO EFFECTS, on purpose, and only these two:

  1. `useml.paths.apply()` — if a root is configured, every library cache variable
     (HF_HOME, TORCH_HOME, TMPDIR…) points under it. Without a root, nothing changes.
  2. `useml.hub.apply_configured_policy()` — downloads become sequential (one thread
     at a time), unless the machine config says `[hub] parallel = true`.

Import useml FIRST, before torch / transformers / open_clip, so they read the right paths.
If they were imported earlier, `paths.apply()` realigns huggingface_hub's paths anyway.

Set `USEML_NO_AUTO=1` to import without either effect (then call them yourself).
"""
from __future__ import annotations

import os as _os

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("useml")
except PackageNotFoundError:            # pragma: no cover - source checkout
    __version__ = "dev"

from . import paths, hub, store

if _os.environ.get("USEML_NO_AUTO", "").strip() not in ("1", "true", "yes"):
    paths.apply()
    hub.apply_configured_policy()

__all__ = ["paths", "hub", "store", "data", "config", "exp", "__version__"]


def __getattr__(name):
    """`useml.data` and `useml.config` are imported on first use: they need numpy / PyYAML,
    `import useml` does not."""
    if name in ("data", "config", "exp"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'useml' has no attribute {name!r}")
