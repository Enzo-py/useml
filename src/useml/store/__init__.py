"""useml.store — the key → value cache of a machine.

    store = useml.store.open()
    masks = store.declare("processing/masks", cost="expensive",
                          description="segmentation masks")
    if (value := masks.get(key)) is None:
        value = compute()
        masks.put(key, value)

WHERE: `[store] root` in the machine config, else `<home>/store` (`useml.paths.home()`).
BUDGET: `[store] max_size = "450GB"`; `useml store gc` evicts down to it, cheapest to
recompute first. Nothing is evicted automatically.
"""
from __future__ import annotations

import os
from pathlib import Path

from ._maintenance import (adopt, check, collect_garbage, free_disk_bytes, garbage_plan,
                           merge_by_rename, move, real_size, repair)
from ._spaces import COSTS, Space, cost_rank, human, parse_size
from ._store import (INLINE_LIMIT, SpaceView, Store, UndeclaredSpaceError, canonical_key,
                     key_digest)

ROOT_VARIABLE = "USEML_STORE_ROOT"
_open: dict = {}


def default_root() -> Path:
    """`$USEML_STORE_ROOT`, else `[store] root` in the config, else `<home>/store`."""
    from .. import paths

    explicit = os.environ.get(ROOT_VARIABLE, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    configured = paths.config().get("store", {}).get("root")
    if configured:
        return Path(os.path.expandvars(str(configured))).expanduser()
    return paths.home() / "store"


def budget() -> int | None:
    """`[store] max_size` in bytes, or None."""
    from .. import paths

    value = paths.config().get("store", {}).get("max_size")
    return parse_size(value) if value else None


def open(root=None, *, strict: bool = True) -> Store:      # noqa: A001 - the natural name
    """The store at `root` (default: `default_root()`), one instance per root and process."""
    root = Path(root).expanduser() if root is not None else default_root()
    resolved = root.resolve()
    key = (resolved, strict)
    if key not in _open:
        _open[key] = Store(resolved, strict=strict)
    return _open[key]


__all__ = [
    "open", "default_root", "budget", "Store", "SpaceView", "Space", "COSTS",
    "UndeclaredSpaceError", "canonical_key", "key_digest", "INLINE_LIMIT",
    "check", "repair", "garbage_plan", "collect_garbage", "move", "adopt",
    "merge_by_rename", "real_size", "free_disk_bytes", "cost_rank", "parse_size", "human",
]
