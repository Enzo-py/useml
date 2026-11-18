"""useml.hub — every download goes through here, one thread at a time by default.

    import useml
    path = useml.hub.get("hf://openai/clip-vit-large-patch14")
    useml.hub.ls()                  # what is on disk, how big, last used
    useml.hub.scan()                # caches left in the home
    useml.hub.allow_parallel()      # lift the sequential policy, explicitly

See `_policy` for the network policy, `_fetch` for `get`, `_scan` for stray caches and
`_watch` for the external monitor.
"""
from ._fetch import adopt, extract, get, hub_errors_explained, ls, parse_hf, registry
from ._policy import (ParallelDownloadError, allow_parallel, apply_configured_policy,
                      enforce_connection_limit, is_parallel_allowed, sequential, status)
from ._scan import move, scan
from ._watch import watch

__all__ = [
    "get", "adopt", "extract", "ls", "registry", "parse_hf", "hub_errors_explained",
    "sequential", "allow_parallel", "is_parallel_allowed", "enforce_connection_limit",
    "apply_configured_policy", "status", "ParallelDownloadError",
    "scan", "move", "watch",
]
