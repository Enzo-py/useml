"""The network policy: ONE thread may download at a time, unless allowed otherwise.

    useml.hub.sequential()       # the default, applied at `import useml`
    useml.hub.allow_parallel()   # lift it, explicitly

A violation must never slip through silently, so this is not one patch
but FOUR independent layers, each covering what the one below cannot see:

  0. HUGGING FACE XET. `huggingface_hub` ≥ 0.32 downloads through `hf_xet`, a Rust
     extension that opens its OWN parallel connections, outside Python's `socket`
     module: layer 2 cannot see them. Sequential mode sets `HF_HUB_DISABLE_XET=1` (and
     the already-imported constant), so files come through plain Python HTTP.
  1. LIBRARY PATCHES. The two known thread pools are replaced by a sequential map:
     `huggingface_hub._snapshot_download` (`thread_map` / `hf_thread_map`, 8 workers)
     and `datasets.download.download_manager.thread_map` (16 workers). `datasets` is
     heavy, so it is not imported here: an import hook patches it when something
     imports it. `hf_transfer` (multi-part single file) is disabled too.
  2. THE SOCKET GUARD. `socket.socket.connect` itself is wrapped: an outbound TCP
     connection from a SECOND thread while another still holds one is REFUSED (raises).
     It counts THREADS, not sockets: one sequential downloader legitimately holds
     several keep-alive sockets (Hugging Face redirects each file to its own CDN host).
     A per-thread cap of 16 sockets stays as a backstop against a runaway.
  3. THE EXTERNAL MONITOR (`useml hub watch -- <command>`, `_watch.py`). A separate
     process watching the whole process tree with psutil: it sees DataLoader workers
     and subprocesses, whose own interpreter layers 0–2 never patched.
"""
from __future__ import annotations

import builtins
import datetime
import functools
import os
import socket
import sys
import threading
import traceback
import warnings
import weakref

_ORIGINAL_HUB_THREAD_MAP = None        # huggingface_hub's parallel thread_map
_ORIGINAL_DATASETS_THREAD_MAP = None   # datasets' own, separate parallel thread_map
_PARALLEL = False                      # last policy requested, applied to late imports
_IMPORT_HOOK_INSTALLED = False
_XET_BEFORE: str | None = None


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if sys.stderr.isatty() else text


def _sequential_map(fn, *iterables, **kwargs):
    """Drop-in for `tqdm.contrib.concurrent.thread_map`: same signature, one item at a
    time."""
    kwargs.pop("max_workers", None)
    kwargs.pop("chunksize", None)
    bar = kwargs.pop("tqdm_class", None)
    if bar is None:
        try:
            from tqdm.auto import tqdm as bar
        except ImportError:                    # pragma: no cover - tqdm comes with the hub
            return list(map(fn, *iterables))
    try:
        wrapped = bar(map(fn, *iterables), **kwargs)
    except (TypeError, KeyError):              # a leftover non-tqdm keyword
        wrapped = map(fn, *iterables)
    return list(wrapped)


# ── layer 0: xet ──────────────────────────────────────────────────────────────

def _set_xet(parallel: bool) -> None:
    """Sequential: xet off. Parallel: back to the user's own `HF_HUB_DISABLE_XET`."""
    global _XET_BEFORE
    if parallel:
        if _XET_BEFORE is None:
            return                              # never disabled by us
        if _XET_BEFORE:
            os.environ["HF_HUB_DISABLE_XET"] = _XET_BEFORE
        else:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        _XET_BEFORE = None
    else:
        if _XET_BEFORE is None:
            _XET_BEFORE = os.environ.get("HF_HUB_DISABLE_XET", "")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and hasattr(constants, "HF_HUB_DISABLE_XET"):
        constants.HF_HUB_DISABLE_XET = os.environ.get(
            "HF_HUB_DISABLE_XET", "").upper() in {"1", "ON", "YES", "TRUE"}


# ── layer 1: library patches ──────────────────────────────────────────────────

def _patch_hub(parallel: bool) -> None:
    global _ORIGINAL_HUB_THREAD_MAP
    snapshot = sys.modules.get("huggingface_hub._snapshot_download")
    if snapshot is None:
        try:
            import huggingface_hub._snapshot_download as snapshot
        except Exception:
            return                             # hub not installed: nothing to patch
    # huggingface_hub ≥ 1.32 renamed `thread_map` to `hf_thread_map` (same signature).
    name = "hf_thread_map" if hasattr(snapshot, "hf_thread_map") else "thread_map"
    if not hasattr(snapshot, name):
        warnings.warn(_yellow(
            "useml network policy, layer 1: huggingface_hub._snapshot_download exposes "
            "neither `hf_thread_map` nor `thread_map`; its pool is NOT neutralised. "
            "Layers 2 and 3 still hold."), RuntimeWarning, stacklevel=3)
        return
    current = getattr(snapshot, name)
    if _ORIGINAL_HUB_THREAD_MAP is None and current is not _sequential_map:
        _ORIGINAL_HUB_THREAD_MAP = current
    if parallel:
        if _ORIGINAL_HUB_THREAD_MAP is not None:
            setattr(snapshot, name, _ORIGINAL_HUB_THREAD_MAP)
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    setattr(snapshot, name, _sequential_map)


def _patch_datasets(parallel: bool) -> bool:
    """Patch `datasets.download.download_manager.thread_map`; True once it is SET.

    Being in `sys.modules` does not mean being ready: Python registers a module BEFORE
    running its body (that is how circular imports work), so the hook can fire in the
    middle of `datasets`' own import, on a module whose `thread_map` is not bound yet.
    We test the ATTRIBUTE, and retry at the next import until it is there."""
    global _ORIGINAL_DATASETS_THREAD_MAP
    manager = sys.modules.get("datasets.download.download_manager")
    if manager is None or not hasattr(manager, "thread_map"):
        return False
    if _ORIGINAL_DATASETS_THREAD_MAP is None and manager.thread_map is not _sequential_map:
        _ORIGINAL_DATASETS_THREAD_MAP = manager.thread_map
    if parallel:
        if _ORIGINAL_DATASETS_THREAD_MAP is not None:
            manager.thread_map = _ORIGINAL_DATASETS_THREAD_MAP
    else:
        manager.thread_map = _sequential_map
    return True


def _datasets_symbol_moved() -> bool:
    """`datasets` FULLY imported without the `thread_map` we patch: the library moved the
    symbol, and layer 1 no longer covers it. Said loudly, never silently."""
    package = sys.modules.get("datasets")
    manager = sys.modules.get("datasets.download.download_manager")
    return (package is not None and hasattr(package, "__version__")
            and manager is not None and not hasattr(manager, "thread_map"))


def _install_datasets_import_hook() -> None:
    """Wrap `builtins.__import__` until `datasets.download.download_manager` is ready,
    patch it, then step out of the way."""
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return
    _IMPORT_HOOK_INSTALLED = True
    real_import = builtins.__import__

    def step_out() -> None:
        global _IMPORT_HOOK_INSTALLED
        if builtins.__import__ is guarded_import:   # never unwrap someone else's wrapper
            builtins.__import__ = real_import
        _IMPORT_HOOK_INSTALLED = False

    def guarded_import(name, *args, **kwargs):
        module = real_import(name, *args, **kwargs)
        if not _IMPORT_HOOK_INSTALLED:
            return module                           # already done, still wrapped by others
        if _patch_datasets(_PARALLEL):
            step_out()
        elif _datasets_symbol_moved():
            step_out()
            warnings.warn(_yellow(
                "useml network policy, layer 1: `datasets.download.download_manager` no "
                "longer exposes `thread_map`, so its download pool is NOT neutralised. "
                "Layers 2 and 3 still hold."), RuntimeWarning, stacklevel=2)
        return module

    builtins.__import__ = guarded_import


# ── public switches ───────────────────────────────────────────────────────────

def sequential(*, log_path=None) -> None:
    """One thread downloads at a time, in every library (the default). Idempotent, safe
    before or after the libraries are imported."""
    global _PARALLEL
    _PARALLEL = False
    _set_xet(False)
    _patch_hub(False)
    if not _patch_datasets(False):
        _install_datasets_import_hook()
    enforce_connection_limit(log_path=log_path)


def allow_parallel() -> None:
    """Restore each library's parallel downloads and lift the socket guard. Explicit on
    purpose: nothing in useml calls it."""
    global _PARALLEL
    _PARALLEL = True
    _set_xet(True)
    _patch_hub(True)
    _patch_datasets(True)
    _lift_connection_limit()


def is_parallel_allowed() -> bool:
    return _PARALLEL


def apply_configured_policy() -> None:
    """What `import useml` runs: `[hub] parallel = true` in the config allows parallel
    downloads, anything else (or no config) is sequential."""
    from .. import paths

    try:
        parallel = bool(paths.config().get("hub", {}).get("parallel", False))
    except ValueError as error:                # an unreadable config must not open the gate
        print(f"⚠ {error}; downloads stay sequential.", file=sys.stderr)
        parallel = False
    allow_parallel() if parallel else sequential()


# ── layer 2: the socket guard ─────────────────────────────────────────────────

_REAL_CONNECT = None
_REAL_CONNECT_EX = None
_REAL_CLOSE = None
_lock = threading.Lock()
_MAX_DOWNLOAD_THREADS = 1               # THE policy
_MAX_SOCKETS_PER_THREAD = 16            # backstop only: a pool is small, a fan-out is not
_GUARD_INSTALLED = False
_LOG_PATH = None
#: thread ident → number of tracked sockets it holds open
_thread_sockets: dict = {}
# `socket.socket` has __slots__: "is this socket counted" cannot be an attribute. It maps
# to a `weakref.finalize`, which releases the count on close() OR on garbage collection,
# exactly once (a socket collected without close() must not stay counted forever).
_tracked_sockets: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class ParallelDownloadError(RuntimeError):
    """A second thread tried to open a network connection while another holds one."""


def _is_local_address(address) -> bool:
    try:
        host = address[0]
    except (TypeError, IndexError):
        return True                     # not a (host, port) tuple: a UNIX socket
    return host in ("", "localhost", "127.0.0.1", "::1") or str(host).startswith("127.")


def _log_violation(address, reason: str) -> None:
    line = (f"\n=== NETWORK GUARD VIOLATION — {datetime.datetime.now().isoformat()} ===\n"
            f"blocked connection to {address!r}: {reason}\n"
            f"threads holding sockets: {dict(_thread_sockets)}\n"
            f"{''.join(traceback.format_stack()[:-2])}")
    sys.stderr.write(line)
    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a") as file:
                file.write(line)
        except OSError:
            pass                         # logging must never be why the guard crashes


def _release(ident: int) -> None:
    with _lock:
        left = _thread_sockets.get(ident, 0) - 1
        if left > 0:
            _thread_sockets[ident] = left
        else:
            _thread_sockets.pop(ident, None)


def _guarded_connect(self, address, *, _real):
    track = getattr(self, "type", None) == socket.SOCK_STREAM and \
        not _is_local_address(address)
    if track:
        ident = threading.get_ident()
        with _lock:
            mine = _thread_sockets.get(ident, 0)
            others = [thread for thread in _thread_sockets if thread != ident]
            if mine == 0 and len(others) + 1 > _MAX_DOWNLOAD_THREADS:
                reason = (f"a second thread tried to open a connection while thread(s) "
                          f"{others} still hold one — parallel downloading")
            elif mine + 1 > _MAX_SOCKETS_PER_THREAD:
                reason = (f"thread {ident} already holds {mine} open connections "
                          f"(limit {_MAX_SOCKETS_PER_THREAD}) — runaway connection use")
            else:
                reason = None
                _thread_sockets[ident] = mine + 1
        if reason is not None:
            _log_violation(address, reason)
            raise ParallelDownloadError(
                f"[useml network guard] refused connection to {address!r}: {reason}. "
                f"The download policy is sequential (useml.hub.allow_parallel() lifts "
                f"it). Log: {_LOG_PATH}.")
        _tracked_sockets[self] = weakref.finalize(self, _release, ident)
    try:
        return _real(self, address)
    except BaseException:
        if track:
            finalizer = _tracked_sockets.pop(self, None)
            if finalizer is not None:
                finalizer()
        raise


def _guarded_close(self, *, _real):
    finalizer = _tracked_sockets.pop(self, None)
    if finalizer is not None:
        finalizer()
    return _real(self)


def enforce_connection_limit(max_threads: int = 1, max_sockets_per_thread: int = 16,
                             log_path=None) -> None:
    """Install the process-wide socket guard (idempotent). `max_threads` is the policy;
    `max_sockets_per_thread` is only a backstop and must stay well above a keep-alive
    pool (2 makes ordinary sequential downloads fail)."""
    global _REAL_CONNECT, _REAL_CONNECT_EX, _REAL_CLOSE, _GUARD_INSTALLED
    global _MAX_DOWNLOAD_THREADS, _MAX_SOCKETS_PER_THREAD, _LOG_PATH
    _MAX_DOWNLOAD_THREADS = int(max_threads)
    _MAX_SOCKETS_PER_THREAD = int(max_sockets_per_thread)
    if log_path is None:
        from .. import paths

        log_path = paths.directory("logs") / "network_guard.log"
    _LOG_PATH = str(log_path)
    if _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True
    _REAL_CONNECT = socket.socket.connect
    _REAL_CONNECT_EX = socket.socket.connect_ex
    _REAL_CLOSE = socket.socket.close
    socket.socket.connect = functools.wraps(_REAL_CONNECT)(
        lambda self, address: _guarded_connect(self, address, _real=_REAL_CONNECT))
    socket.socket.connect_ex = functools.wraps(_REAL_CONNECT_EX)(
        lambda self, address: _guarded_connect(self, address, _real=_REAL_CONNECT_EX))
    socket.socket.close = functools.wraps(_REAL_CLOSE)(
        lambda self: _guarded_close(self, _real=_REAL_CLOSE))


def _lift_connection_limit() -> None:
    global _GUARD_INSTALLED
    if not _GUARD_INSTALLED:
        return
    socket.socket.connect = _REAL_CONNECT
    socket.socket.connect_ex = _REAL_CONNECT_EX
    socket.socket.close = _REAL_CLOSE
    _GUARD_INSTALLED = False


def guard_installed() -> bool:
    return _GUARD_INSTALLED


def status() -> dict:
    """What holds right now, layer by layer."""
    snapshot = sys.modules.get("huggingface_hub._snapshot_download")
    hub_patched = None
    if snapshot is not None:
        name = "hf_thread_map" if hasattr(snapshot, "hf_thread_map") else "thread_map"
        hub_patched = getattr(snapshot, name, None) is _sequential_map
    manager = sys.modules.get("datasets.download.download_manager")
    datasets_patched = None if manager is None else (
        getattr(manager, "thread_map", None) is _sequential_map)
    return {"parallel_allowed": _PARALLEL,
            "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET") == "1",
            "hub_pool_patched": hub_patched,
            "datasets_pool_patched": datasets_patched,
            "datasets_hook_waiting": _IMPORT_HOOK_INSTALLED and manager is None,
            "socket_guard": _GUARD_INSTALLED,
            "log": _LOG_PATH}
