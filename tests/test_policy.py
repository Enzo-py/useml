"""The network policy. A violation must never slip through, so each layer is tested on
the case it exists for."""
from __future__ import annotations

import builtins
import gc
import os
import socket
import sys
import threading
import types

import pytest

from useml.hub import _policy


# ── layer 2: the socket guard ─────────────────────────────────────────────────

@pytest.fixture
def server(monkeypatch):
    """A local TCP server, with loopback treated as REMOTE so the guard counts it."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    accepted = []
    stop = threading.Event()

    def accept():
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                accepted.append(listener.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    monkeypatch.setattr(_policy, "_is_local_address", lambda address: False)
    yield listener.getsockname()
    stop.set()
    thread.join()
    for connection in accepted:
        connection.close()
    listener.close()


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as error:          # noqa: BLE001 - reported to the test
            out["error"] = error
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return out


def test_one_thread_may_hold_several_sockets(server, tmp_path):
    _policy.enforce_connection_limit(log_path=tmp_path / "guard.log")
    sockets = [socket.create_connection(server) for _ in range(5)]
    for opened in sockets:
        opened.close()


def test_a_second_thread_is_refused_while_the_first_holds_one(server, tmp_path):
    log = tmp_path / "guard.log"
    _policy.enforce_connection_limit(log_path=log)
    held = socket.create_connection(server)
    try:
        out = _in_thread(lambda: socket.create_connection(server))
        assert isinstance(out.get("error"), _policy.ParallelDownloadError)
        assert "VIOLATION" in log.read_text()
    finally:
        held.close()


def test_after_close_another_thread_may_connect(server, tmp_path):
    _policy.enforce_connection_limit(log_path=tmp_path / "guard.log")
    socket.create_connection(server).close()
    out = _in_thread(lambda: socket.create_connection(server))
    assert "error" not in out
    out["value"].close()


def test_a_socket_collected_without_close_is_released(server, tmp_path):
    _policy.enforce_connection_limit(log_path=tmp_path / "guard.log")
    held = socket.create_connection(server)
    del held                                    # never closed: only collected
    gc.collect()
    out = _in_thread(lambda: socket.create_connection(server))
    assert "error" not in out, out.get("error")
    out["value"].close()


def test_runaway_backstop(server, tmp_path):
    _policy.enforce_connection_limit(max_sockets_per_thread=3, log_path=tmp_path / "g.log")
    sockets = [socket.create_connection(server) for _ in range(3)]
    with pytest.raises(_policy.ParallelDownloadError):
        socket.create_connection(server)
    for opened in sockets:
        opened.close()


def test_allow_parallel_lifts_the_guard(server, tmp_path):
    _policy.sequential(log_path=tmp_path / "guard.log")
    _policy.allow_parallel()
    held = socket.create_connection(server)
    out = _in_thread(lambda: socket.create_connection(server))
    assert "error" not in out
    out["value"].close()
    held.close()


# ── layer 0: xet ──────────────────────────────────────────────────────────────

def test_sequential_disables_xet_and_parallel_restores_it(tmp_path, monkeypatch):
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_DISABLE_XET = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    _policy.sequential(log_path=tmp_path / "g.log")
    assert os.environ["HF_HUB_DISABLE_XET"] == "1" and constants.HF_HUB_DISABLE_XET
    _policy.allow_parallel()
    assert "HF_HUB_DISABLE_XET" not in os.environ and not constants.HF_HUB_DISABLE_XET


# ── layer 1: library pools ────────────────────────────────────────────────────

def _real_thread_map(fn, *iterables, **kwargs):
    return [fn(*arguments) for arguments in zip(*iterables)]


@pytest.fixture
def fake_libraries(monkeypatch):
    snapshot = types.ModuleType("huggingface_hub._snapshot_download")
    snapshot.hf_thread_map = _real_thread_map
    monkeypatch.setitem(sys.modules, "huggingface_hub._snapshot_download", snapshot)
    monkeypatch.setattr(_policy, "_ORIGINAL_HUB_THREAD_MAP", None)
    monkeypatch.setattr(_policy, "_ORIGINAL_DATASETS_THREAD_MAP", None)
    monkeypatch.setattr(_policy, "_IMPORT_HOOK_INSTALLED", False)
    real_import = builtins.__import__
    for name in ("datasets", "datasets.download.download_manager"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield snapshot
    builtins.__import__ = real_import


def _install_datasets(*, ready: bool, complete: bool = True):
    manager = types.ModuleType("datasets.download.download_manager")
    if ready:
        manager.thread_map = _real_thread_map
    package = types.ModuleType("datasets")
    if complete:
        package.__version__ = "4.0.0"
    sys.modules["datasets"] = package
    sys.modules["datasets.download.download_manager"] = manager
    return manager


def test_hub_pool_is_made_sequential_and_restored(fake_libraries, tmp_path):
    _policy.sequential(log_path=tmp_path / "g.log")
    assert fake_libraries.hf_thread_map is _policy._sequential_map
    _policy.allow_parallel()
    assert fake_libraries.hf_thread_map is _real_thread_map


def test_datasets_imported_later_is_patched_by_the_hook(fake_libraries, tmp_path):
    _policy.sequential(log_path=tmp_path / "g.log")
    assert _policy._IMPORT_HOOK_INSTALLED
    manager = _install_datasets(ready=True)
    import json  # noqa: F401  - any import triggers the hook
    assert manager.thread_map is _policy._sequential_map
    assert not _policy._IMPORT_HOOK_INSTALLED


def test_half_initialised_datasets_does_not_raise_and_hook_waits(fake_libraries, tmp_path):
    _policy.sequential(log_path=tmp_path / "g.log")
    manager = _install_datasets(ready=False, complete=False)
    import json  # noqa: F401
    assert _policy._IMPORT_HOOK_INSTALLED
    manager.thread_map = _real_thread_map           # its body finished running
    import json  # noqa: F401,F811
    assert manager.thread_map is _policy._sequential_map


def test_moved_symbol_is_announced(fake_libraries, tmp_path):
    _policy.sequential(log_path=tmp_path / "g.log")
    _install_datasets(ready=False, complete=True)
    with pytest.warns(RuntimeWarning, match="NOT neutralised"):
        import json  # noqa: F401


def test_sequential_map_really_is_sequential():
    seen = []
    result = _policy._sequential_map(lambda x: seen.append(threading.get_ident()) or x * 2,
                                     [1, 2, 3], max_workers=8, chunksize=1)
    assert result == [2, 4, 6] and set(seen) == {threading.get_ident()}
