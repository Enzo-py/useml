"""Every test runs in a fake home with its own config file: no real cache is ever read,
written or moved, and the policy state of `useml.hub` is restored after each test."""
from __future__ import annotations

import os
import socket

os.environ["USEML_NO_AUTO"] = "1"          # before any `import useml`

import pytest

from useml import paths
from useml.hub import _fetch, _policy

_VARIABLES = (*paths.CACHE_VARIABLES, *paths.HUGGING_FACE_PART_VARIABLES, "HF_TOKEN_PATH",
              "HF_TOKEN", "USEML_ROOT", "USEML_CONFIG", "XDG_CONFIG_HOME",
              "HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER")


@pytest.fixture(autouse=True)
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for variable in _VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(paths, "_applied_root", None)
    monkeypatch.setattr(paths, "_original_values", {})
    monkeypatch.setattr(paths, "_kept_variables", {})
    monkeypatch.setattr(_fetch, "_REGISTRY", None)
    real = (socket.socket.connect, socket.socket.connect_ex, socket.socket.close)
    state = (_policy._GUARD_INSTALLED, _policy._PARALLEL, _policy._XET_BEFORE)
    yield home
    _policy._lift_connection_limit()
    socket.socket.connect, socket.socket.connect_ex, socket.socket.close = real
    _policy._GUARD_INSTALLED, _policy._PARALLEL, _policy._XET_BEFORE = state
    _policy._GUARD_INSTALLED = False
    _policy._thread_sockets.clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A configured root, OUTSIDE the fake home."""
    folder = tmp_path / "disk" / "useml"
    monkeypatch.setenv("USEML_ROOT", str(folder))
    return folder
