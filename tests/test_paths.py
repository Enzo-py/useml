from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from useml import paths


def test_without_root_nothing_changes(fake_home):
    assert paths.apply() is None
    assert "HF_HOME" not in os.environ
    assert paths.home() == fake_home / ".cache" / "useml"


def test_root_from_config_file(tmp_path):
    target = tmp_path / "big"
    Path(os.environ["USEML_CONFIG"]).write_text(f'root = "{target}"\n')
    assert paths.root() == target


def test_variable_wins_over_config_file(tmp_path, root):
    Path(os.environ["USEML_CONFIG"]).write_text(f'root = "{tmp_path / "other"}"\n')
    assert paths.root() == root


def test_apply_points_every_variable_under_the_root(root):
    assert paths.apply() == root
    assert os.environ["HF_HOME"] == str(root / "hub" / "huggingface")
    assert os.environ["TMPDIR"] == str(root / "tmp")
    assert Path(os.environ["TORCH_HOME"]).is_dir()


def test_a_variable_already_outside_the_home_is_kept(tmp_path, root, monkeypatch):
    chosen = tmp_path / "bigdisk" / "hf"
    monkeypatch.setenv("HF_HOME", str(chosen))
    paths.apply()
    assert os.environ["HF_HOME"] == str(chosen)
    assert "HF_HOME" in paths._kept_variables


def test_a_variable_under_the_home_is_moved(fake_home, root, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(fake_home / ".cache" / "huggingface"))
    paths.apply()
    assert os.environ["HF_HOME"] == str(root / "hub" / "huggingface")


def test_config_env_table_places_a_cache_elsewhere(tmp_path, root):
    elsewhere = tmp_path / "weights"
    Path(os.environ["USEML_CONFIG"]).write_text(f'[env]\nHF_HOME = "{elsewhere}"\n')
    paths.apply()
    assert os.environ["HF_HOME"] == str(elsewhere)


def test_empty_part_variable_follows_hf_home(fake_home, root, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_home / "empty_hub"))
    paths.apply()
    assert os.environ["HF_HUB_CACHE"] == str(root / "hub" / "huggingface" / "hub")


def test_populated_part_variable_is_kept(fake_home, root, monkeypatch, capsys):
    populated = fake_home / "hub"
    (populated / "models--org--name").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(populated))
    paths.apply()
    assert os.environ["HF_HUB_CACHE"] == str(populated)
    assert "already holds a cache" in capsys.readouterr().err


def test_login_token_stays_readable(fake_home, root):
    token = fake_home / ".cache" / "huggingface" / "token"
    token.parent.mkdir(parents=True)
    token.write_text("hf_x")
    paths.apply()
    assert os.environ["HF_TOKEN_PATH"] == str(token)


def test_huggingface_imported_too_early_is_realigned(root, monkeypatch):
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HOME = "/somewhere/old"
    constants.HF_HUB_CACHE = "/somewhere/old/hub"
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    paths.apply()
    assert constants.HF_HUB_CACHE == str(root / "hub" / "huggingface" / "hub")


def test_apply_is_idempotent(root):
    paths.apply()
    first = dict(os.environ)
    paths.apply()
    assert dict(os.environ) == first


def test_diagnose_runs(root):
    text = paths.diagnose()
    assert str(root) in text and "HF_HOME" in text
