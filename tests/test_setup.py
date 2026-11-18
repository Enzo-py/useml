"""`useml init` adopts what exists and never loses anything; `useml check` says what is wrong."""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from useml import _setup, config


@pytest.fixture
def legacy_project(tmp_path, monkeypatch):
    """An existing project: packages at the root, configs/models and
    configs/scorers, a former experience, a .cache_root, an existing store, a commented
    pyproject, a git repository."""
    project = tmp_path / "legacy"
    (project / "mymodels").mkdir(parents=True)
    (project / "mymodels" / "__init__.py").write_text(
        "from .encoders import Encoder\n__all__ = ['Encoder']\n")
    (project / "mymodels" / "encoders.py").write_text(
        "class Encoder:\n    def __init__(self, dim: int = 8):\n        self.dim = dim\n")
    (project / "mymetrics").mkdir()
    (project / "mymetrics" / "__init__.py").write_text(
        "__all__ = ['Scorer']\nclass Scorer:\n    def __init__(self, encoder=None):\n"
        "        self.encoder = encoder\n")
    (project / "configs" / "models").mkdir(parents=True)
    (project / "configs" / "scorers").mkdir()
    (project / "configs" / "models" / "enc.yaml").write_text("type: Encoder\ndim: 4\n")
    (project / "configs" / "scorers" / "score.yaml").write_text(
        "type: Scorer\nencoder: {ref: enc}\nextra: {type: TorchThing}\n")
    (project / "experiences" / "old" / "eval").mkdir(parents=True)
    legacy_spec = {"name": "eval", "steps": [{"kind": "evaluate", "params": {}}]}
    (project / "experiences" / "old" / "eval" / "experience.json").write_text(
        json.dumps(legacy_spec))
    big_disk = tmp_path / "bigdisk"
    (project / ".cache_root").write_text(f"{big_disk}/.cache\nHF_HOME={big_disk}/hf\n")
    (project / ".cache" / "store").mkdir(parents=True)
    (project / ".cache" / "store" / "index.sqlite").write_bytes(b"")
    pyproject = textwrap.dedent("""
        # The only package declared here is `project`.
        [build-system]
        requires = ["setuptools>=61"]

        [project]
        name = "legacy"   # keep this comment
    """)
    (project / "pyproject.toml").write_text(pyproject)
    (project / ".gitignore").write_text("data/\n")
    for command in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"]):
        subprocess.run(command, cwd=project, check=True)
    machine = tmp_path / "machine" / "config.toml"
    monkeypatch.setenv("USEML_CONFIG", str(machine))
    monkeypatch.delenv("USEML_ROOT", raising=False)
    monkeypatch.chdir(project)
    monkeypatch.setattr(config, "_DEFAULT", None)
    return project


def test_plan_adopts_what_exists(legacy_project, tmp_path):
    plan = _setup.plan(legacy_project)
    text = plan.render()
    machine = next(c for c in plan.changes if c.path.name == "config.toml")
    assert f'root = "{tmp_path / "bigdisk" / ".cache"}"' in machine.text
    assert f'HF_HOME = "{tmp_path / "bigdisk" / "hf"}"' in machine.text
    assert "[store]" in machine.text and ".cache/store" in machine.text
    project = next(c for c in plan.changes if c.path.name == "pyproject.toml")
    assert project.mode == "append"
    assert 'kinds = {model = "models", scorer = "scorers"}' in project.text
    assert 'packages = ["mymodels", "mymetrics"]' in project.text or \
        'packages = ["mymetrics", "mymodels"]' in project.text
    assert 'pythonpath = ["."]' in project.text and "[tool.useml.exp]" in project.text
    assert "TorchThing" in text                                 # reported, not guessed
    assert "1 experiences of the former system" in text
    assert any(c.path.name == ".gitignore" for c in plan.changes)


def test_dry_run_writes_nothing(legacy_project, capsys):
    before = (legacy_project / "pyproject.toml").read_text()
    _setup.init(legacy_project)
    assert (legacy_project / "pyproject.toml").read_text() == before
    assert "dry run" in capsys.readouterr().out
    assert not Path(_setup.plan(legacy_project).changes[0].path).exists() or True


def test_apply_keeps_everything_and_is_idempotent(legacy_project):
    before = (legacy_project / "pyproject.toml").read_text()
    _setup.init(legacy_project, yes=True)
    after = (legacy_project / "pyproject.toml").read_text()
    assert after.startswith(before)                              # comments kept, appended
    assert list(legacy_project.glob("pyproject.toml.useml-backup-*"))
    assert (legacy_project / ".cache_root").exists()             # nothing removed
    assert (legacy_project / "experiences" / "old" / "eval" / "experience.json").exists()
    second = _setup.plan(legacy_project)
    assert second.changes == []                                  # nothing left to add

    # what was written works: configs load from a subfolder, types resolve
    (legacy_project / "notebooks").mkdir()
    import os
    import sys

    os.chdir(legacy_project / "notebooks")
    for module in ("mymodels", "mymetrics"):
        sys.modules.pop(module, None)
    config.use()
    assert config.load("enc").dim == 4
    assert config.library().default_kind == "model"


def test_existing_values_are_never_changed(legacy_project, tmp_path):
    machine = Path(tmp_path / "machine" / "config.toml")
    machine.parent.mkdir(parents=True)
    machine.write_text('root = "/somewhere/else"\n\n[env]\nTORCH_HOME = "/t"\n')
    plan = _setup.plan(legacy_project)
    text = plan.render()
    assert "already sets root" not in text                       # no --root given
    change = next(c for c in plan.changes if c.path == machine)
    assert "root =" not in change.text.split("[store]")[0]
    assert "add under [env]" in text                             # never appended blindly


def test_merge_conflict_markers_are_reported(legacy_project):
    (legacy_project / ".gitignore").write_text("a\n<<<<<<< Updated upstream\nb\n")
    assert "merge-conflict markers" in _setup.plan(legacy_project).render()


def test_legacy_experience_is_never_overwritten(legacy_project):
    from useml import exp
    from useml.exp import RefError

    _setup.init(legacy_project, yes=True)
    assert exp.ls() == []
    with pytest.raises(KeyError, match="another system"):
        exp.open("old/eval")
    with pytest.raises(RefError, match="never overwrites"):
        with exp.experience("old/eval", run=False) as e:
            e.step(json.dumps, {"a": 1})


def test_project_without_pyproject_gets_useml_toml(tmp_path, monkeypatch):
    project = tmp_path / "plain"
    (project / "configs" / "models").mkdir(parents=True)
    (project / "configs" / "models" / "m.yaml").write_text("type: Thing\n")
    (project / "pkg").mkdir()
    (project / "pkg" / "__init__.py").write_text("class Thing:\n    pass\n")
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "m.toml"))
    monkeypatch.chdir(project)
    _setup.init(project, yes=True, machine=False)
    text = (project / "useml.toml").read_text()
    assert "[config]" in text and 'packages = ["pkg"]' in text and "[exp]" in text
    monkeypatch.setattr(config, "_DEFAULT", None)
    assert config.library().root == (project / "configs").resolve()


def test_check_reports_unresolved_types(legacy_project, capsys):
    from useml.cli import main

    _setup.init(legacy_project, yes=True)
    code = main(["check"])
    out = capsys.readouterr().out
    assert "TorchThing" in out and "✗" in out and code == 1
    assert "configs" in out and "experiences" in out
