"""Curves, progress and log for steps that never call ctx."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, exp  # noqa: E402
from useml.exp._capture import parse_line  # noqa: E402


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "useml.toml").write_text('[exp]\nroot = "experiences"\n')
    monkeypatch.chdir(folder)
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    monkeypatch.setenv("USEML_NO_TMUX", "1")
    monkeypatch.setenv("USEML_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "machine.toml"))
    monkeypatch.setenv("PYTHONPATH", str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(config, "_DEFAULT", None)
    return folder


def points(name, step):
    path = exp.open(name).path(step) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("line,expected", [
    ("epoch=3 loss=0.23 acc=0.81", (3.0, {"loss": 0.23, "acc": 0.81})),
    ("step 120 | train/loss: 0.5, lr: 1e-4", (120.0, {"train/loss": 0.5, "lr": 1e-4})),
    ("Epoch: 2, loss: 0.4", (2.0, {"loss": 0.4})),
    ("Epoch 3/10 - loss: 0.2310 - accuracy: 0.9120", (3.0, {"loss": 0.231, "accuracy": 0.912})),
    ("seed: 42", (None, {})),                                 # one `name: n` alone
    ("[experience] train: done in 3s (json)", (None, {})),
    ("saved to /data/run-3/ckpt.pt", (None, {})),
    ("port=8080 loss=1", (None, {"loss": 1.0})),
    ("100%|####| 6/6 [00:00<00:00, 1.2it/s, loss=0.167]", (None, {})),   # read from the bar
    ("2026-11-26 18:03:48  retrying in 5 s", (None, {})),
])
def test_parse_line(line, expected):
    assert parse_line(line) == expected


def test_in_process_run_gets_a_log_curves_and_progress(capsys):
    with exp.experience("vision/tq", detach=False) as e:
        e.step(steps.tqdm_train, 6, name="train")
    log = exp.open("vision/tq").folder.joinpath("run.log").read_text()
    assert "(in this process)" in log and "epoch=5 val_loss=0.333" in log
    assert "\r" not in log and "[experience] finished" in log
    found = points("vision/tq", "train")
    losses = [p for p in found if "val_loss" in p]
    assert [p["step"] for p in losses] == [0, 1, 2, 3, 4, 5]                # x from epoch=
    assert any("loss" in p and "acc" in p for p in found)                   # tqdm postfix
    assert not any("seed" in p for p in found)
    assert "epoch=5 val_loss=0.333" in capsys.readouterr().out             # still printed


def test_detached_run_too():
    with exp.experience("vision/tq-detached") as e:
        e.step(steps.tqdm_train, 4, name="train")
    experience = exp.open("vision/tq-detached")
    assert experience.status == "done"
    assert [p["step"] for p in points("vision/tq-detached", "train") if "val_loss" in p] == [0, 1, 2, 3]


def test_ctx_metric_wins_and_the_switch(monkeypatch):
    with exp.experience("vision/explicit", detach=False) as e:
        e.step(steps.explicit_metrics, 3, name="run")
    found = points("vision/explicit", "run")
    assert found and all("mine" in p for p in found)                       # no `other`
    monkeypatch.setenv("USEML_NO_AUTO_METRICS", "1")
    with exp.experience("vision/off", detach=False) as e:
        e.step(steps.tqdm_train, 3, name="train")
    assert points("vision/off", "train") == []
    assert "epoch=2" in exp.open("vision/off").folder.joinpath("run.log").read_text()


def test_tqdm_is_restored():
    from tqdm import std

    before = std.tqdm.set_postfix
    with exp.experience("vision/restore", detach=False) as e:
        e.step(steps.tqdm_train, 2, name="train")
    assert std.tqdm.set_postfix is before and sys.stdout.__class__.__name__ != "_Watcher"
