from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, data, exp  # noqa: E402
from useml.exp import RefError  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    """A project folder: pyproject with configs and experiences, the cwd of every run."""
    folder = tmp_path / "project"
    (folder / "configs" / "models").mkdir(parents=True)
    (folder / "configs" / "models" / "linear.yaml").write_text("type: Linear\nseed: 1\n")
    (folder / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.useml.config]
        root = "configs"
        packages = ["exp_steps"]
        [tool.useml.exp]
        root = "experiences"
    """))
    monkeypatch.chdir(folder)
    monkeypatch.setenv("PYTHONPATH", str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    monkeypatch.setenv("USEML_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("USEML_TEST_CALLS", str(tmp_path / "calls.log"))
    monkeypatch.delenv("USEML_EXP_ROOT", raising=False)
    monkeypatch.setattr(config, "_DEFAULT", None)
    return folder


def calls(tmp_path) -> list:
    path = tmp_path / "calls.log"
    return path.read_text().split() if path.exists() else []


def _three_steps(e, n=5, label="x"):
    values = e.step(steps.numbers, n, name="numbers")
    squared = e.step(steps.square, values, name="square")
    e.step(steps.summary, squared, label=label, name="summary")


# ── declaring and running ─────────────────────────────────────────────────────

def test_inline_run_and_one_line_retrieval(project):
    with exp.experience("demo/basic", detach=False) as e:
        _three_steps(e)
    assert e.status == "done"
    assert np.array_equal(exp.get("demo/basic/square"), np.arange(5) ** 2)
    assert exp.get("demo/basic", "summary") == {"label": "x", "sum": 30.0, "count": 5}
    assert exp.open("demo/basic").results() == {"summary": {"label": "x", "sum": 30.0,
                                                            "count": 5}}
    assert (project / "experiences" / "demo" / "basic" / "steps" / "square" /
            "done.json").exists()


def test_running_the_block_again_does_nothing(project, tmp_path):
    for _ in range(2):
        with exp.experience("demo/again", detach=False) as e:
            _three_steps(e)
    assert calls(tmp_path) == ["numbers", "square", "summary"]


def test_changing_one_step_reruns_only_what_depends_on_it(project, tmp_path):
    with exp.experience("demo/chain", detach=False) as e:
        _three_steps(e, label="a")
    with exp.experience("demo/chain", detach=False) as e:
        _three_steps(e, label="b")                         # only `summary` changed
    assert calls(tmp_path) == ["numbers", "square", "summary", "summary"]
    assert exp.get("demo/chain/summary")["label"] == "b"
    stale = list((project / "experiences" / "demo" / "chain" / "steps").glob("summary.stale-*"))
    assert len(stale) == 1
    with exp.experience("demo/chain", detach=False) as e:
        _three_steps(e, n=6, label="b")                    # the first step changed: all rerun
    assert calls(tmp_path)[-3:] == ["numbers", "square", "summary"]


def test_detached_run_followed(project):
    with exp.experience("demo/detached") as e:
        _three_steps(e)
    assert e.status == "done", e.log()
    assert exp.get("demo/detached/summary")["sum"] == 30.0
    assert "already done" not in e.log()


def test_failure_then_resume(project, tmp_path):
    marker = tmp_path / "marker"
    with exp.experience("demo/fails", detach=False, run=False) as e:
        values = e.step(steps.numbers, 3, name="numbers")
        e.step(steps.fail_unless, str(marker), name="check")
        e.step(steps.summary, values, name="summary")
    with pytest.raises(RuntimeError, match="marker missing"):
        e.run(detach=False)
    assert e.status == "failed" and "marker missing" in e.state["error"]
    marker.touch()
    e.run(detach=False)
    assert e.status == "done"
    assert calls(tmp_path) == ["numbers", "fail_unless", "fail_unless", "summary"]


def test_clean_stop_between_steps(project):
    with exp.experience("demo/stop", follow=False) as e:
        e.step(steps.slow, 30.0, name="slow")
        e.step(steps.summary, [1.0], name="after")
    deadline = time.time() + 20
    while e.state.get("step") != "slow" and time.time() < deadline:
        time.sleep(0.1)
    e.stop()
    e.follow(poll=0.2)
    assert e.status == "stopped"
    assert exp.get("demo/stop/slow") == {"stopped": True}
    assert [s["done"] for s in e.steps()] == [True, False]


# ── what can be declared ──────────────────────────────────────────────────────

def test_declaration_errors_fall_on_the_line(project):
    with pytest.raises(RefError, match="notebook cell"):
        with exp.experience("demo/bad") as e:
            e.step(lambda: 1)
    with pytest.raises(RefError, match="cannot be rebuilt"):
        with exp.experience("demo/bad") as e:
            e.step(steps.identity, object())
    with pytest.raises(RefError, match="two steps named"):
        with exp.experience("demo/bad") as e:
            e.step(steps.numbers, 1, name="a")
            e.step(steps.numbers, 2, name="a")
    with pytest.raises(RuntimeError, match="only valid inside"):
        builder = exp.experience("demo/bad")
        builder.step(steps.numbers, 1)
    assert not (project / "experiences" / "demo" / "bad" / "experience.json").exists()


def test_processed_datasets_are_passed_by_name(project):
    from useml import data as d

    feats = d.process(d.texts(["a", "bb", "ccc"]), steps_length(), name="lengths")
    with exp.experience("demo/data", detach=False) as e:
        e.step(steps.rows, d.load("lengths"), name="rows")
    assert exp.get("demo/data/rows") == {"rows": 3, "first": [1.0]}
    unnamed = d.process(d.texts(["z"]), steps_length())
    with pytest.raises(RefError, match="has no name"):
        with exp.experience("demo/data2") as e:
            e.step(steps.rows, unnamed)
    assert len(feats) == 3


def steps_length():
    import data_procs  # noqa: F401  (registers nothing we need; keeps import paths alike)

    @data.processor("text-length", dtype="float32")
    def length(texts):
        return np.array([[len(text)] for text in texts], dtype=np.float32)
    return length


# ── models: config + weights ──────────────────────────────────────────────────

def test_trained_config_model_comes_back_with_its_weights(project):
    model = config.load("linear")
    before = model.weight.detach().clone()
    with exp.experience("demo/train") as e:
        e.step(steps.train, model, 3, name="trained")
    assert e.status == "done", e.log()
    trained = exp.get("demo/train/trained")
    assert isinstance(trained, steps.Linear)
    assert torch.allclose(trained.weight, before + 3)
    assert torch.allclose(config.load("linear").weight, before)    # the memo is untouched
    folder = exp.open("demo/train").path("trained")
    assert (folder / "weights.pt").exists() and (folder / "checkpoints" / "index.json").exists()


def test_a_config_object_modified_by_hand_is_refused(project):
    model = config.load("linear", seed=4)
    model.seed = 99
    with pytest.raises(RefError, match="modified after loading.*seed"):
        with exp.experience("demo/modified") as e:
            e.step(steps.identity, model)


def test_output_of_another_experience(project, tmp_path):
    with exp.experience("demo/first", detach=False) as e:
        e.step(steps.numbers, 4, name="numbers")
    values = exp.get("demo/first/numbers")
    with exp.experience("demo/second", detach=False) as e:
        e.step(steps.summary, values, name="summary")
    assert exp.get("demo/second/summary")["sum"] == 6.0
    spec = json.loads((project / "experiences" / "demo" / "second" / "experience.json")
                      .read_text())
    assert spec["steps"][0]["args"][0]["$output"]["experience"] == "demo/first"


def test_figures_are_saved_as_png(project):
    pytest.importorskip("matplotlib")
    with exp.experience("demo/figure", detach=False) as e:
        e.step(steps.figure, name="plot")
    out = exp.get("demo/figure/plot")
    assert out["value"] == 3 and Path(out["plot"]).suffix == ".png" and Path(out["plot"]).exists()


# ── finding experiences ───────────────────────────────────────────────────────

def test_ls_open_and_cli(project, capsys):
    from useml.cli import main

    with exp.experience("demo/listed", detach=False) as e:
        _three_steps(e)
    assert [entry.name for entry in exp.ls()] == ["demo/listed"]
    with pytest.raises(KeyError, match="Did you mean demo/listed"):
        exp.open("demo/listd")
    assert main(["exp", "ls"]) == 0
    assert main(["exp", "show", "demo/listed"]) == 0
    out = capsys.readouterr().out
    assert "done" in out and "✓ square" in out
