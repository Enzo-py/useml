"""Projects: declared in the repository, runs recorded with their factors and results."""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, exp, projects  # noqa: E402

PROJECT = textwrap.dedent("""
    [tool.useml.exp]
    root = "experiences"

    [tool.useml.project]
    name = "robust-vision"
    description = "Classifiers that survive corrupted inputs."
    focus = "wide"
    datasets = ["cifar10"]

    [tool.useml.project.methods.baseline]
    description = "Standard training."
    status = "reference"
    experiences = ["vision/base*"]

    [tool.useml.project.methods.wide]
    description = "Wider blocks."
""")


@pytest.fixture
def project(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "pyproject.toml").write_text(PROJECT)
    monkeypatch.chdir(folder)
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    monkeypatch.setenv("USEML_NO_TMUX", "1")
    monkeypatch.setenv("USEML_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "machine.toml"))
    monkeypatch.setenv("PYTHONPATH", str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(config, "_DEFAULT", None)
    return folder


def run(name, n, **options):
    with exp.experience(name, detach=False, **options) as e:
        trained = e.step(steps.train_with_metrics, n, name="train")
        e.step(steps.summary, [1.0, float(n)], label="final", name="evaluate")
    return trained


def test_declaration(project):
    declared = projects.declared()
    assert declared["name"] == "robust-vision" and declared["focus"] == "wide"
    assert declared["methods"]["baseline"]["experiences"] == ["vision/base*"]
    assert declared["datasets"] == ["cifar10"]


def test_records_hold_factors_results_and_versions(project):
    run("vision/base-a", 4)
    run("vision/base-b", 8)
    run("vision/wide-1", 8, method="wide", tags={"variant": "x2"})
    records = {r["experience"]: r for r in projects.runs()}
    a, b, wide = records["vision/base-a"], records["vision/base-b"], records["vision/wide-1"]
    assert a["method"] == "baseline" and wide["method"] == "wide"           # glob / explicit
    assert a["factors"]["train.steps"] == 4 and b["factors"]["train.steps"] == 8   # named
    assert wide["factors"]["tag.variant"] == "x2"
    assert a["results"]["train.final_loss"] == 0.25
    assert a["results"]["train.loss.best"] == 0.25 and a["results"]["train.accuracy.best"] == 1.0
    assert a["version"] == b["version"]                    # same code, other factors
    assert a["id"] != b["id"] and a["status"] == "done"
    assert a["curves"]["train"][0]["step"] == 1
    assert "def " not in json.dumps(a)                     # never code
    assert "with exp.experience('vision/base-a', method='baseline') as e:" in a["declaration"]
    assert "train = e.step(train_with_metrics, steps=4, name='train')" in a["declaration"]
    assert "from exp_steps import summary, train_with_metrics" in a["declaration"]


def test_factors_of_configs_datasets_and_references():
    spec = {"steps": [
        {"name": "train", "args": [{"$dataset": {"name": "cifar10", "split": "train"}}],
         "kwargs": {"model": {"$config": {"name": "resnet", "kind": "models",
                                          "overrides": {"width": 2, "head.dropout": 0.1}}},
                    "epochs": 10, "milestones": [3, 6], "log": {"$path": "/x"}}},
        {"name": "evaluate", "args": [{"$step": "train"}], "kwargs": {}}]}
    assert projects.factors(spec) == {
        "train.arg0": "cifar10[train]", "train.model": "resnet", "train.model.width": 2,
        "train.model.head.dropout": 0.1, "train.epochs": 10, "train.milestones": "[3, 6]"}
    code = projects.declaration_code(spec | {"steps": [
        {**spec["steps"][0], "function": "pkg.train:fit"},
        {**spec["steps"][1], "function": "pkg.eval:score"}]}, "vision/b", "wide")
    assert "datasets.load('cifar10', split='train')" in code
    assert "config.load('resnet', 'models', overrides={'width': 2, 'head.dropout': 0.1})" in code
    assert "e.step(score, train, name='evaluate')" in code
    shadowed = projects.declaration_code({"steps": [
        {"name": "fit", "function": "pkg:fit", "args": [], "kwargs": {}},
        {"name": "score", "function": "pkg:score", "args": [{"$step": "fit"}], "kwargs": {}}]}, "x")
    assert "fit_out = e.step(fit, name='fit')" in shadowed            # never hides `fit`
    assert "e.step(score, fit_out, name='score')" in shadowed


def test_export_and_cli(project, capsys):
    from useml.cli import main

    run("vision/base-a", 4)
    run("vision/wide-1", 8, method="wide")
    assert main(["project", "show"]) == 0
    out = capsys.readouterr().out
    assert "working on: wide" in out and "baseline" in out
    assert main(["project", "export"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# robust-vision") and "Working on: **wide**" in markdown
    assert "train.steps" in markdown
    assert main(["project", "export", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["project"]["methods"]["wide"]["runs"] == 1 and len(data["runs"]) == 2


def test_agent_records_reach_the_history_once(project, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from useml import monitor
    from useml.monitor import _snapshot

    run("vision/base-a", 4)
    history = tmp_path / "history"
    monkeypatch.setenv("USEML_HISTORY_DIR", str(history))
    monkeypatch.setenv("USEML_MONITOR_TOKEN", "local-secret")
    monkeypatch.delenv("SPACE_ID", raising=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("history_app", monitor.space_folder() / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    agent = monitor.Agent(url="http://testserver", token="local-secret")
    sent = []

    def request(method, path, body=None, timeout=0):
        response = client.request(method, path, content=body,
                                  headers={"X-Useml-Token": "local-secret",
                                           "Content-Type": "application/json"})
        sent.append(json.loads(body) if body else None)
        return response.json()
    monkeypatch.setattr(agent, "_request", request)
    agent.push(full=True)
    assert len(sent[0]["records"]) == 1 and sent[0]["snapshot"]["projects"][0]["focus"] == "wide"
    assert sent[0]["snapshot"]["experiences"][0]["method"] == "baseline"
    kept = list(history.glob("projects/robust-vision/runs/*.json"))
    assert len(kept) == 1 and (history / "projects/robust-vision/project.json").exists()
    agent.push(full=False)
    assert "records" not in sent[1]                        # kept: never sent again
    assert len(_snapshot.pending_records(agent.config, agent.sent)) == 0
    client.get("/?token=local-secret")
    page = client.get("/api/project", params={"name": "robust-vision"}).json()
    assert page["runs"][0]["experience"] == "vision/base-a" and page["history"]["kept"]
    listing = client.get("/api/projects").json()["projects"][0]
    assert listing["runs"] == 1 and listing["methods"] == 2
    # a restarted dashboard reads them back
    module._records.clear(); module._projects.clear(); module._history["loaded"] = False
    assert client.get("/api/projects").json()["projects"][0]["runs"] == 1


def test_records_are_masked_and_need_the_group(project, tmp_path):
    from useml.monitor import _share, _snapshot

    run("vision/base-a", 4)
    config_ = _share.settings()
    records = _snapshot.pending_records(config_, set())
    assert records and str(project) not in json.dumps(records)
    assert _snapshot.pending_records({**config_, "share": ["status"]}, set()) == []


def test_proposed_alternative_runs_as_is(project):
    run("vision/base-a", 4)
    code = projects.record(exp.open("vision/base-a"))["declaration"]
    code = code.replace("'vision/base-a'", "'vision/base-c'").replace("steps=4", "steps=6")
    code = code.replace("method='baseline')", "method='baseline', detach=False)")
    exec(compile(code, "<alternative>", "exec"), {})
    record = projects.record(exp.open("vision/base-c"))
    assert record["status"] == "done" and record["factors"]["train.steps"] == 6
