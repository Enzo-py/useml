"""useml info / useml mcp: short, read-only views for people and AI assistants."""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, exp, info, mcp, projects  # noqa: E402
from useml.monitor import _share  # noqa: E402

PROJECT = textwrap.dedent("""
    [tool.useml.exp]
    root = "experiences"
    [tool.useml.project]
    name = "robust-vision"
    focus = "wide"
    [tool.useml.project.methods.wide]
    description = "Wider blocks."
    experiences = ["wide/*"]
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
    for name, n in (("wide/a", 4), ("wide/b", 8), ("wide/c", 12)):
        with exp.experience(name, detach=False) as e:
            e.step(steps.train_with_metrics, n, name="train")
    with exp.experience("wide/broken", detach=False, run=False) as e:
        e.step(steps.boom, name="boom")
    with pytest.raises(Exception):
        exp.open("wide/broken").run(detach=False)
    return folder


# ── the machine's identifier ─────────────────────────────────────────────────

def test_alias_survives_a_hostname_change(monkeypatch):
    monkeypatch.setattr(_share, "machine_id", lambda: "HARDWARE-UUID")
    monkeypatch.setattr(platform, "node", lambda: "laptop.local")
    first = _share.default_alias()
    monkeypatch.setattr(platform, "node", lambda: "laptop")
    assert _share.default_alias() == first and "HARDWARE" not in first


# ── what is a factor, which result ranks ─────────────────────────────────────

@pytest.mark.parametrize("key,value,plumbing", [
    ("model.train_sets.coco.fingerprint", "ab12", True),
    ("model.train_sets.coco.path", "x", True),
    ("model.data_dir", "x", True),
    ("model.weights", "/data/w.pt", True),
    ("model.weights", "<project>/w.pt", True),
    ("model.digest", 3, True),
    ("model.run_id", "9f8e7d6c5b4a3f2e", True),
    ("model.lr", 0.001, False),
    ("model.parser", "slot", False),
    ("train.epochs", 10, False),
])
def test_plumbing(key, value, plumbing):
    assert projects.is_plumbing(key, value) is plumbing


def test_main_result_prefers_measures_and_never_counts():
    runs = [["report.n_test", "report.runner_version", "report.accuracy"]] * 3 + [["report.f1"]]
    keys = [k for run in runs for k in run]
    assert projects.main_result(keys) == "report.accuracy"
    assert projects.main_result(["train.loss.last", "train.loss.best"]) == "train.loss.best"
    assert projects.main_result(["report.n_test"]) == "report.n_test"      # nothing else


def test_a_method_declares_its_result(project):
    projects.set_method("wide", result="train.final_loss")
    assert projects.declared()["methods"]["wide"]["result"] == "train.final_loss"
    text = info.project(info.Local(), "robust-vision")
    assert "best train.final_loss=0.08333" in text and "guessed" not in text


# ── the views, on this machine ───────────────────────────────────────────────

def test_overview_is_short_and_points_further(project):
    text = info.overview(info.Local())
    assert "PROJECTS\n  robust-vision · focus wide" in text
    assert "FAILED IN THE LAST 24H (1)" in text and ":wide/broken · failed" in text
    assert "MORE: useml info project" in text and len(text) < 1500


def test_project_summary_then_method_table(project):
    src = info.Local()
    summary = info.project(src, "robust-vision")
    assert "▶ wide · 4 runs" in summary and "LATEST RUNS" in summary and "| " not in summary
    table = info.project(src, "robust-vision", method="wide")
    header = next(line for line in table.splitlines() if line.startswith("  id | run | status | "))
    assert header.split(" | ")[3] == "train.accuracy.best"      # a measure, not train.steps
    assert "train.steps" in table                               # the factor that differs
    with pytest.raises(info.InfoError, match="no method"):
        info.project(src, "robust-vision", method="nope")
    with pytest.raises(info.InfoError, match="no project"):
        info.project(src, "nope")


def test_run_experience_and_machine_views(project):
    src = info.Local()
    run = info.run(src, "wide/c", code=True)
    assert run.startswith("RUN wide/c (id ") and "FACTORS: train.steps=12" in run
    assert "CURVES (first → last):" in run and "train.loss↓: 1 → 0.08333" in run
    assert "DECLARATION (python" in run and "steps=12" in run
    alias = src.alias
    live = info.experience(src, f"{alias}:wide/broken")
    assert "· failed ·" in live and "ERROR:" in live and "LOG (last" in live
    machine = info.machine(src, alias)
    assert machine.startswith(f"MACHINE {alias}") and "experiences:" in machine


def test_cli(project, capsys):
    from useml.cli import main

    assert main(["info", "--local"]) == 0
    assert "robust-vision" in capsys.readouterr().out
    assert main(["info", "project", "robust-vision", "--local", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["project"]["name"] == "robust-vision" and len(data["runs"]) == 4
    assert main(["info", "run", "nothing-like-this", "--local"]) == 2


# ── the same views from the dashboard ────────────────────────────────────────

def test_remote_reads_the_dashboard(project, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from useml import monitor
    from useml.monitor import _snapshot

    monkeypatch.setenv("USEML_MONITOR_TOKEN", "local-secret")
    monkeypatch.delenv("SPACE_ID", raising=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("info_app", monitor.space_folder() / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    config_ = _share.settings()
    body = {"machine": "gpu-a", "full": True, "snapshot": monitor.snapshot(full=True, config=config_),
            "records": _snapshot.pending_records(config_, set())}
    client.post("/agent/push", json=body, headers={"X-Useml-Token": "local-secret"})
    client.get("/?token=local-secret")
    remote = info.Remote(url="http://testserver", token="local-secret")
    monkeypatch.setattr(remote, "get", lambda path, **params: client.get(path, params=params).json()
                        if client.get(path, params=params).status_code == 200 else None)
    text = info.overview(remote)
    assert "gpu-a" in text and "robust-vision" in text
    assert "wide/c" in info.run(remote, "wide/c")


# ── MCP ──────────────────────────────────────────────────────────────────────

def test_mcp_protocol(project, monkeypatch):
    monkeypatch.setattr(info, "source", lambda local=False: info.Local())
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "useml_project", "arguments": {"name": "robust-vision", "method": "wide"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "useml_run", "arguments": {"ref": "missing"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\nnot json\n")
    stdout = io.StringIO()
    saved = sys.stdout
    try:
        assert mcp.serve(stdin, stdout) == 0
    finally:
        sys.stdout = saved
    answers = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [a.get("id") for a in answers] == [1, 2, 3, 4, 5, None]        # no answer to a notification
    assert answers[0]["result"]["protocolVersion"] == "2025-06-18"
    assert {t["name"] for t in answers[1]["result"]["tools"]} == {
        "useml_overview", "useml_project", "useml_run", "useml_experience", "useml_machine"}
    assert "train.steps" in answers[2]["result"]["content"][0]["text"]
    assert answers[2]["result"]["isError"] is False and answers[3]["result"]["isError"] is True
    assert answers[4]["error"]["code"] == -32601 and answers[5]["error"]["code"] == -32700


# ── reports, curves of unknown direction, evolution, resend ──────────────────

def test_a_big_report_gives_its_numbers(project):
    from useml.monitor import snapshot

    with exp.experience("wide/report", detach=False) as e:
        e.step(steps.big_report, 3000, name="report")
    record = projects.record(exp.open("wide/report"))
    assert record["results"]["report.metrics.edge_f1"] == 0.42          # > 64 KB, still read
    sent = next(x for x in snapshot(full=True)["experiences"] if x["name"] == "wide/report")
    report = sent["results"]["report"]
    assert report["__numbers_of_a_report_of__"] > 64 * 1024
    assert report["metrics.edge_f1"] == 0.42 and "per_sample" not in json.dumps(report)


def test_no_best_for_a_curve_of_unknown_direction(project):
    with exp.experience("wide/curves", detach=False) as e:
        e.step(steps.assign_curve, 10, name="train")
    results = projects.record(exp.open("wide/curves"))["results"]
    assert "train.assign.last" in results and "train.assign.best" not in results
    assert results["train.val_acc.best"] == 0.9
    text = info.run(info.Local(), "wide/curves")
    assert "train.assign: 2 → 1.1 (min 1.1 @9, max 2 @0" in text and "train.val_acc↑" in text
    assert not info.trusted("model.assign.best") and info.trusted("val_acc.best")


def test_evolution_lists_each_new_best_and_what_changed(project):
    projects.set_method("wide", result="train.final_loss")
    text = info.project(info.Local(), "robust-vision", method="wide")
    block = text.split("EVOLUTION of train.final_loss")[1].split("VERSIONS")[0]
    assert "a train.final_loss=0.25" in block and "b train.final_loss=0.125 (-0.125 after 0 other runs)" in block
    assert "changed: train.steps 4→8" in block and "c train.final_loss=0.08333" in block


def test_resend_lets_the_dashboard_replace_a_record(project, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from useml import monitor
    from useml.monitor import _agent

    history = Path(os.environ["USEML_ROOT"]) / "history"
    monkeypatch.setenv("USEML_HISTORY_DIR", str(history))
    monkeypatch.setenv("USEML_MONITOR_TOKEN", "local-secret")
    monkeypatch.delenv("SPACE_ID", raising=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("resend_app", monitor.space_folder() / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    agent = monitor.Agent(url="http://testserver", token="local-secret")
    monkeypatch.setattr(agent, "_request", lambda method, path, body=None, timeout=0: client.request(
        method, path, content=body, headers={"X-Useml-Token": "local-secret",
                                             "Content-Type": "application/json"}).json())
    agent.push(full=True)
    kept = len(agent.sent)
    assert kept == 4
    file = next(history.glob("projects/robust-vision/runs/*.json"))
    old = json.loads(file.read_text())
    old["results"] = {}                                   # as an older useml wrote it
    file.write_text(json.dumps(old))
    module._records[old["id"]] = old
    assert monitor.resend() == kept
    agent.push(full=False)                                # sends them again
    assert json.loads(file.read_text())["results"]        # replaced with the full record
