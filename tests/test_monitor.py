"""The monitor: what leaves the machine (and what never does), the relay, the agent."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, exp, monitor  # noqa: E402

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    folder = tmp_path / "secretproject"
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
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    monkeypatch.setenv("USEML_NO_TMUX", "1")
    monkeypatch.setenv("USEML_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "machine.toml"))
    monkeypatch.setenv("PYTHONPATH", str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(config, "_DEFAULT", None)
    return folder


def machine_config(tmp_path, text: str) -> None:
    (tmp_path / "machine.toml").write_text(textwrap.dedent(text))


@pytest.fixture
def experiences(project):
    pytest.importorskip("torch")
    with exp.experience("vision/run") as e:              # detached: run.log is written
        e.step(steps.train_with_metrics, 5, name="train")
        e.step(steps.figure, name="plot")
        e.step(steps.summary, [1.0, 2.0], label="final", name="summary")
    with exp.experience("vision/broken", detach=False, run=False) as e:
        e.step(steps.boom, name="boom")
    with pytest.raises(ValueError):
        e.run(detach=False)
    with exp.experience("other/model", detach=False) as e:
        e.step(steps.identity, config.load("linear"), name="model")
    return project


# ── what leaves the machine ───────────────────────────────────────────────────

def test_full_snapshot_has_every_group(experiences):
    snap = monitor.snapshot()
    names = {e["name"]: e for e in snap["experiences"]}
    run = names["vision/run"]
    assert run["status"] == "done" and run["done"] == run["total"] == 3
    assert [s["function"] for s in run["steps"]][0] == "exp_steps:train_with_metrics"
    assert [p["loss"] for p in run["metrics"]["train"]][:2] == [1.0, 0.5]
    assert run["results"]["summary"]["sum"] == 3.0
    assert run["figures"] and run["figures"][0]["png"]
    assert any("trained in" in line for line in run["log"])
    broken = names["vision/broken"]
    assert broken["status"] == "failed" and "ValueError" in broken["error"]["message"]
    other = names["other/model"]
    assert other["configs"][0]["name"] == "linear" and "type: Linear" in other["configs"][0]["yaml"]
    assert snap["machine"]["alias"].startswith("machine-")


def test_masking_leaves_no_path_host_or_user(experiences, tmp_path):
    text = json.dumps(monitor.snapshot())
    assert "<project>" in text                         # the step printed its folder
    for secret in (str(tmp_path), str(experiences), str(Path.home()), platform.node(),
                   "secretproject"):
        assert secret not in text, secret
    import getpass

    assert getpass.getuser() not in text


def test_code_and_packages_never_leave(experiences):
    text = json.dumps(monitor.snapshot())
    source = (TESTS / "exp_steps.py").read_text()
    assert "def train_with_metrics" not in text and "ctx.metric(step" not in source[:0] + text
    assert "packages.txt" not in text and "numpy==" not in text and "diff.patch" not in text


def test_share_restricts_what_is_sent(experiences, tmp_path):
    machine_config(tmp_path, """
        [monitor]
        machine = "gpu-a"
        share = ["status", "progress"]
        experiences = ["vision/*"]
    """)
    snap = monitor.snapshot()
    assert {e["name"] for e in snap["experiences"]} == {"vision/run", "vision/broken"}
    for entry in snap["experiences"]:
        assert set(entry) <= {"name", "status", "updated", "done", "total", "current_step",
                              "progress"}
    assert set(snap["machine"]) == {"alias", "useml", "time"} and snap["machine"]["alias"] == "gpu-a"


def test_redact_and_unknown_groups(experiences, tmp_path):
    machine_config(tmp_path, '[monitor]\nredact = ["final"]\n')
    assert "final" not in json.dumps(monitor.snapshot())
    machine_config(tmp_path, '[monitor]\nshare = ["status", "code"]\n')
    with pytest.raises(ValueError, match="unknown groups \\['code'\\]"):
        monitor.snapshot()


def test_light_snapshot_has_no_heavy_fields(experiences):
    for entry in monitor.snapshot(full=False)["experiences"]:
        assert not {"results", "figures", "configs"} & set(entry)


# ── the relay ─────────────────────────────────────────────────────────────────

def load_app(monkeypatch, *, space: bool = False, owner: str = "alice"):
    if space:
        monkeypatch.setenv("SPACE_ID", f"{owner}/useml-monitor")
        monkeypatch.setenv("SPACE_AUTHOR_NAME", owner)
        monkeypatch.delenv("USEML_MONITOR_TOKEN", raising=False)
    else:
        monkeypatch.delenv("SPACE_ID", raising=False)
        monkeypatch.setenv("USEML_MONITOR_TOKEN", "local-secret")
    spec = importlib.util.spec_from_file_location(f"app_{time.time_ns()}",
                                                  monitor.space_folder() / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_mode_requires_the_token(monkeypatch):
    client = TestClient(load_app(monkeypatch).app)
    assert client.get("/").status_code == 403
    assert client.get("/api/state").status_code == 403
    assert client.get("/?token=local-secret").status_code == 200
    assert client.get("/api/state").status_code == 200            # the cookie is set now
    assert client.post("/agent/push", json={}).status_code == 403
    bad = {"X-Useml-Token": "nope"}
    assert client.post("/agent/push", json={}, headers=bad).status_code == 403


def test_space_mode_accepts_only_the_owners_token(monkeypatch):
    module = load_app(monkeypatch, space=True)
    calls = []
    module._whoami = lambda token: calls.append(token) or (
        {"name": "alice"} if token == "good" else {"name": "mallory", "orgs": []})
    client = TestClient(module.app)
    push = {"machine": "m", "full": True, "snapshot": {"experiences": []}}
    assert client.post("/agent/push", json=push, headers={"X-Useml-Token": "bad"}).status_code == 403
    for _ in range(3):
        assert client.post("/agent/push", json=push, headers={"X-Useml-Token": "good"}).status_code == 200
    assert calls.count("good") == 1                               # checked once, then cached
    assert client.get("/api/state").json()["machines"][0]["alias"] == "m"


def test_light_push_keeps_the_heavy_fields(monkeypatch):
    client = TestClient(load_app(monkeypatch).app)
    headers = {"X-Useml-Token": "local-secret"}
    full = {"name": "e", "status": "running", "results": {"s": 1}, "figures": [1]}
    client.post("/agent/push", headers=headers,
                json={"machine": "m", "full": True, "snapshot": {"experiences": [full]}})
    client.post("/agent/push", headers=headers,
                json={"machine": "m", "full": False,
                      "snapshot": {"experiences": [{"name": "e", "status": "done"}]}})
    state = client.get("/api/state?token=local-secret").json()
    entry = state["machines"][0]["snapshot"]["experiences"][0]
    assert entry["status"] == "done" and entry["results"] == {"s": 1}


# ── end to end: a real server, a real agent ───────────────────────────────────

@pytest.fixture
def served(monkeypatch):
    import socket

    import uvicorn

    module = load_app(monkeypatch)
    module.WAIT_SECONDS = 2.0
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(module.app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def test_agent_answers_a_refresh_within_seconds(experiences, served):
    import urllib.request

    agent = monitor.Agent(url=served, token="local-secret")
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    try:
        def get(path, data=None):
            request = urllib.request.Request(f"{served}{path}", data=data,
                                             headers={"Cookie": "useml_monitor=local-secret",
                                                      "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read())

        deadline = time.time() + 15
        while not get("/api/state")["machines"] and time.time() < deadline:
            time.sleep(0.1)
        first = get("/api/state")["machines"][0]["received"]
        time.sleep(0.2)
        started = time.time()
        state = get("/api/refresh", data=b"{}")
        latency = time.time() - started
        machine = state["machines"][0]
        assert machine["received"] > first                         # a fresh snapshot
        assert latency < 5.0, latency
        names = {e["name"] for e in machine["snapshot"]["experiences"]}
        assert "vision/run" in names
    finally:
        agent.running = False
        thread.join(10)


# ── deploying ─────────────────────────────────────────────────────────────────

class FakeApi:
    created, uploaded, variables = [], [], []
    public = False
    public_dataset = False

    def __init__(self, token=None):
        pass

    def whoami(self):
        return {"name": "alice"}

    def create_repo(self, repo_id, **options):
        FakeApi.created.append((repo_id, options))

    def space_info(self, repo_id):
        return type("Info", (), {"private": not FakeApi.public})()

    def upload_folder(self, **options):
        FakeApi.uploaded.append(options)

    def dataset_info(self, repo_id):
        return type("Info", (), {"private": not FakeApi.public_dataset})()

    def add_space_variable(self, repo_id, key, value):
        FakeApi.variables.append((repo_id, key, value))


@pytest.fixture
def fake_hub(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    FakeApi.created, FakeApi.uploaded, FakeApi.variables = [], [], []
    FakeApi.public = FakeApi.public_dataset = False
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return FakeApi


def test_deploy_creates_a_private_docker_space(fake_hub, tmp_path):
    result = monitor.deploy()
    repo_id, options = fake_hub.created[0]
    assert repo_id == "alice/useml-monitor" and options["private"] is True
    assert options["space_sdk"] == "docker"
    assert set(fake_hub.uploaded[0]["allow_patterns"]) == {"app.py", "index.html",
                                                          "Dockerfile", "README.md"}
    assert result["url"] == "https://alice-useml-monitor.hf.space"
    assert 'space = "alice/useml-monitor"' in (tmp_path / "machine.toml").read_text()
    assert monitor.status()["dashboard"] == "https://alice-useml-monitor.hf.space"
    dataset, options = fake_hub.created[1]
    assert dataset == "alice/useml-monitor-history" and options == {"repo_type": "dataset",
                                                                     "private": True,
                                                                     "exist_ok": True}
    assert fake_hub.variables == [("alice/useml-monitor", "USEML_HISTORY_REPO",
                                   "alice/useml-monitor-history")]
    assert "USEML_HISTORY_TOKEN" in " ".join(result["history_steps"])


def test_deploy_refuses_a_public_history(fake_hub):
    fake_hub.public_dataset = True
    with pytest.raises(RuntimeError, match="PUBLIC"):
        monitor.deploy()
    assert fake_hub.variables == []


def test_deploy_without_history(fake_hub):
    assert monitor.deploy(history=False)["history"] is None
    assert len(fake_hub.created) == 1


def test_deploy_refuses_a_public_space(fake_hub):
    fake_hub.public = True
    with pytest.raises(RuntimeError, match="PUBLIC"):
        monitor.deploy("alice/open")
    assert fake_hub.uploaded == []


def test_space_url_rule():
    assert monitor.space_url("Some_User/my.board") == "https://some-user-my-board.hf.space"


def test_cli_snapshot(experiences, capsys):
    from useml.cli import main

    assert main(["monitor", "snapshot", "--light"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["full"] is False and data["experiences"]


def test_setup_group_in_full_snapshots_only(experiences, tmp_path):
    machine_config(tmp_path, '[exp]\nnotify = "curl https://hooks.example/secret-token"\n')
    full = monitor.snapshot()
    setup = full["setup"]
    assert setup["experiences"] == {"done": 2, "failed": 1}
    assert {"store", "processed", "datasets", "downloads", "policy", "settings"} <= set(setup)
    notify = next(s for s in setup["settings"] if s["key"] == "exp.notify")
    assert notify["value"] == "<set>" and "secret-token" not in json.dumps(full)
    assert "setup" not in monitor.snapshot(full=False)


def test_space_keeps_the_setup_on_light_pushes(monkeypatch):
    client = TestClient(load_app(monkeypatch).app)
    headers = {"X-Useml-Token": "local-secret"}
    client.post("/agent/push", headers=headers, json={"machine": "m", "full": True,
                "snapshot": {"experiences": [], "setup": {"useml": "x"}}})
    client.post("/agent/push", headers=headers, json={"machine": "m", "full": False,
                "snapshot": {"experiences": []}})
    state = client.get("/api/state?token=local-secret").json()
    assert state["machines"][0]["snapshot"]["setup"] == {"useml": "x"}


# ── the page's requests: light state, details on demand, light refresh ───────

def _push(client, alias, snapshot, full=True):
    return client.post("/agent/push", headers={"X-Useml-Token": "local-secret"},
                       json={"machine": alias, "full": full, "snapshot": snapshot})


def test_light_state_leaves_heavy_parts_for_the_detail_requests(monkeypatch):
    client = TestClient(load_app(monkeypatch).app)
    client.get("/?token=local-secret")
    heavy = {"name": "e", "status": "done", "log": ["x"], "results": {"s": 1},
             "figures": [{"png": "AAAA"}], "configs": [{"name": "c"}]}
    _push(client, "m", {"experiences": [heavy], "setup": {"useml": "1"}})
    light = client.get("/api/state?light=1").json()["machines"][0]["snapshot"]
    assert light["experiences"] == [{"name": "e", "status": "done", "log": ["x"]}]
    assert light["has_setup"] is True and "setup" not in light
    assert client.get("/api/state").json()["machines"][0]["snapshot"]["setup"] == {"useml": "1"}
    assert client.get("/api/experience", params={"machine": "m", "name": "e"}).json() == heavy
    assert client.get("/api/experience", params={"machine": "m", "name": "?"}).status_code == 404
    assert client.get("/api/setup", params={"machine": "m"}).json() == {"useml": "1"}
    assert client.get("/api/setup", params={"machine": "?"}).status_code == 404


def test_forget_removes_a_machine_from_memory_only(monkeypatch):
    client = TestClient(load_app(monkeypatch).app)
    client.get("/?token=local-secret")
    _push(client, "old-name", {"experiences": []})
    assert client.post("/api/forget", json={"machine": "old-name"}).json() == {"ok": True}
    assert client.get("/api/state").json()["machines"] == []
    assert client.post("/api/forget", json={}).status_code == 400


def test_refresh_kind_reaches_the_agent_and_silent_machines_do_not_slow_it(monkeypatch):
    module = load_app(monkeypatch)
    module.WAIT_SECONDS = 0.2
    client = TestClient(module.app)
    client.get("/?token=local-secret")
    _push(client, "quiet", {"experiences": []})
    module._machines["quiet"]["polled"] = time.time() - 3600           # its agent is gone
    started = time.time()
    client.post("/api/refresh", json={"full": False})
    assert time.time() - started < 1.0
    _push(client, "m", {"experiences": []})
    module._pending["m"] = "light"
    answer = client.get("/agent/wait?machine=m", headers={"X-Useml-Token": "local-secret"})
    assert answer.json() == {"command": "refresh", "full": False}
    module._pending["m"] = "full"
    answer = client.get("/agent/wait?machine=m", headers={"X-Useml-Token": "local-secret"})
    assert answer.json() == {"command": "refresh", "full": True}


def test_agent_pushes_the_kind_it_is_asked(experiences, monkeypatch):
    agent = monitor.Agent(url="http://127.0.0.1:1", token="t")
    pushed = []
    monkeypatch.setattr(agent, "push", lambda *, full: pushed.append(full))
    for answer in ({"command": "refresh", "full": False}, {"command": "refresh"}):
        monkeypatch.setattr(agent, "_request", lambda *a, answer=answer, **k: answer)
        assert agent.step() == "refresh"
    assert pushed == [False, True]                    # an old dashboard means full


def test_machine_label_is_sent(experiences, tmp_path):
    machine_config(tmp_path, '[monitor]\nlabel = "GPU server"\n')
    assert monitor.snapshot(full=False)["machine"]["label"] == "GPU server"


def test_setup_carries_the_checks_and_light_state_their_count(experiences, monkeypatch):
    setup = monitor.snapshot()["setup"]
    assert setup["checks"] and {"level", "area", "message", "fix"} == set(setup["checks"][0])
    client = TestClient(load_app(monkeypatch).app)
    client.get("/?token=local-secret")
    checks = [{"level": "fail", "area": "x", "message": "m", "fix": "f"},
              {"level": "warn", "area": "x", "message": "m", "fix": None},
              {"level": "ok", "area": "x", "message": "m", "fix": None}]
    _push(client, "m", {"experiences": [], "setup": {"checks": checks}})
    light = client.get("/api/state?light=1").json()["machines"][0]["snapshot"]
    assert light["health"] == {"warn": 1, "fail": 1}


def test_stop_waits_for_the_agent_so_start_works_right_after(experiences, served, tmp_path,
                                                            monkeypatch):
    """The agent sits in a long poll (25 s on a real dashboard): stop must not leave it
    half-alive, or `start` right after believes it still runs and starts nothing."""
    from useml.monitor import _agent

    machine_config(tmp_path, f'[monitor]\nurl = "{served}"\n')
    monkeypatch.setenv("HF_TOKEN", "local-secret")
    first = monitor.start()
    deadline = time.time() + 20
    while not _agent.agent_pid() and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(1.5)                                    # now inside a long poll
    started = time.time()
    assert monitor.stop() is True
    assert time.time() - started < 5 and _agent.agent_pid() is None
    second = monitor.start()
    try:
        assert not second.get("already") and second["pid"] != first["pid"]
    finally:
        monitor.stop()
