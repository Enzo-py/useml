"""Runs that must survive their launcher, and crashes that must never be silent."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
sys.path.insert(0, str(TESTS))

import exp_steps as steps  # noqa: E402
from useml import config, exp  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
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
    monkeypatch.setenv("USEML_NO_TMUX", "1")
    monkeypatch.setenv("USEML_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("USEML_CONFIG", str(tmp_path / "machine.toml"))
    monkeypatch.delenv("USEML_EXP_ROOT", raising=False)
    monkeypatch.setattr(config, "_DEFAULT", None)
    return folder


def wait_for(predicate, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


# ── crashes are never silent ──────────────────────────────────────────────────

def test_sigkill_is_named(project):
    with exp.experience("crash/kill") as e:
        e.step(steps.numbers, 3, name="before")
        e.step(steps.sigkill, name="killed")
    assert e.status == "failed"
    why = e.why()
    assert "SIGKILL" in why and "OOM" in why and "about to be killed" in why
    assert (e.folder / "crash.txt").exists()
    assert e.state["exit"]["signal"] == "SIGKILL"
    assert [s["done"] for s in e.steps()] == [True, False]


def test_segfault_leaves_the_python_stack(project):
    with exp.experience("crash/segv") as e:
        e.step(steps.segfault, name="native")
    assert e.status == "failed"
    assert "SIGSEGV" in e.why() and "segmentation fault" in e.why()
    log = (e.folder / "run.log").read_text()
    assert "Fatal Python error" in log and "segfault" in log        # faulthandler's stack


def test_python_error_shows_its_traceback(project):
    with exp.experience("crash/python") as e:
        e.step(steps.boom, name="boom")
    assert e.status == "failed"
    assert "ValueError: a clear Python error" in e.why()


def test_notify_command_runs_on_crash(project, tmp_path):
    out = tmp_path / "notified.txt"
    (tmp_path / "machine.toml").write_text(
        f'[exp]\nnotify = "echo $USEML_EXP_STATUS:$USEML_EXP_NAME > {out}"\n')
    with exp.experience("crash/notify") as e:
        e.step(steps.boom, name="boom")
    assert wait_for(out.exists, 10)
    assert out.read_text().strip() == "failed:crash/notify"


# ── surviving the launcher ────────────────────────────────────────────────────

def test_run_survives_its_launcher_and_a_hangup(project):
    launcher = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(TESTS)!r})
        import exp_steps as steps
        from useml import exp
        with exp.experience("survive/launcher", follow=False) as e:
            e.step(steps.sleepy, 2.0, name="sleep")
        print("launcher exits", flush=True)
    """)
    launched = subprocess.Popen([sys.executable, "-c", launcher], start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    launched.wait(60)
    try:                                           # the launcher's session hangs up
        os.killpg(launched.pid, signal.SIGHUP)
    except OSError:
        pass
    experience = exp.open("survive/launcher")
    assert wait_for(lambda: experience.status == "done", 60), experience.why()
    assert exp.get("survive/launcher/sleep") == {"slept": 2.0}


def test_supervisor_and_run_killed_together_are_reported_as_died(project):
    with exp.experience("survive/reboot", follow=False) as e:
        e.step(steps.sleepy, 30.0, name="sleep")
    assert wait_for(lambda: e.state.get("runtime_pid") and e.state.get("step") == "sleep")
    for key in ("supervisor_pid", "runtime_pid"):
        os.kill(int(e.state[key]), signal.SIGKILL)
    assert wait_for(lambda: e.status == "died", 10)
    assert "disappeared without recording anything" in e.why()


def test_stack_of_a_running_run_and_stop_now(project):
    with exp.experience("survive/stack", follow=False) as e:
        e.step(steps.sleepy, 30.0, name="sleep")
    assert wait_for(lambda: e.state.get("step") == "sleep")
    stacks = []
    assert wait_for(lambda: stacks.append(e.stack(wait=0.5)) or "sleepy" in stacks[-1], 20)
    assert "exp_steps.py" in stacks[-1]
    assert e.status == "running"                    # SIGUSR1 did not stop it
    e.stop(now=True)
    assert wait_for(lambda: e.status == "stopped", 15)
    assert "stopped on request" in e.why()


def test_fake_tmux_backend(project, tmp_path, monkeypatch):
    """A fake `tmux` on PATH: the launch goes through it and the run completes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    fake = bin_dir / "tmux"
    fake.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, subprocess, sys
        args = sys.argv[1:]
        folder = {str(sessions)!r}
        if args[:1] == ["-V"]:
            print("tmux 3.4"); sys.exit(0)
        if args[:1] == ["new-session"]:
            name = args[args.index("-s") + 1]; cwd = args[args.index("-c") + 1]
            p = subprocess.Popen(["sh", "-c", args[-1]], cwd=cwd, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            open(os.path.join(folder, name), "w").write(str(p.pid)); sys.exit(0)
        if args[:1] in (["has-session"], ["kill-session"]):
            name = args[args.index("-t") + 1].lstrip("=")
            path = os.path.join(folder, name)
            if not os.path.exists(path): sys.exit(1)
            pid = int(open(path).read())
            try: os.kill(pid, 0)
            except OSError: sys.exit(1)
            if args[0] == "kill-session": os.kill(pid, 15)
            sys.exit(0)
        sys.exit(2)
    """))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("USEML_NO_TMUX")
    with exp.experience("survive/tmux") as e:
        e.step(steps.numbers, 3, name="numbers")
    assert e.status == "done", e.why()
    assert e.state["backend"] == "tmux" and e.state["session"] == "useml-survive_tmux"


# ── code provenance and bundles ───────────────────────────────────────────────

@pytest.fixture
def git_project(project, monkeypatch):
    """The project as a git repository holding its own step module."""
    module = project / "bundle_steps.py"
    module.write_text((TESTS / "exp_steps.py").read_text())
    (project / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.useml.config]
        root = "configs"
        packages = ["bundle_steps"]
        [tool.useml.exp]
        root = "experiences"
    """))
    (project / ".gitignore").write_text("experiences/\n")
    for command in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"], ["git", "add", "."],
                    ["git", "commit", "-qm", "init"]):
        subprocess.run(command, cwd=project, check=True)
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setenv("PYTHONPATH", f"{project}{os.pathsep}{os.environ['PYTHONPATH']}")
    sys.modules.pop("bundle_steps", None)
    import bundle_steps

    return bundle_steps


def test_code_snapshot_records_commit_diff_and_untracked(git_project, project):
    with open(project / "bundle_steps.py", "a") as file:
        file.write("\n# an uncommitted change\n")
    (project / "helper.py").write_text("VALUE = 1\n")
    with exp.experience("prov/basic", detach=False) as e:
        e.step(git_project.numbers, 2, name="numbers")
    [snapshot] = list((e.folder / "code").iterdir())
    info = json.loads((snapshot / "code.json").read_text())
    assert info["git"]["dirty"] and "helper.py" in info["git"]["untracked"]
    assert "an uncommitted change" in (snapshot / "diff.patch").read_text()
    assert (snapshot / "untracked" / "helper.py").exists()
    assert info["pythonpath"] == ["."]
    done = json.loads((e.folder / "steps" / "numbers" / "done.json").read_text())
    assert done["snapshot"] == snapshot.name


def test_bundle_is_standalone_and_verifiable(git_project, project, tmp_path):
    from useml import data

    @data.processor("bundle-lengths", dtype="int8")
    def lengths(texts):
        import numpy as np
        return np.array([[len(t), -len(t)] for t in texts], dtype=np.float32)

    data.process(data.texts(["a", "bb", "ccc"]), lengths, name="lengths")
    with open(project / "bundle_steps.py", "a") as file:
        file.write("\nDIRTY = True\n")                       # an uncommitted change
    model = config.load("linear")
    with exp.experience("prov/final") as e:
        trained = e.step(git_project.train, model, 2, name="trained")
        e.step(git_project.rows, data.load("lengths"), name="rows")
        e.step(git_project.summary, [1.0, 2.0], name="summary")
    assert e.status == "done", e.why()
    del trained

    target = exp.bundle("prov/final", tmp_path / "archive" / "final")
    assert exp.verify(target) == []
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["code"]["dirty"] and manifest["datasets"][0]["rows"] == 3
    assert "DIRTY = True" in (target / "code" / "bundle_steps.py").read_text()
    assert not list(target.rglob("checkpoints"))

    # A fresh process, from the bundle only: the vendored useml, the bundled code.
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    probe = ("import load, numpy; m = load.get('trained'); "
             "print(load.useml.__file__); print(float(m.weight.sum())); "
             "print(load.get('rows')); print(load.get('summary'))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=target, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0].startswith(str(target))
    assert abs(float(lines[1]) - float((model.weight + 2).sum())) < 1e-4
    assert "'rows': 3" in lines[2] and "'sum': 3.0" in lines[3]

    rerun = subprocess.run([sys.executable, "run.py", "--from-scratch"], cwd=target, env=env,
                           capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stderr[-3000:]
    assert "done" in rerun.stdout

    (target / "code" / "bundle_steps.py").write_text("tampered")
    assert any("changed: code/bundle_steps.py" in p for p in exp.verify(target))


def test_bundle_refuses_unfinished(project):
    with exp.experience("prov/unfinished", run=False) as e:
        e.step(steps.numbers, 2, name="numbers")
    with pytest.raises(exp.BundleError, match="not done"):
        exp.bundle("prov/unfinished")
