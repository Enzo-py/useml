"""Experiences: declared as a script, run as a declaration, read back in one line.

    from useml import exp, config, data

    with exp.experience("vision/baseline") as e:
        feats = data.load("photos-clip")                         # processed ONCE, elsewhere
        trained = e.step(train, feats, epochs=10, name="model")
        e.step(report, trained, config.load("resnet_ft"), name="report")
    # the block ends: declaration written, run DETACHED, followed with a bar

    model = exp.get("vision/baseline/model")             # anywhere, later: one line

THE BLOCK RUNS NOTHING. Each `e.step(fn, *args)` RECORDS a call and returns a reference
(`StepRef`). Arguments become recipes (`_refs`); errors fall on the line that causes them.
At the end of the block, the declaration is fingerprinted, written, and launched.

RUNNING THE BLOCK AGAIN IS SAFE: steps already done (same fingerprint) are skipped;
changing one line redoes only the steps that depend on it. A running experience with the
same declaration is re-attached; with another declaration, it raises.

WHERE: `[tool.useml.exp] root` in the project's pyproject (relative to it), else
`[exp] root` in the machine config, else `<home>/experiences`. An experience is named by
its path under the root: "vision/baseline".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .. import _progress
from . import _outputs
from ._refs import (RefError, StepRef, code_hash, freeze, importable, recipe, set_origin,
                    steps_read)
from ._runtime import DIED, DONE, NEW, RUNNING, is_done, read_state, write_state

#: Written in every declaration: experiences of other systems sharing the folder (an
#: `experience.json` without it) are left alone.
FORMAT = "useml-exp/1"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*(/[A-Za-z0-9][A-Za-z0-9_.@+-]*)*$")


# ── where experiences live ────────────────────────────────────────────────────

def root() -> Path:
    explicit = os.environ.get("USEML_EXP_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    from .._project import section

    folder, settings = section("exp")
    if settings is not None:
        return folder / settings.get("root", "experiences")
    from .. import paths

    configured = paths.config().get("exp", {}).get("root")
    if configured:
        return Path(os.path.expandvars(str(configured))).expanduser()
    return paths.home() / "experiences"


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError(f"experience name {name!r}: parts of letters, digits, _ . @ + -, "
                         f"separated by '/'")
    return name


# ── one experience on disk ────────────────────────────────────────────────────

class Experience:
    def __init__(self, folder, name: str | None = None) -> None:
        self.folder = Path(folder)
        self.name = name or self.folder.name

    # ── what is on disk ──────────────────────────────────────────────────────

    @property
    def spec(self) -> dict | None:
        path = self.folder / "experience.json"
        return json.loads(path.read_text()) if path.exists() else None

    @property
    def state(self) -> dict:
        from ._launch import live_status

        state = read_state(self.folder)
        state["status"] = live_status(self.folder, state) if state else NEW
        return state

    @property
    def status(self) -> str:
        if self.spec is None:
            return NEW
        return self.state.get("status", NEW)

    def steps(self) -> list[dict]:
        """Each step: name, function, done (with the current fingerprint), seconds."""
        out = []
        for step in (self.spec or {}).get("steps", []):
            done_path = self.folder / "steps" / step["name"] / "done.json"
            done = json.loads(done_path.read_text()) if done_path.exists() else {}
            out.append({"name": step["name"], "function": step["function"],
                        "done": done.get("fingerprint") == step["fingerprint"],
                        "seconds": done.get("seconds"),
                        "output": (done.get("output") or {}).get("kind")})
        return out

    def log(self, lines: int = 40) -> str:
        path = self.folder / "run.log"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) \
            if path.exists() else ""

    # ── reading outputs ──────────────────────────────────────────────────────

    def get(self, step: str):
        """The output of `step`, loaded from disk (lazily: nothing is read before)."""
        names = [entry["name"] for entry in (self.spec or {}).get("steps", [])]
        if step not in names:
            raise KeyError(f"{self.name}: no step {step!r}; steps: {', '.join(names) or 'none'}")
        declared = next(s for s in self.spec["steps"] if s["name"] == step)
        done_path = self.folder / "steps" / step / "done.json"
        if not done_path.exists():
            raise RuntimeError(f"{self.name}/{step} has not run yet (status: {self.status}); "
                               f"exp.follow() waits for it")
        done = json.loads(done_path.read_text())
        if done.get("fingerprint") != declared["fingerprint"]:
            raise RuntimeError(f"{self.name}/{step}: the output on disk comes from an older "
                               f"declaration of this step; run the experience again")
        with _progress.step(f"loading {self.name}/{step}"):
            value = _outputs.load(self.folder / "steps" / step, done["output"])
        set_origin(value, {"experience": self.name, "step": step,
                           "fingerprint": done["fingerprint"]})
        return value

    def results(self) -> dict:
        """{step: output} for every finished step whose output is JSON (scores, reports):
        the numbers, without loading any model."""
        out = {}
        for entry in self.steps():
            if entry["done"] and entry["output"] in ("json", "figure", "none"):
                out[entry["name"]] = self.get(entry["name"])
        return out

    def path(self, step: str) -> Path:
        """The folder of `step` (what it wrote, checkpoints included)."""
        return self.folder / "steps" / step

    # ── running ──────────────────────────────────────────────────────────────

    def run(self, *, detach: bool = True, follow: bool = True, gpu=None) -> "Experience":
        """Run the steps not done yet. Detached by default (survives the notebook)."""
        if self.spec is None:
            raise RuntimeError(f"{self.name}: nothing declared yet")
        status = self.status
        if status == RUNNING:
            _progress.note(f"{self.name} is already running; following it")
            return self.follow() if follow else self
        todo = [entry["name"] for entry in self.steps() if not entry["done"]]
        if not todo:
            _progress.note(f"{self.name} is up to date: every step already ran. "
                           f"exp.get('{self.name}/<step>') reads an output.")
            write_state(self.folder, status=DONE)
            return self
        done = [entry["name"] for entry in self.steps() if entry["done"]]
        _progress.note(f"{self.name}: {len(todo)} step(s) to run {todo}"
                       + (f", {len(done)} already done" if done else ""))
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu) \
                if isinstance(gpu, (list, tuple)) else str(gpu)
        from ._provenance import record_code

        if not detach:
            from ._runtime import execute

            record_code(self.folder, self.spec)
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ.update({k: v for k, v in env.items() if k == "CUDA_VISIBLE_DEVICES"})
            from ._capture import tee_log

            with open(self.folder / "run.log", "ab") as log:
                log.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} (in this process) ===\n".encode())
            try:
                with tee_log(self.folder / "run.log"):        # the dashboard reads it
                    execute(self.folder)
            finally:
                if gpu is not None:
                    if previous is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = previous
            return self
        from ._launch import launch

        with open(self.folder / "run.log", "ab") as log:
            log.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        record_code(self.folder, self.spec)
        started = launch(self.folder, self.name, env)
        where = (f"tmux session {started['session']} (useml exp attach {self.name})"
                 if started["backend"] == "tmux" else f"detached, pid {started['pid']}")
        _progress.note(f"started {self.name}: {where}. It survives this notebook and a "
                       f"closed laptop; log: {self.folder / 'run.log'}")
        return self.follow() if follow else self

    def why(self) -> str:
        """Where the run is — and, if it ended badly, why (signal, OOM, traceback, last
        log lines). Also `useml exp why <name>`."""
        from ._launch import why

        return why(self.folder, self.name, self.log(30))

    def stack(self, *, wait: float = 1.5) -> str:
        """The Python stack of every thread of the RUNNING run, without stopping it: where
        a run that seems stuck is."""
        import signal as signals

        pid = read_state(self.folder).get("runtime_pid")
        if self.status != RUNNING or not pid:
            raise RuntimeError(f"{self.name} is not running")
        log_path = self.folder / "run.log"
        position = log_path.stat().st_size
        os.kill(int(pid), signals.SIGUSR1)
        time.sleep(wait)
        with open(log_path, "rb") as file:
            file.seek(position)
            return file.read().decode(errors="replace")

    def attach(self) -> None:
        """Open the run's tmux terminal (from a shell: `useml exp attach <name>`)."""
        session = read_state(self.folder).get("session")
        if not session:
            raise RuntimeError(f"{self.name} does not run in tmux; `useml exp follow "
                               f"{self.name}` shows its log")
        if not sys.stdin.isatty():
            print(f"in a terminal: tmux attach -t {session}")
            return
        os.execvp("tmux", ["tmux", "attach", "-t", f"={session}"])

    def follow(self, *, poll: float = 1.0) -> "Experience":
        """A bar over the steps, the log as it comes. Ctrl-C stops FOLLOWING only."""
        steps = [entry["name"] for entry in self.steps()]
        bar = _progress.bar(total=len(steps), desc=self.name, unit="step", delay=0)
        log_path = self.folder / "run.log"
        position = log_path.stat().st_size if log_path.exists() else 0
        try:
            while True:
                if log_path.exists():
                    with open(log_path, "rb") as file:
                        file.seek(position)
                        chunk = file.read()
                        position += len(chunk)
                    for line in chunk.decode(errors="replace").splitlines():
                        if line.strip() and "\r" not in line:
                            bar.write(line)
                state = self.state
                bar.update(len(state.get("steps_done") or []) - bar.n)
                progress = state.get("step_progress")
                current = state.get("step") or ""
                bar.set_postfix_str(f"{current} {progress[0]}/{progress[1]} {progress[2]}"
                                    if progress else current)
                if state.get("status") != RUNNING:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            bar.close()
            _progress.note(f"stopped following; {self.name} keeps running "
                           f"(exp.open('{self.name}').follow() to come back)")
            return self
        bar.close()
        # The supervisor writes its verdict a moment after the run ends.
        deadline = time.time() + 5
        while self.state.get("exit") is None and time.time() < deadline and \
                read_state(self.folder).get("supervisor_pid"):
            time.sleep(0.2)
        if self.state.get("status") != DONE:
            _progress.note(self.why())
        return self

    def stop(self, *, now: bool = False) -> "Experience":
        """Clean stop between steps (and in steps that check `ctx.should_stop()`), or with
        `now=True` kill the run at once (nothing of the current step is saved)."""
        state = self.state
        if state.get("status") != RUNNING:
            return self
        if now:
            write_state(self.folder, stop_requested=True)
            pid = state.get("runtime_pid") or state.get("pid")
            try:
                os.kill(int(pid), signal.SIGTERM)       # the supervisor records the end
            except (OSError, TypeError, ValueError):
                write_state(self.folder, status="stopped", error="killed on request")
        else:
            (self.folder / "control").write_text("stop")
            _progress.note(f"{self.name}: stop requested; it stops after the current step")
        return self

    def __repr__(self) -> str:
        steps = self.steps()
        done = sum(entry["done"] for entry in steps)
        return f"Experience({self.name!r}, {self.status}, {done}/{len(steps)} steps done)"


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


# ── the `with` block ──────────────────────────────────────────────────────────

class Builder:
    def __init__(self, name: str, *, note: str = "", run: bool = True, detach: bool = True,
                 follow: bool = True, gpu=None, method: str | None = None,
                 tags: dict | None = None) -> None:
        self.name = _check_name(name)
        self.note = note
        # The project's method this experience tries, and free labels (`{"variant": "no-aug"}`).
        # Neither changes the fingerprint: they describe, they do not compute.
        self.method = method
        self.tags = dict(tags or {})
        try:
            json.dumps(self.tags)
        except TypeError:
            raise RefError(f"experience {name!r}: tags must be plain values") from None
        self.options = {"run": run, "detach": detach, "follow": follow, "gpu": gpu}
        self._steps: list = []
        self._open = False
        self.experience = Experience(root() / name, name)

    def __enter__(self) -> "Builder":
        self._open = True
        return self

    def __exit__(self, kind, error, traceback_) -> bool:
        self._open = False
        if kind is not None:
            return False                           # an error in the block: nothing launched
        self.build()
        if self.options["run"]:
            self.run(detach=self.options["detach"], follow=self.options["follow"],
                     gpu=self.options["gpu"])
        return False

    def step(self, fn, *args, name: str | None = None, **kwargs) -> StepRef:
        """Record `fn(*args, **kwargs)` as a step and return its output's reference. `fn`
        is importable (defined in a module). If it has a `ctx` parameter, it receives a
        Context (folder, log, progress, checkpoints, should_stop)."""
        if not self._open:
            raise RuntimeError("e.step(…) is only valid inside the `with` block")
        path = importable(fn)
        where = f"{name or path}"
        declared_args = [recipe(value, self, where=f"{where} argument {index}")
                         for index, value in enumerate(args)]
        declared_kwargs = {key: recipe(value, self, where=f"{where} argument {key!r}")
                           for key, value in kwargs.items()}
        ref = StepRef(self, path)
        if name is not None:
            if any(other["ref"].name == name for other in self._steps):
                raise RefError(f"two steps named {name!r}")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise RefError(f"step name {name!r}: letters, digits, _ . - only")
            ref.name = name
        target = getattr(fn, "target", fn)
        self._steps.append({"ref": ref, "function": path, "args": declared_args,
                            "kwargs": declared_kwargs, "code": code_hash(target),
                            "arg_names": _positional_names(target, len(args))})
        return ref

    def build(self) -> Experience:
        """Fix names, fingerprints, and write the declaration."""
        if not self._steps:
            raise RefError(f"experience {self.name!r}: the block declared no step")
        taken = {entry["ref"].name for entry in self._steps if entry["ref"].name}
        counts: dict = {}
        for entry in self._steps:
            if entry["ref"].name is None:
                short = entry["function"].split(":")[1].split(".")[-1]
                while True:
                    counts[short] = counts.get(short, 0) + 1
                    candidate = f"{short}-{counts[short]}"
                    if candidate not in taken:
                        break
                entry["ref"].name = candidate
                taken.add(candidate)
        fingerprints: dict = {}
        steps = []
        for entry in self._steps:
            args, kwargs = freeze(entry["args"]), freeze(entry["kwargs"])
            upstream = steps_read([args, kwargs])
            payload = json.dumps({"function": entry["function"], "args": args,
                                  "kwargs": kwargs}, sort_keys=True, default=str)
            chained = payload + "|" + "|".join(fingerprints[name] for name in upstream)
            fingerprint = hashlib.sha256(chained.encode()).hexdigest()[:12]
            fingerprints[entry["ref"].name] = fingerprint
            steps.append({"name": entry["ref"].name, "function": entry["function"],
                          "args": args, "kwargs": kwargs, "reads": upstream,
                          "fingerprint": fingerprint, "code": entry["code"],
                          "arg_names": entry["arg_names"]})
        spec = {"format": FORMAT, "name": self.name, "note": self.note, "steps": steps,
                "method": self.method, "tags": self.tags,
                "fingerprint": hashlib.sha256("|".join(fingerprints.values()).encode())
                .hexdigest()[:12], "declared": time.time()}
        folder = self.experience.folder
        if (folder / "experience.json").exists() and not _is_ours(folder / "experience.json"):
            raise RefError(f"{folder} holds an experience of another system; useml never "
                           f"overwrites it. Choose another name.")
        previous = self.experience.spec
        if self.experience.status == RUNNING and previous and \
                previous["fingerprint"] != spec["fingerprint"]:
            raise RuntimeError(f"{self.name!r} is running with ANOTHER declaration; "
                               f"exp.open({self.name!r}).stop() it first")
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / ".experience.json.tmp"
        temporary.write_text(json.dumps(spec, indent=1, default=str))
        os.replace(temporary, folder / "experience.json")
        return self.experience

    def get(self, ref):
        return self.experience.get(ref.name if isinstance(ref, StepRef) else ref)

    def __getattr__(self, item):
        # After the block, `e` behaves like its Experience (run, follow, get, results…).
        experience = self.__dict__.get("experience")
        if experience is not None and not item.startswith("_"):
            return getattr(experience, item)
        raise AttributeError(item)


def _positional_names(function, count: int) -> list | None:
    """The parameter names the first `count` positional arguments fill (for reading a
    declaration: `train.lr` rather than `train.arg0`). None when they cannot be known."""
    import inspect

    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return None
    names = []
    for parameter in parameters:
        if len(names) == count:
            break
        if parameter.kind == parameter.VAR_POSITIONAL:
            names += [f"{parameter.name}[{i}]" for i in range(count - len(names))]
            break
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            names.append(parameter.name)
    return names if len(names) == count else None


def experience(name: str, **options) -> Builder:
    """`with exp.experience("folder/name") as e:` — see the module."""
    return Builder(name, **options)


# ── finding experiences again ─────────────────────────────────────────────────

def open_experience(name: str) -> Experience:
    folder = root() / _check_name(name)
    if (folder / "experience.json").exists() and not _is_ours(folder / "experience.json"):
        raise KeyError(f"{name!r} is an experience of another system (no useml format "
                       f"marker): useml does not read it")
    if not (folder / "experience.json").exists():
        known = [entry.name for entry in ls()]
        import difflib

        close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"no experience {name!r} under {root()}.{hint}")
    return Experience(folder, name)


def get(path: str, step: str | None = None):
    """`exp.get("vision/baseline/model")` or `exp.get("vision/baseline", "model")`:
    a step's output, in one line."""
    if step is None:
        path, _, step = path.rpartition("/")
        if not path:
            raise ValueError("expected 'experience/step' (or pass step=)")
    return open_experience(path).get(step)


def ls() -> list[Experience]:
    base = root()
    if not base.is_dir():
        return []
    found = [Experience(spec.parent, spec.parent.relative_to(base).as_posix())
             for spec in base.rglob("experience.json")
             if "steps" not in spec.parent.parts and _is_ours(spec)]
    return sorted(found, key=lambda e: -(e.folder / "experience.json").stat().st_mtime)


def _is_ours(spec_path: Path) -> bool:
    try:
        return json.loads(spec_path.read_text()).get("format") == FORMAT
    except (OSError, ValueError, AttributeError):
        return False


def running() -> list[Experience]:
    return [entry for entry in ls() if entry.status == RUNNING]
