"""Running an experience: in its own process (detached) or in this one.

    <experience>/
        experience.json        the declaration: steps, recipes, fingerprints
        state.json             status, pid, current step and its progress (atomic writes)
        run.log                everything the run prints
        control                write `stop` here: the run stops cleanly between steps
                               (and inside a step that checks `ctx.should_stop()`)
        steps/<name>/          what the step writes (`ctx.dir`) — survives an interruption
        steps/<name>/done.json written LAST: {fingerprint, seconds, output, code}

RESUMING. A step whose `done.json` carries the RIGHT fingerprint is skipped. The
fingerprint CHAINS those of the steps it reads: changing a step redoes the ones that read
it, not the others. A step started under ANOTHER fingerprint does not resume its old
folder (old checkpoints would be read under the new declaration): the folder is set aside
as `steps/<name>.stale-<date>`, never deleted.

THE CODE of a step's function is hashed too, but only to WARN when it changed: rerunning
hours of training for a renamed variable would be worse than a warning.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import _outputs
from ._refs import code_hash, rebuild, resolve_function

RUNNING, DONE, FAILED, DIED, STOPPED, NEW = "running", "done", "failed", "died", "stopped", "new"


class StopRequested(Exception):
    """A clean stop asked through the `control` file."""


def write_state(folder: Path, **fields) -> None:
    """Merge `fields` into state.json. The supervisor and the run both write it: a file
    lock makes each read-modify-write whole, so neither erases the other's fields."""
    import fcntl
    import threading

    path = folder / "state.json"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".state.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                state = {}
            state.update(fields, updated=time.time())
            temporary = folder / f".state.{os.getpid()}.{threading.get_ident()}.tmp"
            temporary.write_text(json.dumps(state, indent=1, default=str))
            os.replace(temporary, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_state(folder: Path) -> dict:
    try:
        return json.loads((folder / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def is_done(folder: Path, step: str, fingerprint: str) -> bool:
    try:
        return json.loads((folder / "steps" / step / "done.json").read_text()) \
            .get("fingerprint") == fingerprint
    except (OSError, ValueError):
        return False


class Checkpointer:
    """A step's checkpoints: write atomically, keep the last N (and the best of each
    metric), resume. `index.json` lists them with their metrics: picking the best does
    not load weights."""

    def __init__(self, folder, *, keep_last: int = 3) -> None:
        self.folder = Path(folder)
        self.keep_last = int(keep_last)

    def index(self) -> list:
        try:
            return json.loads((self.folder / "index.json").read_text())
        except (OSError, ValueError):
            return []

    def save(self, state: dict, *, step: int, metrics: dict | None = None) -> Path:
        import torch

        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"step-{int(step):08d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        index = [entry for entry in self.index() if entry["step"] != int(step)]
        index.append({"step": int(step), "file": path.name, "metrics": dict(metrics or {})})
        index.sort(key=lambda entry: entry["step"])
        keep = {entry["file"] for entry in index[-self.keep_last:]} | self._bests(index)
        for entry in index:
            if entry["file"] not in keep:
                (self.folder / entry["file"]).unlink(missing_ok=True)
        temporary = self.folder / "index.tmp"
        temporary.write_text(json.dumps([e for e in index if e["file"] in keep], indent=1))
        os.replace(temporary, self.folder / "index.json")
        return path

    @staticmethod
    def _bests(index: list) -> set:
        out = set()
        for metric in {key for entry in index for key in entry["metrics"]}:
            scored = [e for e in index if isinstance(e["metrics"].get(metric), (int, float))]
            if scored:
                out.add(min(scored, key=lambda e: e["metrics"][metric])["file"])
                out.add(max(scored, key=lambda e: e["metrics"][metric])["file"])
        return out

    def _load(self, entry):
        if entry is None:
            return None
        import torch

        return torch.load(self.folder / entry["file"], map_location="cpu", weights_only=False)

    def latest(self):
        index = self.index()
        return self._load(index[-1] if index else None)

    def latest_step(self) -> int | None:
        index = self.index()
        return index[-1]["step"] if index else None

    def best(self, metric: str, *, mode: str = "min"):
        scored = [e for e in self.index() if isinstance(e["metrics"].get(metric), (int, float))]
        if not scored:
            return None
        return self._load((min if mode == "min" else max)(scored, key=lambda e: e["metrics"][metric]))


class Context:
    """What a step function receives if it has a parameter named `ctx`."""

    def __init__(self, folder: Path, name: str) -> None:
        self.experience_dir = folder
        self.name = name
        self.dir = folder / "steps" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last = 0.0

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}", flush=True)

    def progress(self, done: int, total: int, label: str = "") -> None:
        """Where the step is (epoch i/N, batch i/N): shown by `exp.follow()`."""
        if time.monotonic() - self._last > 0.5 or done >= total:
            write_state(self.experience_dir, step_progress=[int(done), int(total), label])
            self._last = time.monotonic()

    def metric(self, step: int, **values) -> None:
        """Record metric values at `step` (`ctx.metric(120, loss=0.31, acc=0.8)`): kept in
        `metrics.jsonl` of the step, drawn as curves by the monitor."""
        numbers = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
        if not numbers:
            return
        with open(self.dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps({"step": int(step), "time": time.time(), **numbers}) + "\n")

    @property
    def checkpoints(self) -> Checkpointer:
        return Checkpointer(self.dir / "checkpoints")

    def should_stop(self) -> bool:
        """True when `stop` was written to the control file: save, then return early."""
        try:
            return (self.experience_dir / "control").read_text().strip() == "stop"
        except OSError:
            return False


def _set_aside_if_stale(step_dir: Path, fingerprint: str) -> Path | None:
    if not step_dir.is_dir():
        return None
    previous = None
    for name in ("started.json", "done.json"):
        try:
            previous = json.loads((step_dir / name).read_text()).get("fingerprint")
            break
        except (OSError, ValueError):
            continue
    if previous is None or previous == fingerprint:
        return None
    stamp = time.strftime('%Y%m%d-%H%M%S')
    stale = step_dir.with_name(f"{step_dir.name}.stale-{stamp}")
    suffix = 1
    while stale.exists():                       # set aside twice in the same second
        suffix += 1
        stale = step_dir.with_name(f"{step_dir.name}.stale-{stamp}-{suffix}")
    step_dir.rename(stale)
    return stale


def execute(folder) -> None:
    """Every step, in order, skipping those already done."""
    folder = Path(folder)
    spec = json.loads((folder / "experience.json").read_text())
    control = folder / "control"
    control.unlink(missing_ok=True)                       # a forgotten `stop` stops nothing
    import platform

    write_state(folder, status=RUNNING, pid=os.getpid(), runtime_pid=os.getpid(),
                host=platform.node(), started=time.time(), error=None, step=None,
                step_progress=None, name=spec.get("name", folder.name), crash=None)
    (folder / "crash.txt").unlink(missing_ok=True)
    cache: dict = {}

    def output(name: str):
        if name not in cache:
            meta = json.loads((folder / "steps" / name / "done.json").read_text())
            cache[name] = _outputs.load(folder / "steps" / name, meta["output"])
        return cache[name]

    done_steps = []
    try:
        for index, step in enumerate(spec["steps"]):
            name, fingerprint = step["name"], step["fingerprint"]
            if control.exists() and control.read_text().strip() == "stop":
                control.unlink(missing_ok=True)
                raise StopRequested(f"stopped on request before step {name!r}")
            step_dir = folder / "steps" / name
            if is_done(folder, name, fingerprint):
                print(f"[experience] {name}: already done, skipped", flush=True)
                done_steps.append(name)
                write_state(folder, steps_done=done_steps)
                continue
            stale = _set_aside_if_stale(step_dir, fingerprint)
            if stale is not None:
                print(f"[experience] {name}: declaration changed; the former folder is set "
                      f"aside as {stale.name} and the step starts from scratch", flush=True)
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "started.json").write_text(json.dumps({"fingerprint": fingerprint}))
            write_state(folder, step=name, step_index=index, step_progress=None)
            print(f"[experience] {name}: start ({step['function']})", flush=True)
            started = time.time()
            function = resolve_function(step["function"])
            target = getattr(function, "target", function)       # a decorated processor
            current = code_hash(target)
            if step.get("code") and current and current != step["code"]:
                print(f"[experience] {name}: the code of {step['function']} changed since "
                      f"the declaration (the step runs with the current code)", flush=True)
            args = rebuild(step["args"], output)
            kwargs = rebuild(step["kwargs"], output)
            try:
                accepts_ctx = "ctx" in inspect.signature(function).parameters
            except (TypeError, ValueError):
                accepts_ctx = False
            if accepts_ctx:
                kwargs["ctx"] = Context(folder, name)
            value = function(*args, **kwargs)
            meta = _outputs.save(value, step_dir)
            cache[name] = value
            seconds = time.time() - started
            (step_dir / "done.json").write_text(json.dumps(
                {"fingerprint": fingerprint, "seconds": seconds, "finished": time.time(),
                 "output": meta, "code": current,
                 "snapshot": read_state(folder).get("code")}, indent=1))
            print(f"[experience] {name}: done in {seconds:.0f}s ({meta['kind']})", flush=True)
            done_steps.append(name)
            write_state(folder, steps_done=done_steps, step=None, step_progress=None)
    except StopRequested as stop:
        write_state(folder, status=STOPPED, finished=time.time(), error=str(stop))
        print(f"[experience] {stop}; run it again to resume", flush=True)
        return
    except BaseException as error:
        write_state(folder, status=FAILED, finished=time.time(),
                    error="".join(traceback.format_exception(type(error), error,
                                                             error.__traceback__))[-6000:])
        raise
    write_state(folder, status=DONE, finished=time.time(), step=None, steps_done=done_steps)
    print(f"[experience] finished: {len(done_steps)} step(s)", flush=True)


def main(folder: str) -> int:
    """The run process (child of the supervisor).

    FAULTHANDLER: a segfault or an abort in native code prints the Python stack of every
    thread to the log before dying — the one trace such a crash can leave. SIGUSR1 prints
    the same stacks WITHOUT dying: `useml exp stack <name>` shows where a run that seems
    stuck is."""
    import faulthandler
    import signal

    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    import useml  # noqa: F401  - cache paths and the download policy, as in the notebook

    try:
        execute(folder)
    except BaseException:                                  # noqa: BLE001 - state written
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
