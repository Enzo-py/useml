"""Detached jobs: a processing run in its own process, like an experience.

    job = useml.data.process(useml.data.images("/data/photos"), "mypkg.procs:ClipImage",
                             name="photos-clip", detach=True)
    job.follow()          # a bar fed by the job's state; Ctrl-C stops FOLLOWING, not the job
    feats = job.result()  # the lazy view, once done

    useml data jobs       # in a shell: every job, its state and progress

ON DISK: `<data root>/jobs/<id>/` — `spec.json` (what to run), `state.json` (status, pid,
done / total, written atomically by the job), `run.log` (all its output).

ONE JOB PER TABLE AND SOURCE. Submitting what is already running returns the running job
(re-attach) instead of starting a second one. The table lock would make the second one
wait anyway; this avoids the extra process.

WHAT CAN BE DETACHED: a processor importable as `module:qualname` (defined in a module,
not in a notebook cell) and a source that can be rebuilt: `images`, `texts` or
`source("module:function", ...)`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .. import _progress
from . import _catalog
from ._keys import fingerprint

RUNNING, DONE, FAILED, DIED = "running", "done", "failed", "died"


def jobs_root() -> Path:
    return _catalog.data_root() / "jobs"


def _write_state(folder: Path, **fields) -> None:
    path = folder / "state.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    state.update(fields, updated=time.time())
    temporary = folder / f".state.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(state, indent=1))
    os.replace(temporary, path)


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


class Job:
    def __init__(self, folder) -> None:
        self.folder = Path(folder)
        self.id = self.folder.name

    @property
    def spec(self) -> dict:
        return json.loads((self.folder / "spec.json").read_text())

    @property
    def state(self) -> dict:
        try:
            state = json.loads((self.folder / "state.json").read_text())
        except (OSError, ValueError):
            state = {"status": RUNNING}
        if state.get("status") == RUNNING and not _alive(state.get("pid")):
            state["status"] = DIED                # killed without a chance to say so
        return state

    @property
    def status(self) -> str:
        return self.state.get("status", RUNNING)

    def log(self, lines: int = 40) -> str:
        path = self.folder / "run.log"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) \
            if path.exists() else ""

    def follow(self, *, poll: float = 1.0) -> str:
        """Show the job's progress until it ends. Ctrl-C stops following only."""
        progress = None
        try:
            while True:
                state = self.state
                total, done = state.get("total"), state.get("done", 0)
                if total and progress is None:
                    progress = _progress.bar(total=total, desc=f"{self.id}", unit="item",
                                             delay=0)
                if progress is not None:
                    progress.update(done - progress.n)
                    progress.set_postfix_str(state.get("phase", ""))
                elif state.get("phase"):
                    _progress.note(f"{self.id}: {state['phase']}")
                if state["status"] != RUNNING:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            _progress.note(f"stopped following; {self.id} keeps running "
                           f"(useml data follow {self.id})")
            return RUNNING
        finally:
            if progress is not None:
                progress.close()
        if state["status"] in (FAILED, DIED):
            _progress.note(f"{self.id} {state['status']}:\n{state.get('error') or self.log()}")
        return state["status"]

    def wait(self) -> "Job":
        self.follow()
        return self

    def result(self):
        """The lazy view, once the job is done."""
        status = self.status
        if status != DONE:
            raise RuntimeError(f"job {self.id} is {status}" + (
                f":\n{self.state.get('error') or self.log()}" if status in (FAILED, DIED)
                else "; job.follow() waits for it"))
        name = self.spec.get("name")
        if name:
            return _catalog.load(name)
        from ._processor import resolve
        from ._run import run
        from ._sources import from_spec

        spec = self.spec
        return run(from_spec(spec["source"]), resolve(spec["processor"]), spec["params"])

    def __repr__(self) -> str:
        state = self.state
        progress = f" {state.get('done', 0)}/{state['total']}" if state.get("total") else ""
        return f"Job({self.id!r}, {state['status']}{progress})"


def submit(source, processor, params: dict, *, name=None, batch_size=None) -> Job:
    target = processor.import_path
    if target is None:
        raise ValueError(f"processor {processor.name!r} is defined in a notebook or a "
                         f"script's __main__: a detached job cannot import it. Move it "
                         f"into a module, or run without detach=True.")
    source_spec = source.spec()
    if source_spec is None:
        raise ValueError("this source holds Python objects another process cannot rebuild: "
                         "use useml.data.images(...), texts(...) or source('module:function',"
                         " ...) to detach, or run without detach=True.")
    spec = {"processor": target, "params": params, "source": source_spec, "name": name,
            "batch_size": batch_size, "processor_name": processor.name,
            "table": processor.fingerprint(params)}
    identity = fingerprint({"table": spec["table"], "source": source_spec, "name": name}, 10)
    for job in jobs():
        if job.status == RUNNING and job.spec.get("identity") == identity:
            _progress.note(f"already running: {job.id}; following it")
            return job
    spec["identity"] = identity
    folder = jobs_root() / f"{time.strftime('%Y%m%d-%H%M%S')}-{processor.name}-{identity}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.json").write_text(json.dumps(spec, indent=1, default=str))
    _write_state(folder, status=RUNNING, phase="starting", created=time.time())
    log = open(folder / "run.log", "ab")
    entry = "import sys; from useml.data._jobs import main; sys.exit(main(sys.argv[1]))"
    process = subprocess.Popen([sys.executable, "-c", entry, str(folder)],
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True, cwd=os.getcwd(), env=dict(os.environ))
    log.close()
    _write_state(folder, pid=process.pid)
    _progress.note(f"started {folder.name} (pid {process.pid}); job.follow() shows progress")
    return Job(folder)


def jobs() -> list[Job]:
    root = jobs_root()
    if not root.is_dir():
        return []
    return [Job(path) for path in sorted(root.iterdir(), reverse=True)
            if (path / "spec.json").exists()]


def job(identifier: str) -> Job:
    matches = [candidate for candidate in jobs() if candidate.id.startswith(identifier)
               or identifier in candidate.id]
    if not matches:
        raise KeyError(f"no job matches {identifier!r}")
    return matches[0]


def main(folder: str) -> int:
    """The detached process."""
    folder = Path(folder)
    _write_state(folder, status=RUNNING, pid=os.getpid(), phase="preparing")
    try:
        import useml  # noqa: F401  - paths and network policy, as in the caller
        from ._processor import resolve
        from ._run import run
        from ._sources import from_spec

        spec = json.loads((folder / "spec.json").read_text())
        source = from_spec(spec["source"])
        processor = resolve(spec["processor"])
        _write_state(folder, phase="computing keys")

        def report(done, total):
            _write_state(folder, done=done, total=total, phase="processing")

        run(source, processor, spec["params"], name=spec.get("name"),
            batch_size=spec.get("batch_size"), callback=report)
        _write_state(folder, status=DONE, phase="done", finished=time.time())
        return 0
    except BaseException as error:              # noqa: BLE001 - reported, then re-raised
        _write_state(folder, status=FAILED, phase="failed", finished=time.time(),
                     error="".join(traceback.format_exception(type(error), error,
                                                              error.__traceback__))[-4000:])
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
