"""The SUPERVISOR: the parent of every detached run, so that no run dies without a word.

    tmux session "useml-<experience>"            (if tmux exists: `useml exp attach`)
      └── supervisor   python -m useml.exp._supervise <folder>
            └── run    python -m useml.exp._runtime <folder>

WHY A PARENT. A process killed by the kernel (out of memory: SIGKILL) or by a native crash
(SIGSEGV in CUDA or a C extension) cannot write anything about its own death. Its parent
can: `wait()` returns the exit code or the signal. The supervisor does nothing else — it
holds no model, no data, a few MB of RAM — so it is not what the OOM killer picks.

WHAT IT RECORDS when the run ends (`state.json` → "exit", and `crash.txt` if it failed):
exit code or signal, what that signal usually means, the peak RAM of the run, the machine's
RAM, the kernel's OOM message when it can read it, how long it ran, and the last lines
of the log. `useml exp why <name>` prints it.

SURVIVING A CLOSED LAPTOP. The supervisor starts its own session (setsid) and ignores
SIGHUP: closing the SSH connection or the Jupyter tab sends a hang-up to the terminal's
session, not to this one. tmux adds a terminal one can reattach to. What neither survives:
a reboot, or a machine configured to kill a user's processes at logout
(systemd `KillUserProcesses=yes` without lingering) — checked at launch, said loudly.

THE LOG is written by the supervisor (it reads the run's output and writes each line to
`run.log` AND to its own output, i.e. the tmux pane): nothing depends on a `| tee`.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ._runtime import read_state, write_state

HEARTBEAT_SECONDS = 15

# `-c`, not `-m useml.exp._runtime`: the package `useml.exp` imports `_runtime` itself, and
# runpy then warns that the module was imported twice.
RUNTIME_ENTRY = "import sys; from useml.exp._runtime import main; sys.exit(main(sys.argv[1]))"
SUPERVISOR_ENTRY = ("import sys; from useml.exp._supervise import supervise; "
                    "sys.exit(supervise(sys.argv[1]))")

SIGNAL_MEANING = {
    "SIGKILL": "killed by SIGKILL, which cannot be caught. Most often the kernel's OOM "
               "killer (the machine ran out of RAM); otherwise an admin, a job scheduler "
               "or `useml exp stop --now`.",
    "SIGSEGV": "segmentation fault in native code (CUDA, a C/C++ extension). The Python "
               "stack at the crash is in the log just above (faulthandler).",
    "SIGABRT": "aborted by native code (a failed C/CUDA assertion). The Python stack is in "
               "the log just above (faulthandler).",
    "SIGBUS": "bus error: often a memory-mapped file that shrank or a full /dev/shm.",
    "SIGTERM": "terminated by SIGTERM: `useml exp stop --now`, a job scheduler, or a "
               "machine shutting down.",
    "SIGINT": "interrupted (Ctrl-C in the tmux pane).",
    "SIGHUP": "hang-up: the terminal or session it was attached to closed.",
}


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def machine_memory() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _peak_children_rss() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024   # Linux: KB


def _kernel_oom_lines(pid: int) -> list[str]:
    """The kernel's OOM-killer lines about `pid`, if the kernel log is readable."""
    for command in (["journalctl", "-k", "--since", "-2h", "--no-pager", "-q"],
                    ["dmesg", "-T"], ["dmesg"]):
        try:
            text = subprocess.run(command, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        lines = [line for line in text.splitlines()
                 if ("Out of memory" in line or "oom-kill" in line or "Killed process" in line)
                 and str(pid) in line]
        if lines:
            return lines[-5:]
    return []


def _human(size) -> str:
    if not size:
        return "?"
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def explain(returncode: int, *, pid: int, peak: int, seconds: float, log_tail: str,
            state: dict) -> dict:
    """What the exit of the run means, as data and as text."""
    info = {"returncode": returncode, "pid": pid, "seconds": seconds, "peak_rss": peak,
            "machine_ram": machine_memory(), "host": platform.node()}
    if returncode < 0:
        name = _signal_name(-returncode)
        info["signal"] = name
        meaning = f"{name}: " + SIGNAL_MEANING.get(name, f"killed by {name}.")
        ram = info["machine_ram"]
        if name == "SIGKILL" and ram and peak:
            meaning += f" Peak RAM of the run: {_human(peak)} of {_human(ram)}."
        oom = _kernel_oom_lines(pid) if name == "SIGKILL" else []
        if oom:
            info["kernel"] = oom
            meaning = "killed by the kernel's OOM killer (out of RAM). " + meaning
    elif returncode == 0:
        meaning = "exited normally"
    else:
        meaning = f"exited with code {returncode}"
        error = state.get("error") or ""
        if "CUDA out of memory" in error or "OutOfMemoryError" in error:
            meaning += ": the GPU ran out of memory (lower the batch size, or free the card)"
        elif error:
            meaning += ": a Python exception (traceback below)"
    info["meaning"] = meaning
    info["log_tail"] = log_tail
    return info


def crash_report(info: dict, state: dict, name: str) -> str:
    lines = [f"experience {name} — {info['meaning']}",
             f"host {info['host']}, pid {info['pid']}, ran {info['seconds']:.0f}s, "
             f"step {state.get('step') or '?'}",
             f"peak RAM {_human(info['peak_rss'])} of {_human(info['machine_ram'])}"]
    if info.get("kernel"):
        lines += ["", "kernel:"] + [f"  {line}" for line in info["kernel"]]
    if state.get("error"):
        lines += ["", "Python error:", state["error"].rstrip()]
    lines += ["", "last lines of run.log:", info["log_tail"]]
    return "\n".join(lines)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 64 * 1024))
            text = file.read().decode(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _notify(folder: Path, status: str, message: str) -> None:
    """Run the machine's `[exp] notify` command, if any (a phone push, a mail...), with
    USEML_EXP_NAME / _STATUS / _MESSAGE in its environment. Never fails the run."""
    try:
        from .. import paths

        command = paths.config().get("exp", {}).get("notify")
        if not command:
            return
        env = dict(os.environ, USEML_EXP_NAME=read_state(folder).get("name", folder.name),
                   USEML_EXP_STATUS=status, USEML_EXP_MESSAGE=message[:1000])
        subprocess.run(command, shell=True, env=env, timeout=60, capture_output=True)
    except Exception:                                         # noqa: BLE001
        pass


def supervise(folder) -> int:
    folder = Path(folder)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    env = dict(os.environ)
    env_file = folder / ".launch-env.json"
    if env_file.exists():                      # the launcher's environment, then removed
        try:
            env.update(json.loads(env_file.read_text()))
        finally:
            env_file.unlink(missing_ok=True)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = folder / "run.log"
    started = time.time()
    write_state(folder, supervisor_pid=os.getpid(), host=platform.node(),
                heartbeat=started, exit=None)
    child = subprocess.Popen([sys.executable, "-u", "-c", RUNTIME_ENTRY, str(folder)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                             cwd=env.get("USEML_LAUNCH_CWD") or os.getcwd())
    write_state(folder, runtime_pid=child.pid)
    # The supervisor's own SIGTERM (`stop --now`, a shutdown) is passed to the run, then
    # the supervisor waits to record how it ended.
    signal.signal(signal.SIGTERM, lambda *_: child.send_signal(signal.SIGTERM))
    stop_beating = threading.Event()

    def beat():
        while not stop_beating.wait(HEARTBEAT_SECONDS):
            try:
                write_state(folder, supervisor_heartbeat=time.time())
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()
    with open(log_path, "ab", buffering=0) as log:
        for line in iter(child.stdout.readline, b""):
            log.write(line)
            try:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
            except (OSError, ValueError):
                pass                           # no terminal (detached without tmux)
    returncode = child.wait()
    stop_beating.set()
    state = read_state(folder)
    info = explain(returncode, pid=child.pid, peak=_peak_children_rss(),
                   seconds=time.time() - started, log_tail=_tail(log_path), state=state)
    status = state.get("status")
    if state.get("stop_requested") and returncode != 0:
        write_state(folder, status="stopped", exit=info, finished=time.time(),
                    error=f"stopped on request ({info['meaning']})")
        _notify(folder, "stopped", f"{state.get('name', folder.name)}: stopped on request")
        return 0
    if returncode != 0 or status == "running":
        # The run could not say it failed (killed, crashed natively): say it for it.
        # Every step done, then a crash at exit (a native destructor): the results stand.
        status = "failed" if status in (None, "running") else status
        report = crash_report(info, state, state.get("name", folder.name))
        (folder / "crash.txt").write_text(report + "\n")
        write_state(folder, status=status, exit=info, finished=time.time(),
                    error=state.get("error") or info["meaning"])
        with open(log_path, "a") as log:
            log.write(f"\n=== useml supervisor: the run {info['meaning']} — see crash.txt "
                      f"or `useml exp why` ===\n")
        _notify(folder, status, report)
    else:
        write_state(folder, exit=info)
        _notify(folder, status or "done", f"{state.get('name', folder.name)}: {status}")
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1]))
