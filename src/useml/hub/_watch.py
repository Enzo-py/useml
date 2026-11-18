"""Layer 3 of the network policy: a SEPARATE process watching a process tree from outside.

    useml hub watch -- python train.py --dataset coco     # run a command and watch it
    useml hub watch --pid 12345                           # watch something running

Layers 0–2 live inside one Python process and are blind to a DataLoader worker or any
subprocess with its own interpreter. This one is not. Every `interval` seconds it counts
the non-local TCP connections of the target and of all its descendants, logs the count,
and, if it stays above `max_connections` for `sustained` checks in a row, kills the whole
tree (SIGTERM, then SIGKILL). A single spike does not kill: DNS and TLS open more than one
socket for an instant.

Exit code: the command's own, or 137 (128 + SIGKILL) if it had to be killed.
"""
from __future__ import annotations

import datetime
import signal
import subprocess
import time
from pathlib import Path


def _psutil():
    try:
        import psutil
    except ImportError as error:
        raise ImportError("useml hub watch needs psutil: pip install 'useml[watch]'") \
            from error
    return psutil


def _is_local(address: str) -> bool:
    return address in ("127.0.0.1", "::1", "") or address.startswith("127.")


def _log(log_path: Path, line: str) -> None:
    stamped = f"{datetime.datetime.now().isoformat()}  {line}"
    print(stamped, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as file:
        file.write(stamped + "\n")


def outbound_connections(root_pid: int) -> list[str]:
    """Non-local TCP connections (established or opening) of `root_pid` and descendants."""
    psutil = _psutil()
    try:
        root = psutil.Process(root_pid)
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    detail = []
    for process in processes:
        try:
            for connection in process.net_connections(kind="tcp"):
                if connection.raddr and connection.status in ("ESTABLISHED", "SYN_SENT") \
                        and not _is_local(connection.raddr.ip):
                    detail.append(f"pid={process.pid} {connection.laddr.ip}:"
                                  f"{connection.laddr.port} -> {connection.raddr.ip}:"
                                  f"{connection.raddr.port} [{connection.status}]")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return detail


def _kill_tree(root_pid: int, log_path: Path, grace: float = 3.0) -> None:
    psutil = _psutil()
    try:
        root = psutil.Process(root_pid)
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    _log(log_path, f"KILL: SIGTERM to {len(processes)} process(es) "
                   f"(pids={[process.pid for process in processes]})")
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace)
    for process in alive:
        _log(log_path, f"KILL: pid={process.pid} still alive after {grace}s, SIGKILL")
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def monitor(root_pid: int, *, max_connections: int = 3, interval: float = 1.0,
            sustained: int = 3, log_path=None, kill: bool = True) -> bool:
    """Poll until the target exits or is killed. True iff it had to kill. The detail of
    the connections is logged on the first tick of a streak and right before a kill only:
    every tick would bury a real alert under duplicates."""
    psutil = _psutil()
    log_path = Path(log_path) if log_path else default_log()
    streak = 0
    while psutil.pid_exists(root_pid):
        try:
            if psutil.Process(root_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        detail = outbound_connections(root_pid)
        if len(detail) > max_connections:
            streak += 1
            about_to_kill = kill and streak >= sustained
            _log(log_path, f"ALERT: {len(detail)} concurrent outbound connections "
                           f"(limit={max_connections}, streak={streak}/{sustained})")
            if streak == 1 or about_to_kill:
                for line in detail:
                    _log(log_path, f"        {line}")
            if streak >= sustained:
                _log(log_path, f"THRESHOLD SUSTAINED for {sustained} checks.")
                if kill:
                    _kill_tree(root_pid, log_path)
                    return True
        else:
            if streak:
                _log(log_path, f"recovered: {len(detail)} connections")
            streak = 0
        time.sleep(interval)
    return False


def default_log() -> Path:
    from .. import paths

    return paths.directory("logs") / "network_monitor.log"


def watch(command=None, *, pid: int | None = None, max_connections: int = 3,
          interval: float = 1.0, sustained: int = 3, kill: bool = True, log_path=None) -> int:
    """Run `command` (a list) under watch, or watch `pid`. Returns an exit code."""
    if (command is None) == (pid is None):
        raise ValueError("pass exactly one of a command or a pid")
    log_path = Path(log_path) if log_path else default_log()
    _log(log_path, f"=== useml hub watch — limit={max_connections} sustained={sustained} "
                   f"interval={interval}s kill={'on' if kill else 'off'} ===")
    process = None
    if command is not None:
        process = subprocess.Popen(list(command))
        pid = process.pid
        _log(log_path, f"watching command (pid={pid}): {' '.join(command)}")
    else:
        _log(log_path, f"watching pid={pid}")
    killed = monitor(pid, max_connections=max_connections, interval=interval,
                     sustained=sustained, log_path=log_path, kill=kill)
    if process is not None:
        if killed:
            _log(log_path, "=== command KILLED for a policy violation ===")
            return 128 + signal.SIGKILL
        code = process.wait()
        _log(log_path, f"=== command exited (code={code}) ===")
        return code
    _log(log_path, f"=== pid={pid} {'killed' if killed else 'exited'} ===")
    return 128 + signal.SIGKILL if killed else 0
