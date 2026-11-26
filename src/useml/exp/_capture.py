"""Live curves, progress and log for steps that never call `ctx` — captured while they run.

Most training code already says how it goes, just not to useml: a tqdm bar
(`bar.set_postfix(loss=0.23)`) or a printed line (`epoch 3: loss=0.23 acc=0.81`). While a
step runs, useml listens to both:

    tqdm postfix      every number given to set_postfix / set_postfix_str → a point
    printed lines     `key=number` pairs (or two or more `key: number` pairs) → a point;
                      `epoch`, `step`, `iter`, `iteration` or `global_step` on the line is
                      the point's x
    progress          the outermost tqdm bar (n / total, its description), when the step
                      never calls ctx.progress

Points go to the step's `metrics.jsonl`, like `ctx.metric` (a bar: at most 5 points per second,
its last value always kept; printed lines: at most 50 per second). A step that calls `ctx.metric` itself keeps only its own
points. `USEML_NO_AUTO_METRICS=1` turns the capture off.

LOG: a run in this process (`detach=False`, a notebook) also writes what it prints to
`run.log` — the dashboard and `useml exp log` read it — while printing as before. Progress
bars redrawn with `\\r` are written once, as their final line.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

PAIR = re.compile(r"(?<![\w.])([A-Za-z][\w./-]{0,40}?)\s*([=:])\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?![\w.])")
X_KEYS = ("global_step", "step", "iteration", "iter", "epoch")
X_WORD = re.compile(r"\b(global_step|step|iteration|iter|epoch)\s+(\d+)(?:\s*/\s*\d+)?\b", re.I)
TQDM_LINE = re.compile(r"\d+/\d+\s*\[\d")       # a progress bar's line: its postfix is read apart
NOT_METRICS = {"pid", "port", "seed", "gpu", "rank", "worker", "workers", "batch_size", "bs",
               "line", "id", "n", "total", "it", "s"}
MAX_POINTS_PER_SECOND = {"bar": 5, "line": 50}


def enabled() -> bool:
    return os.environ.get("USEML_NO_AUTO_METRICS", "").strip() not in ("1", "true", "yes")


def parse_line(line: str) -> tuple[float | None, dict]:
    """(x or None, {name: number}) from a printed line; ({}, when it holds no metric)."""
    if TQDM_LINE.search(line):
        return None, {}
    pairs = PAIR.findall(line)
    if not pairs:
        return None, {}
    if not any(sign == "=" for _, sign, _ in pairs) and len(pairs) < 2:
        return None, {}                    # one `name: 3` alone is too often not a metric
    word = X_WORD.search(line)
    x, values = (float(word.group(2)) if word else None), {}
    for key, _, number in pairs:
        name = key.strip("./-")
        lowered = name.lower()
        try:
            value = float(number)
        except ValueError:
            continue
        if lowered in X_KEYS:
            x = value if x is None or X_KEYS.index(lowered) < 3 else x
            continue
        if lowered in NOT_METRICS:
            continue
        values[name] = value
    return x, values


class Capture:
    """Installed around one step: writes points and progress, tees the log if asked."""

    def __init__(self, folder: Path, step: str, *, context=None) -> None:
        self.folder = Path(folder)
        self.step = step
        self.dir = self.folder / "steps" / step
        self.context = context
        self.lock = threading.Lock()
        self.count = {"bar": 0, "line": 0}     # each source's x when none is given
        self.window = {"bar": (0, 0), "line": (0, 0)}   # (second, points written in it)
        self.pending: dict = {}                # bar id -> its last unwritten values
        self.last_progress = 0.0
        self.bars: list = []
        self.patched: list = []
        self.streams: tuple = ()

    # ── writing ──────────────────────────────────────────────────────────────

    def _explicit(self, kind: str) -> bool:
        return bool(self.context is not None and getattr(self.context, f"_used_{kind}", False))

    def point(self, values: dict, x: float | None = None, *, source: str = "line",
              force: bool = False) -> bool:
        if not values or self._explicit("metric"):
            return False
        with self.lock:
            now = time.time()
            second, written = self.window[source]
            if int(now) != second:
                second, written = int(now), 0
            if written >= MAX_POINTS_PER_SECOND[source] and not force:
                return False
            self.window[source] = (second, written + 1)
            self.count[source] += 1
            entry = {"step": x if x is not None else self.count[source], "time": now, **values}
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.dir / "metrics.jsonl", "a") as file:
                    file.write(json.dumps(entry) + "\n")
            except OSError:
                return False
            return True

    def progress(self, bar) -> None:
        if self._explicit("progress") or not self.bars or bar is not self.bars[0]:
            return
        total = getattr(bar, "total", None)
        if not total:
            return
        now = time.monotonic()
        if now - self.last_progress < 0.5 and bar.n < total:
            return
        self.last_progress = now
        from ._runtime import write_state

        try:
            write_state(self.folder, step_progress=[int(bar.n), int(total),
                                                    str(getattr(bar, "desc", "") or "")[:60]])
        except Exception:                                    # noqa: BLE001 - never break a run
            pass

    # ── tqdm ─────────────────────────────────────────────────────────────────

    def _patch_tqdm(self) -> None:
        try:
            from tqdm import std
        except ImportError:
            return
        cls = std.tqdm
        capture = self
        originals = {name: getattr(cls, name) for name in
                     ("__init__", "update", "close", "set_postfix", "set_postfix_str")}

        def __init__(bar, *args, **kwargs):
            originals["__init__"](bar, *args, **kwargs)
            if not getattr(bar, "disable", False):
                capture.bars.append(bar)

        def update(bar, n=1):
            result = originals["update"](bar, n)
            capture.progress(bar)
            return result

        def close(bar):
            values = capture.pending.pop(id(bar), None)
            if values:
                capture.point(values, source="bar", force=True)   # the last value counts
            if bar in capture.bars:
                capture.progress(bar)
                capture.bars.remove(bar)
            return originals["close"](bar)

        def numbers(items) -> dict:
            out = {}
            for key, value in items:
                if str(key).lower() in NOT_METRICS:
                    continue
                try:
                    out[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            return out

        def remember(bar, values):
            if values and not capture.point(values, source="bar"):
                capture.pending[id(bar)] = values
            elif values:
                capture.pending.pop(id(bar), None)

        def set_postfix(bar, ordered_dict=None, refresh=True, **kwargs):
            result = originals["set_postfix"](bar, ordered_dict, refresh, **kwargs)
            remember(bar, numbers([*(ordered_dict or {}).items(), *kwargs.items()]))
            return result

        def set_postfix_str(bar, s="", refresh=True):
            result = originals["set_postfix_str"](bar, s, refresh)
            remember(bar, parse_line(str(s))[1])
            return result

        for name, function in (("__init__", __init__), ("update", update), ("close", close),
                               ("set_postfix", set_postfix),
                               ("set_postfix_str", set_postfix_str)):
            setattr(cls, name, function)
        self.patched = [(cls, originals)]

    # ── printed lines ────────────────────────────────────────────────────────

    def _wrap_streams(self) -> None:
        self.streams = (sys.stdout, sys.stderr)
        sys.stdout = _Watcher(sys.stdout, self.line)
        sys.stderr = _Watcher(sys.stderr, self.line)

    def line(self, text: str) -> None:
        x, values = parse_line(text)
        if values:
            self.point(values, x)

    # ── installing ───────────────────────────────────────────────────────────

    def __enter__(self) -> "Capture":
        if enabled():
            self._patch_tqdm()
            self._wrap_streams()
        return self

    def __exit__(self, *exc) -> bool:
        for values in list(self.pending.values()):
            self.point(values, source="bar", force=True)
        self.pending.clear()
        for cls, originals in self.patched:
            for name, function in originals.items():
                setattr(cls, name, function)
        if self.streams:
            for watcher in (sys.stdout, sys.stderr):
                if isinstance(watcher, _Watcher):
                    watcher.flush_line()
            sys.stdout, sys.stderr = self.streams
        return False


class tee_log:
    """While open, what this process prints also goes to `path` (whole lines)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "tee_log":
        self.file = open(self.path, "a", encoding="utf-8", errors="replace")
        self.streams = (sys.stdout, sys.stderr)

        def write(line: str) -> None:
            self.file.write(line + "\n")
            self.file.flush()
        sys.stdout = _Watcher(sys.stdout, write)
        sys.stderr = _Watcher(sys.stderr, write)
        return self

    def __exit__(self, *exc) -> bool:
        for watcher in (sys.stdout, sys.stderr):
            if isinstance(watcher, _Watcher):
                watcher.flush_line()
        sys.stdout, sys.stderr = self.streams
        self.file.close()
        return False


class _Watcher:
    """A stream that passes everything through, keeps whole lines for the log and reads
    metrics from them. `\\r` redraws (progress bars) keep only their last state."""

    def __init__(self, stream, on_line) -> None:
        self._stream = stream
        self._on_line = on_line
        self._buffer = ""

    def write(self, text):
        written = self._stream.write(text)
        try:
            self._buffer += str(text)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line.rsplit("\r", 1)[-1])
            if len(self._buffer) > 65536:
                self._buffer = self._buffer.rsplit("\r", 1)[-1][-4096:]
        except Exception:                                    # noqa: BLE001 - never break output
            pass
        return written

    def _emit(self, line: str) -> None:
        self._on_line(line)

    def flush_line(self) -> None:
        tail = self._buffer.rsplit("\r", 1)[-1]
        self._buffer = ""
        if tail.strip():
            self._emit(tail)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)
