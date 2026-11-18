"""Feedback for anything slow: a bar when there is a count, a line when there is not.

THE RULE: nothing that takes more than ~2 seconds runs silently.

    for item in bar(items, desc="hashing images", unit="file"):   # tqdm, shown after 2 s
        ...
    with step("loading CLIP"):                                   # a line after 2 s,
        model = load()                                             # then its duration

Both stay quiet for quick work: a bar appears only once the loop has run for `delay`
seconds, a step prints only if it is still running after `after` seconds. Output goes to
stderr, so it never mixes with what a command prints on stdout.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

DELAY = 2.0


def bar(iterable=None, *, total=None, desc: str | None = None, unit: str = "it",
        delay: float = DELAY, disable: bool = False, **options):
    """A tqdm progress bar shown once the work has lasted `delay` seconds."""
    try:
        from tqdm.auto import tqdm
    except ImportError:                       # pragma: no cover - tqdm is a dependency
        return _PlainBar(iterable, total=total, desc=desc)
    return tqdm(iterable, total=total, desc=desc, unit=unit, delay=delay, file=sys.stderr,
                dynamic_ncols=True, disable=disable, unit_scale=unit == "B", **options)


class _PlainBar:
    def __init__(self, iterable, *, total, desc):
        self.iterable, self.total, self.desc, self.n = iterable, total, desc, 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, count=1):
        self.n += count

    def set_postfix_str(self, *_args, **_kwargs):
        pass

    def write(self, text):
        print(text, file=sys.stderr, flush=True)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextmanager
def step(message: str, *, after: float = DELAY):
    """Say what is happening if it lasts more than `after` seconds, then how long it took."""
    start = time.monotonic()
    announced = threading.Event()

    def announce():
        announced.set()
        print(f"⏳ {message}…", file=sys.stderr, flush=True)

    timer = threading.Timer(after, announce)
    timer.daemon = True
    timer.start()
    try:
        yield
    except BaseException:
        timer.cancel()
        if announced.is_set():
            print(f"✗ {message}: failed after {time.monotonic() - start:.1f}s",
                  file=sys.stderr, flush=True)
        raise
    timer.cancel()
    if announced.is_set():
        print(f"✓ {message} ({time.monotonic() - start:.1f}s)", file=sys.stderr, flush=True)


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
