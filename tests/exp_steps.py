"""Step functions and config types used by the experience tests (importable by the
detached process)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

__all__ = ["Linear"]

try:
    import torch

    class Linear(torch.nn.Linear):
        def __init__(self, in_features: int = 4, out_features: int = 2, seed: int = 0):
            torch.manual_seed(seed)
            super().__init__(in_features, out_features)
            self.seed = seed
except ImportError:                                       # pragma: no cover
    Linear = None


def _count(name: str) -> None:
    log = os.environ.get("USEML_TEST_CALLS")
    if log:
        with open(log, "a") as file:
            file.write(name + "\n")


def numbers(n: int):
    _count("numbers")
    return np.arange(n, dtype=np.float32)


def square(values, *, ctx=None):
    _count("square")
    for index in range(3):
        if ctx is not None:
            ctx.progress(index + 1, 3, "squaring")
    return values ** 2


def summary(values, label: str = "x"):
    _count("summary")
    return {"label": label, "sum": float(np.sum(values)), "count": int(len(values))}


def fail_unless(marker: str):
    _count("fail_unless")
    if not Path(marker).exists():
        raise RuntimeError("marker missing")
    return {"ok": True}


def train(model, steps: int, *, ctx):
    _count("train")
    with torch.no_grad():
        for step in range(steps):
            model.weight.add_(1.0)
            ctx.progress(step + 1, steps, "training")
    ctx.checkpoints.save({"weight": model.weight.clone()}, step=steps, metrics={"loss": 0.1})
    return model


def rows(dataset):
    _count("rows")
    return {"rows": len(dataset), "first": dataset[0].tolist()}


def slow(seconds: float, *, ctx):
    _count("slow")
    deadline = time.time() + seconds
    while time.time() < deadline:
        if ctx.should_stop():
            return {"stopped": True}
        time.sleep(0.05)
    return {"stopped": False}


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    axis.plot([1, 2, 3])
    return {"plot": fig, "value": 3}


def identity(value):
    return value


def sigkill():
    import signal

    print("about to be killed", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


def segfault():
    import ctypes

    print("about to crash", flush=True)
    ctypes.string_at(0)


def boom():
    raise ValueError("a clear Python error")


def sleepy(seconds: float):
    time.sleep(seconds)
    return {"slept": seconds}


def rows_of(dataset):
    return len(list(dataset))


def train_with_metrics(steps: int, *, ctx):
    for step in range(1, steps + 1):
        ctx.metric(step, loss=1.0 / step, accuracy=step / steps)
        ctx.progress(step, steps, "training")
    print(f"trained in {ctx.dir}")                       # a path, to check the masking
    return {"final_loss": 1.0 / steps}


def tqdm_train(epochs: int):
    """Training code that never calls ctx: a tqdm bar with a postfix, and printed lines."""
    from tqdm import tqdm

    bar = tqdm(range(epochs), desc="train")
    for epoch in bar:
        bar.set_postfix(loss=1.0 / (epoch + 1), acc=f"{epoch / epochs:.2f}")
        print(f"epoch={epoch} val_loss={2.0 / (epoch + 1):.3f} lr: 0.001")
    print("seed: 3")                                      # one `name: number` alone: no metric
    return {"done": True}


def explicit_metrics(n: int, *, ctx):
    from tqdm import tqdm

    for i in tqdm(range(n)):
        ctx.metric(i, mine=float(i))
        print(f"other={i}")
    return n


def big_report(n: int):
    """A report too big to send whole (> 64 KB): its numbers still count."""
    return {"metrics": {"edge_f1": 0.42, "n_test": n}, "per_sample": [{"i": i, "score": i / n} for i in range(n)]}


def assign_curve(steps: int, *, ctx):
    for step in range(steps):
        ctx.metric(step, assign=2.0 - step / steps, val_acc=step / steps)
    return {}


try:                                                     # a small PyTorch model
    from torch import nn as _nn

    class TinyBlock(_nn.Module):
        def __init__(self, norm: bool):
            super().__init__()
            self.norm = _nn.LayerNorm(8) if norm else _nn.Identity()
            self.mlp = _nn.Sequential(_nn.Linear(8, 16), _nn.GELU(), _nn.Linear(16, 8))

    class TinyNet(_nn.Module):
        def __init__(self, depth: int, norm: bool):
            super().__init__()
            self.layers = _nn.ModuleList([TinyBlock(norm) for _ in range(depth)])
            self.head = _nn.Linear(8, 2)
except ImportError:                                      # pragma: no cover
    pass


def tiny_net(depth: int, norm: bool):
    """A step returning a model: its blocks are recorded."""
    return TinyNet(depth, norm)
