"""Processors used by the tests, in a module so that detached jobs can import them."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from useml import data

CALLS = {"init": 0, "items": 0}


@data.processor("length", dtype="float32", batch_size=4)
def length(texts, *, scale=1.0):
    CALLS["items"] += len(texts)
    log = os.environ.get("USEML_TEST_LOG")
    if log:
        with open(log, "a") as file:
            file.write(f"{len(texts)}\n")
    return np.array([[len(text) * scale, len(text.split())] for text in texts])


@data.processor("bytes-embed", batch_size=2, dtype="float16")
class BytesEmbed:
    """A fake image model: the mean of each byte value channel, per file."""

    def __init__(self, width=8, device="cpu"):
        CALLS["init"] += 1
        self.width = width

    def load(self, path):
        return Path(path).read_bytes()

    def __call__(self, blobs):
        CALLS["items"] += len(blobs)
        out = np.zeros((len(blobs), self.width), dtype=np.float32)
        for row, blob in enumerate(blobs):
            values = np.frombuffer(blob, dtype=np.uint8).astype(np.float32)
            out[row] = [values[index::self.width].mean() if len(values[index::self.width])
                        else 0.0 for index in range(self.width)]
        return out


@data.processor("multi", batch_size=3, dtype={"vector": "int8"})
def multi(texts):
    rng = np.random.default_rng(0)
    return {"vector": np.stack([np.full(16, len(text), dtype=np.float32) +
                                rng.normal(size=16).astype(np.float32) for text in texts]),
            "count": np.array([len(text) for text in texts], dtype=np.int32)}


def captions(count=10):
    return [f"caption number {index}" for index in range(count)]
