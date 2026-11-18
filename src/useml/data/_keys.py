"""Content keys and their digests.

A KEY says what an item IS, not where it sits: `file:<hash of the bytes>` for an image,
`text:<hash of the text>` for a caption. Two datasets holding the same image (one built on top of
the other) give the same key, so the row is computed once.

A DIGEST is the 16-byte BLAKE2b of a key: what tables store and index. Lookups go through
the first 8 bytes read as an integer (sorted array + `searchsorted`, 16 bytes of RAM per
row instead of a Python dict) and are confirmed on the full 16 bytes.
"""
from __future__ import annotations

import hashlib

import numpy as np

DIGEST_SIZE = 16


def digest(key: str) -> bytes:
    return hashlib.blake2b(key.encode("utf-8"), digest_size=DIGEST_SIZE).digest()


def digests(keys) -> np.ndarray:
    """(N, 16) uint8 digests of `keys`."""
    keys = list(keys)
    if not keys:
        return np.zeros((0, DIGEST_SIZE), dtype=np.uint8)
    return np.frombuffer(b"".join(digest(key) for key in keys),
                         dtype=np.uint8).reshape(len(keys), DIGEST_SIZE).copy()


def prefixes(rows: np.ndarray) -> np.ndarray:
    """(N,) uint64: the first 8 bytes of each digest, big-endian (sorting them sorts the
    digests by their first 8 bytes)."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    return rows[:, :8].copy().view(">u8").ravel().astype(np.uint64)


def text_key(text: str) -> str:
    return "text:" + hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def canonical_json(value) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def fingerprint(value, length: int = 16) -> str:
    return hashlib.blake2b(canonical_json(value).encode("utf-8"),
                           digest_size=16).hexdigest()[:length]
