"""Spaces, costs and sizes.

A SPACE groups the entries of one kind: `results/scores`, `processing/masks`… Its name is
`<family>/<name>`. A project declares its spaces once (`store.declare(...)`); the
declaration is written IN THE INDEX, so `useml store gc` knows each space's cost without
importing the project's code.

COST says what an entry costs to recompute, hence the order in which `gc` evicts:
`cheap` first, `expensive` next, `precious` never without `include_precious=True` (a score
that took hours of GPU is not thrown away to make room).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

COSTS = ("cheap", "expensive", "precious")
_NAME = re.compile(r"^[a-z0-9_][a-z0-9_.-]*(/[a-z0-9_][a-z0-9_.-]*)+$")


@dataclass(frozen=True)
class Space:
    name: str
    cost: str = "expensive"
    description: str = ""
    producer: str = ""

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"space name {self.name!r}: expected '<family>/<name>' in lower "
                             f"case, e.g. 'results/scores'")
        if self.cost not in COSTS:
            raise ValueError(f"space {self.name!r}: cost must be one of {COSTS}")

    @property
    def family(self) -> str:
        return self.name.split("/", 1)[0]


def cost_rank(cost: str | None) -> int:
    """0 = evicted first. An undeclared space (an old index) counts as expensive."""
    return COSTS.index(cost) if cost in COSTS else 1


_UNITS = {"": 1, "B": 1, "K": 1 << 10, "KB": 1 << 10, "M": 1 << 20, "MB": 1 << 20,
          "G": 1 << 30, "GB": 1 << 30, "T": 1 << 40, "TB": 1 << 40}


def parse_size(value) -> int:
    """`"450GB"`, `"1.5 TB"`, `"200g"` or a number of bytes → bytes (binary units)."""
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(value))
    if not match or match.group(2).upper() not in _UNITS:
        raise ValueError(f"cannot read the size {value!r}; write e.g. '450GB'")
    return int(float(match.group(1)) * _UNITS[match.group(2).upper()])


def human(size) -> str:
    if size is None:
        return "?"
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
