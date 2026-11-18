"""useml.datasets — many datasets, each declared once by a driver, loaded by name.

    from useml import datasets as ds

    cifar = ds.load("cifar10")                    # prepares on first use: download, index, check
    train = ds.load("cifar10", split="train")
    train[0]                                      # a record: {"id": ..., "image": Path, "label": 3}
    print(cifar.summary())

    data.process(train.field("image"), ClipImage, name="cifar10-train-clip")

Write a driver: see `_driver` (declaration, preparation) and the guide `useml docs datasets`.
Where datasets live: `$USEML_DATASETS_ROOT`, `[tool.useml.datasets] root` of the project,
`[datasets] root` of the machine config, else `<home>/datasets`.
"""
from __future__ import annotations

from ._dataset import Dataset
from ._driver import (Card, DatasetError, Driver, Manual, ManualDownloadRequired, Source,
                      discover, driver, prepare, register, root)
from ._files import Files, SchemaError
from ._index import Index


def load(name: str, split: str | None = None, *, prepare_first: bool = True) -> Dataset:
    """The dataset `name` (and one of its splits). Prepares it if needed: dependencies,
    downloads not done yet, the index if the driver changed, the check."""
    selected = driver(name)
    index = Index(selected.root)
    if prepare_first:
        state = selected._state()
        up_to_date = index.exists() and state.get("checked") and \
            index.meta.get("fingerprint") == selected.index_fingerprint() and \
            all(state.get("sources", {}).get(s.uri) for s in selected.sources
                if isinstance(s, Source))
        if not up_to_date:
            selected.prepare()
            index = Index(selected.root)
    elif not index.exists():
        raise DatasetError(f"{name!r} is not prepared: datasets.prepare({name!r})")
    dataset = Dataset(selected, index)
    return dataset.split(split) if split else dataset


def ls() -> list[dict]:
    """Every registered driver: prepared or not, records, size on disk."""
    out = []
    for name, cls in sorted(discover().items()):
        instance = cls(root() / name)
        index = Index(instance.root)
        size = sum(path.stat().st_size for path in instance.root.rglob("*") if path.is_file()) \
            if instance.root.exists() else 0
        prepared = index.exists()
        out.append({"name": name, "title": cls.card.title if cls.card else name,
                    "prepared": prepared,
                    "current": prepared and index.meta.get("fingerprint") == instance.index_fingerprint(),
                    "records": index.meta["count"] if prepared else None,
                    "splits": index.meta["splits"] if prepared else {},
                    "bytes": size, "root": str(instance.root)})
    return out


def from_spec(spec: dict):
    dataset = load(spec["name"], spec.get("split"))
    return dataset.field(spec["field"]) if spec.get("field") else dataset


__all__ = ["load", "ls", "prepare", "register", "driver", "discover", "root", "from_spec",
           "Driver", "Source", "Manual", "Files", "Card", "Dataset", "Index",
           "DatasetError", "ManualDownloadRequired", "SchemaError"]
