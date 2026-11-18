"""Drivers used by the dataset tests (importable by detached jobs and experiences)."""
from __future__ import annotations

import csv
from pathlib import Path

from useml import datasets as ds


def make_shapes(folder: Path, count: int, *, offset: int = 0, native_split: bool = False,
                blobs=None) -> None:
    """Raw files of a small dataset: images/<i>.png (distinct bytes) and annotations.csv."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(offset, offset + count):
        blob = blobs[index] if blobs else f"image-{index}".encode() * 4
        (folder / "images" / f"{index}.png").write_bytes(blob)
        row = {"image_id": f"s{index}", "caption": f"shape number {index}",
               "file": f"{index}.png", "label": index % 3}
        if native_split:
            row["split"] = "train" if index % 4 else "test"
        rows.append(row)
    path = folder / "annotations.csv"
    existing = list(csv.DictReader(open(path))) if path.exists() else []
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(existing + rows)


@ds.register
class Shapes(ds.Driver):
    name = "shapes"
    version = 1
    card = ds.Card(title="Shapes", summary="small synthetic images with a caption",
                   fields={"text": "the caption", "label": "class 0-2"})
    sources = [ds.Manual("annotations.csv", hint="make_shapes(root, n) writes it")]
    files = [ds.Files("annotations.csv",
                      columns={"id": "image_id", "text": "caption", "image": "file",
                               "label": "label"},
                      paths={"image": "images"}, types={"label": int})]
    hash_splits = {"train": 0.8, "test": 0.2}
    key = "image"


@ds.register
class ShapesCopy(Shapes):
    """The same files under another name: another salt for the hash split."""
    name = "shapes-copy"


@ds.register
class ShapesNative(ds.Driver):
    name = "shapes-native"
    sources = [ds.Manual("annotations.csv")]
    files = [ds.Files("annotations.csv",
                      columns={"id": "image_id", "text": "caption", "image": "file",
                               "split": "split"},
                      paths={"image": "images"})]
    split_field = "split"
    key = "image"


@ds.register
class Pairs(ds.Driver):
    """Its own format (process overridden) and a dependency on `shapes`."""
    name = "pairs"
    depends = ("shapes",)
    path_fields = ("left",)

    def process(self):
        shapes = ds.load("shapes")
        for index in range(0, len(shapes) - 1, 2):
            first, second = shapes[index], shapes[index + 1]
            yield {"id": f"p{index}", "left": str(first["image"].relative_to(self.root.parent)
                                                  .as_posix()).replace("shapes/", "../shapes/", 1),
                   "texts": [first["text"], second["text"]]}

    def check(self, dataset):
        return [] if len(dataset) else ["empty"]


@ds.register
class Broken(ds.Driver):
    name = "broken"
    sources = [ds.Manual("annotations.csv")]
    files = [ds.Files("annotations.csv", columns={"id": "image_id", "text": "captoin"})]
