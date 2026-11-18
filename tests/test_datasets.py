from __future__ import annotations

import http.server
import io
import os
import sys
import textwrap
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).parent
sys.path.insert(0, str(TESTS))

import data_procs  # noqa: E402
import dataset_drivers as drivers  # noqa: E402
from useml import config, data, datasets as ds, exp  # noqa: E402
from useml.exp import RefError  # noqa: E402


@pytest.fixture(autouse=True)
def roots(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.useml.datasets]
        packages = ["dataset_drivers"]
        [tool.useml.exp]
        root = "experiences"
    """))
    monkeypatch.chdir(project)
    monkeypatch.setenv("USEML_DATASETS_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("USEML_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("USEML_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    monkeypatch.setenv("USEML_NO_TMUX", "1")
    monkeypatch.setenv("PYTHONPATH", str(TESTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(config, "_DEFAULT", None)
    data_procs.CALLS.update(init=0, items=0)
    return tmp_path / "datasets"


def shapes(roots, count=10, **options):
    drivers.make_shapes(roots / "shapes", count, **options)
    return ds.load("shapes")


# ── loading and reading ───────────────────────────────────────────────────────

def test_load_prepares_and_reads_records(roots):
    dataset = shapes(roots, 10)
    assert len(dataset) == 10
    first = dataset[0]
    assert first["id"] == "s0" and first["text"] == "shape number 0"
    assert isinstance(first["image"], Path) and first["image"].read_bytes().startswith(b"image-0")
    assert [r["id"] for r in dataset[2:4]] == ["s2", "s3"]
    assert [r["id"] for r in dataset[[5, 1]]] == ["s5", "s1"]
    assert [r["id"] for r in dataset][:3] == ["s0", "s1", "s2"]
    assert (roots / "shapes" / ".useml" / "index" / "records.jsonl").exists()


def test_summary_explains_the_fields(roots):
    text = shapes(roots, 12).summary()
    assert "Shapes — shapes: 12 records" in text and "small synthetic images" in text
    assert "label" in text and "class 0-2" in text and "(paths)" in text


def test_filter_and_field(roots):
    dataset = shapes(roots, 9)
    zeros = dataset.filter(lambda r: r["label"] == 0, name="zeros")
    assert [r["id"] for r in zeros] == ["s0", "s3", "s6"]
    images = dataset.field("image")
    assert isinstance(images[0], Path) and len(images) == 9
    with pytest.raises(KeyError, match="no field"):
        dataset.field("nope")


def test_duplicate_ids_are_refused(roots):
    drivers.make_shapes(roots / "shapes", 3)
    drivers.make_shapes(roots / "shapes", 2)          # appends s0, s1 again
    with pytest.raises(ds.DatasetError, match="two records with id"):
        ds.load("shapes")


# ── declarations fail loudly ──────────────────────────────────────────────────

def test_missing_column_raises_before_indexing(roots):
    drivers.make_shapes(roots / "broken", 3)
    with pytest.raises(ds.SchemaError, match="'captoin' \\(close to caption\\)"):
        ds.load("broken")
    assert not (roots / "broken" / ".useml" / "index").exists()


def test_manual_source_says_what_to_do(roots):
    with pytest.raises(ds.ManualDownloadRequired, match="(?s)make_shapes.*useml datasets prepare"):
        ds.load("shapes")


def test_unknown_dataset_suggests(roots):
    with pytest.raises(KeyError, match="Did you mean shapes"):
        ds.load("shapez")


# ── splits ────────────────────────────────────────────────────────────────────

def test_hash_split_is_stable_salted_and_survives_growth(roots):
    dataset = shapes(roots, 200)
    train = {r["id"] for r in dataset.split("train")}
    test = {r["id"] for r in dataset.split("test")}
    assert not train & test and len(train | test) == 200
    assert 130 < len(train) < 190                       # ≈ 80 %
    drivers.make_shapes(roots / "shapes-copy", 200)
    copy_train = {r["id"] for r in ds.load("shapes-copy", "train")}
    assert copy_train != train                          # another name, another partition
    drivers.make_shapes(roots / "shapes", 50, offset=200)
    grown = {r["id"] for r in ds.load("shapes", "train")}
    assert train <= grown                               # nobody changed side


def test_native_split(roots):
    drivers.make_shapes(roots / "shapes-native", 8, native_split=True)
    dataset = ds.load("shapes-native")
    assert dataset.splits == {"test": 2, "train": 6}
    assert [r["id"] for r in dataset.split("test")] == ["s0", "s4"]
    with pytest.raises(KeyError, match="no split 'val'"):
        dataset.split("val")


# ── preparation invalidates itself ────────────────────────────────────────────

def test_changing_the_driver_rebuilds_the_index(roots, monkeypatch, capsys):
    shapes(roots, 4)
    capsys.readouterr()
    assert ds.load("shapes") is not None
    assert "building the index" not in capsys.readouterr().err   # up to date: nothing
    monkeypatch.setattr(drivers.Shapes, "version", 2)
    ds.load("shapes")
    assert "the driver changed" in capsys.readouterr().err
    monkeypatch.setattr(drivers.Shapes, "hash_splits", {"train": 0.5, "test": 0.5})
    ds.load("shapes")
    assert "the driver changed" in capsys.readouterr().err


def test_changing_process_code_rebuilds(roots, monkeypatch, capsys):
    shapes(roots, 4)
    capsys.readouterr()

    def process(self):
        for record in ds.Driver.process(self):
            record["text"] = record["text"].upper()
            yield record

    monkeypatch.setattr(drivers.Shapes, "process", process)
    assert ds.load("shapes")[0]["text"] == "SHAPE NUMBER 0"
    assert "the driver changed" in capsys.readouterr().err


@pytest.fixture
def server(tmp_path):
    folder = tmp_path / "www"
    folder.mkdir()
    hits = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            hits.append(self.path)

    def handler(*args, **kwargs):
        return Handler(*args, directory=str(folder), **kwargs)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield folder, f"http://127.0.0.1:{httpd.server_address[1]}", hits
    httpd.shutdown()
    httpd.server_close()


def test_http_source_is_fetched_once_and_the_archive_removed(server, roots):
    folder, base, hits = server
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("table.csv", "image_id,caption\na,first\nb,second\n")
    (folder / "pack.zip").write_bytes(buffer.getvalue())

    class Remote(ds.Driver):
        name = "remote"
        sources = [ds.Source(f"{base}/pack.zip", extract=True)]
        files = [ds.Files("table.csv", columns={"id": "image_id", "text": "caption"})]

    ds.register(Remote)
    dataset = ds.load("remote")
    assert [r["text"] for r in dataset] == ["first", "second"]
    assert not list((roots / "remote" / ".useml" / "downloads").glob("*.zip"))
    count = len(hits)
    ds.load("remote")
    Remote.version = 2                                  # the index changes, not the source
    ds.load("remote")
    assert len(hits) == count                           # never fetched again


def test_dependencies_are_prepared_first(roots):
    drivers.make_shapes(roots / "shapes", 6)
    pairs = ds.load("pairs")
    assert len(pairs) == 3 and pairs[0]["texts"] == ["shape number 0", "shape number 1"]
    assert pairs[0]["left"].exists()


# ── with useml.data and useml.exp ─────────────────────────────────────────────

def test_a_dataset_field_is_a_data_source(roots):
    blobs = [f"img{i}".encode() * 8 for i in range(6)]
    drivers.make_shapes(roots / "shapes", 6, blobs=blobs)
    drivers.make_shapes(roots / "shapes-copy", 6, blobs=blobs)     # the same images
    feats = data.process(ds.load("shapes").field("image"), data_procs.BytesEmbed,
                         name="shapes-bytes")
    assert len(feats) == 6 and data_procs.CALLS["items"] == 6
    data.process(ds.load("shapes-copy").field("image"), data_procs.BytesEmbed)
    assert data_procs.CALLS["items"] == 6                          # shared content: nothing new


def test_detached_job_on_a_dataset_split(roots):
    drivers.make_shapes(roots / "shapes", 20)
    split = ds.load("shapes", "train").field("text")
    job = data.process(split, data_procs.length, name="shape-lengths", detach=True)
    assert job.follow(poll=0.1) == "done", job.log()
    assert len(job.result()) == len(split)


def test_experience_receives_a_dataset_by_name(roots):
    import exp_steps

    drivers.make_shapes(roots / "shapes", 10)
    test_split = ds.load("shapes", "test")
    with exp.experience("ds/count", detach=False) as e:
        e.step(exp_steps.identity, {"n": len(test_split)}, name="expected")
        e.step(exp_steps.rows_of, test_split, name="count")
    assert exp.get("ds/count/count") == len(test_split)
    with pytest.raises(RefError, match="filtered dataset cannot be rebuilt"):
        with exp.experience("ds/bad", run=False) as e:
            e.step(exp_steps.rows_of, test_split.filter(lambda r: True))


def test_cli(roots, capsys):
    from useml.cli import main

    drivers.make_shapes(roots / "shapes", 5)
    assert main(["datasets", "prepare", "shapes"]) == 0
    assert main(["datasets", "ls"]) == 0
    assert main(["datasets", "info", "shapes", "--split", "train"]) == 0
    assert main(["datasets", "check", "shapes"]) == 0
    out = capsys.readouterr().out
    assert "ready" in out and "shapes" in out and "Shapes — shapes/train" in out


def test_types_convert_and_report_the_line(roots):
    dataset = shapes(roots, 3)
    assert dataset[1]["label"] == 1
    (roots / "shapes" / "annotations.csv").write_text(
        "image_id,caption,file,label\ns0,a,0.png,1\ns1,b,1.png,two\n")
    with pytest.raises(ds.SchemaError, match="record 2: label = 'two' is not int"):
        ds.load("shapes")
