from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import data_procs  # noqa: E402
from useml import _progress, data  # noqa: E402
from useml.data._keys import DIGEST_SIZE  # noqa: E402
from useml.data._table import Table  # noqa: E402


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("USEML_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("USEML_ROOT", str(tmp_path / "root"))
    data_procs.CALLS.update(init=0, items=0)
    return tmp_path / "data"


def _images(folder: Path, contents) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, content in enumerate(contents):
        path = folder / f"{index}.jpg"
        path.write_bytes(content)
        paths.append(path)
    return paths


# ── process and load ──────────────────────────────────────────────────────────

def test_process_then_load_by_name():
    captions = data_procs.captions(10)
    view = data.process(data.texts(captions), data_procs.length, name="caps-length")
    assert len(view) == 10 and view.complete
    expected = [[len(text), len(text.split())] for text in captions]
    assert np.allclose(view[:], expected)
    loaded = data.load("caps-length")
    assert np.allclose(loaded[3], expected[3])
    assert np.allclose(loaded[[9, 0]], [expected[9], expected[0]])
    assert loaded.numpy().shape == (10, 2)


def test_nothing_missing_means_no_model_and_no_compute():
    paths = _images(Path(os.environ["USEML_DATA_ROOT"]).parent / "imgs",
                    [b"a" * 10, b"b" * 20, b"c" * 30])
    data.process(data.images(paths), data_procs.BytesEmbed)
    assert data_procs.CALLS == {"init": 1, "items": 3}
    data.process(data.images(paths), data_procs.BytesEmbed)
    assert data_procs.CALLS == {"init": 1, "items": 3}          # model never loaded again


def test_only_new_items_are_computed():
    data.process(data.texts(data_procs.captions(10)), data_procs.length)
    data_procs.CALLS["items"] = 0
    data.process(data.texts(data_procs.captions(15)), data_procs.length)
    assert data_procs.CALLS["items"] == 5


def test_same_image_in_two_datasets_is_computed_once(tmp_path):
    first = _images(tmp_path / "first", [b"x" * 50, b"y" * 60])
    second = _images(tmp_path / "second", [b"y" * 60, b"z" * 70])     # shares one image
    data.process(data.images(first), data_procs.BytesEmbed, name="first")
    data_procs.CALLS["items"] = 0
    other = data.process(data.images(second), data_procs.BytesEmbed, name="second")
    assert data_procs.CALLS["items"] == 1
    assert np.allclose(other[0], data.load("first")[1])


def test_duplicates_inside_a_source_are_computed_once():
    data.process(data.texts(["same", "same", "other"]), data_procs.length)
    assert data_procs.CALLS["items"] == 2


def test_parameters_and_version_make_another_table():
    a = data.process(data.texts(["x"]), data_procs.length)
    b = data.process(data.texts(["x"]), data_procs.length, scale=2.0)
    c = data.process(data.texts(["x"]), data_procs.length, scale=1.0)   # the default
    assert a.table.path != b.table.path and a.table.path == c.table.path
    assert np.allclose(b[0], [2.0, 1.0])


def test_ignored_parameters_do_not_change_the_table(tmp_path):
    paths = _images(tmp_path / "i", [b"q" * 9])
    a = data.process(data.images(paths), data_procs.BytesEmbed, device="cpu")
    b = data.process(data.images(paths), data_procs.BytesEmbed, device="cuda:3")
    assert a.table.path == b.table.path


# ── storage ───────────────────────────────────────────────────────────────────

def test_float16_and_int8_storage():
    view = data.process(data.texts(data_procs.captions(7)), data_procs.multi)
    assert view.outputs["vector"]["stored"] == "int8"
    assert view.outputs["count"]["stored"] == "int32"
    values = view[:]
    expected = np.array([len(text) for text in data_procs.captions(7)])
    assert np.array_equal(values["count"], expected)
    assert values["vector"].dtype == np.float32
    assert np.abs(values["vector"].mean(axis=1) - expected).max() < 1.0   # int8 is close
    shard = view.table.shards()[0]
    stored = np.load(view.table.shards_path / f"{shard}.vector.npy")
    assert stored.dtype == np.int8


def test_index_resolves_shared_8_byte_prefixes(tmp_path):
    table = Table(tmp_path / "t")
    table.write_manifest({"outputs": {"data": {"shape": [1], "dtype": "float32",
                                               "stored": "float32"}}})
    digests = np.zeros((3, DIGEST_SIZE), dtype=np.uint8)
    digests[:, 15] = [1, 2, 3]                 # same first 8 bytes for all three
    with table.lock():
        writer = table.writer()
        writer.add(digests, {"data": np.array([[10.0], [20.0], [30.0]])})
        writer.flush()
    query = digests[[2, 0, 1]]
    rows = table.locate(query)
    assert np.allclose(table.read(rows, "data").ravel(), [30, 10, 20])
    other = digests[:1].copy()
    other[0, 15] = 9
    assert table.locate(other)[0] == -1


def test_incomplete_shard_is_ignored_then_removed(tmp_path):
    view = data.process(data.texts(["a", "b"]), data_procs.length)
    leftover = view.table.shards_path / "99999999999999999999-dead.data.npy"
    np.save(leftover, np.zeros((2, 2)))
    assert len(view.table.shards()) == 1
    data.process(data.texts(["c"]), data_procs.length)              # a writer cleans up
    assert not leftover.exists()


def test_interrupted_run_keeps_its_rows_and_resumes():
    @data.processor("fragile", batch_size=2, dtype="float32")
    def fragile(texts, *, fail_after=None):
        data_procs.CALLS["items"] += len(texts)
        if fail_after is not None and data_procs.CALLS["items"] > fail_after:
            raise RuntimeError("boom")
        return np.array([[len(text)] for text in texts], dtype=np.float32)

    texts = data.texts(data_procs.captions(10))
    with pytest.raises(RuntimeError, match="boom"):
        data.process(texts, fragile, fail_after=4)
    # `fail_after` is a parameter: the resumed run must use the SAME table.
    table = data.table_for(fragile, {"fail_after": 4})
    assert len(table) == 4
    data_procs.CALLS["items"] = 0
    fragile_ok = data.process(texts, fragile, fail_after=4 + 10 ** 6)
    assert fragile_ok.table.path != table.path       # another parameter, another table
    data_procs.CALLS["items"] = -10 ** 6             # resume the first table now
    view = data.process(texts, fragile, fail_after=4)
    assert view.complete and len(view.table) == 10


def test_shape_change_is_refused():
    @data.processor("shapeless", batch_size=2, dtype="float32")
    def shapeless(texts):
        return np.zeros((len(texts), len(texts[0])))

    with pytest.raises(ValueError, match="Fixed-size rows"):
        data.process(data.texts(["ab", "cd", "abc", "efg"]), shapeless)


def test_wrong_row_count_is_refused():
    @data.processor("short", dtype="float32")
    def short(texts):
        return np.zeros((1, 2))

    with pytest.raises(ValueError, match="one row per item"):
        data.process(data.texts(["a", "b"]), short)


def test_budget_refuses_before_writing(tmp_path):
    (tmp_path / "config.toml").write_text('[data]\nmax_size = "10B"\n')
    with pytest.raises(data.BudgetError, match="over budget"):
        data.process(data.texts(data_procs.captions(10)), data_procs.length)
    table = data.table_for(data_procs.length, {})
    assert table.shards() == []


# ── the view ──────────────────────────────────────────────────────────────────

def test_batches_cover_everything_once():
    view = data.process(data.texts(data_procs.captions(23)), data_procs.length)
    seen = []
    for positions, rows in view.batches(5, shuffle=True, seed=0, progress=False):
        assert np.allclose(rows, view[positions])
        seen.extend(positions.tolist())
    assert sorted(seen) == list(range(23))


def test_rows_never_processed_raise(tmp_path):
    view = data.process(data.texts(["a"]), data_procs.length)
    from useml.data._keys import digests
    from useml.data._view import Processed

    partial = Processed(view.table, digests(["text:unknown", "item:x"]))
    assert partial.missing == 2
    with pytest.raises(KeyError, match="never processed"):
        partial[0]


def test_load_unknown_name():
    with pytest.raises(KeyError, match="no processed dataset"):
        data.load("nope")


def test_ls_tables_and_rm():
    data.process(data.texts(["a", "b"]), data_procs.length, name="one")
    data.process(data.texts(["a"]), data_procs.length, name="two")
    assert {entry["name"] for entry in data.ls()} == {"one", "two"}
    [table] = data.tables()
    assert sorted(table["names"]) == ["one", "two"]
    assert data.rm("one", table=True) == 0                # still used by "two"
    assert data.rm("two", table=True) > 0
    assert data.tables() == [] and data.ls() == []


# ── several processes ─────────────────────────────────────────────────────────

def _process_captions(root, log):
    os.environ["USEML_DATA_ROOT"] = root
    os.environ["USEML_TEST_LOG"] = log
    os.environ["USEML_NO_AUTO"] = "1"
    sys.path.insert(0, str(Path(__file__).parent))
    import data_procs as procs
    from useml import data as d

    d.process(d.texts(procs.captions(40)), procs.length)


def test_two_processes_compute_each_row_once(tmp_path):
    log = tmp_path / "computed.log"
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_process_captions,
                               args=(os.environ["USEML_DATA_ROOT"], str(log)))
               for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    assert sum(int(line) for line in log.read_text().split()) == 40
    assert len(data.table_for(data_procs.length, {})) == 40


# ── detached jobs ─────────────────────────────────────────────────────────────

def test_detached_job(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent) + os.pathsep +
                       os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("USEML_NO_AUTO", "1")
    job = data.process(data.source("data_procs:captions", count=12), data_procs.length,
                       name="detached", detach=True)
    assert job.follow(poll=0.1) == "done", job.log()
    view = job.result()
    assert len(view) == 12 and view.complete
    assert job.state["done"] == 12
    assert data.jobs()[0].id == job.id


def test_detach_refuses_what_cannot_be_rebuilt():
    with pytest.raises(ValueError, match="cannot rebuild"):
        data.process(data.items([1, 2], key=str), data_procs.length, detach=True)

    @data.processor("local")
    def local(texts):
        return np.zeros((len(texts), 1))

    with pytest.raises(ValueError, match="cannot import"):
        data.process(data.texts(["a"]), local, detach=True)


# ── feedback ──────────────────────────────────────────────────────────────────

def test_step_is_quiet_when_quick_and_speaks_when_slow(capsys):
    with _progress.step("quick", after=1.0):
        pass
    assert capsys.readouterr().err == ""
    with _progress.step("slow", after=0.05):
        time.sleep(0.15)
    err = capsys.readouterr().err
    assert "⏳ slow" in err and "✓ slow" in err


def test_cli_data(capsys):
    from useml.cli import main

    data.process(data.texts(["a", "bb"]), data_procs.length, name="cli")
    assert main(["data", "ls"]) == 0
    assert main(["data", "tables"]) == 0
    assert main(["data", "info", "cli"]) == 0
    out = capsys.readouterr().out
    assert "cli" in out and "missing rows: 0" in out
