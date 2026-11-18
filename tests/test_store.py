from __future__ import annotations

import multiprocessing
import os
import sqlite3
import time

import numpy as np
import pytest

from useml import store as stores
from useml.store import Store, UndeclaredSpaceError


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


# ── values ────────────────────────────────────────────────────────────────────

def test_small_json_lives_inline(store):
    scores = store.declare("results/scores", cost="precious")
    assert scores.put({"b": 1, "a": 2}, {"accuracy": 0.41})
    assert scores.get({"a": 2, "b": 1}) == {"accuracy": 0.41}     # key order is irrelevant
    [entry] = store.entries()
    assert entry["file"] is None


def test_large_json_goes_to_a_file(store):
    space = store.declare("results/records")
    value = ["x" * 100] * 1000
    space.put("k", value)
    assert space.get("k") == value
    assert space.path("k").suffix == ".json"


def test_arrays(store):
    space = store.declare("processing/features", cost="cheap")
    floats = np.random.rand(10, 4).astype(np.float16)
    masks = {"masks": np.zeros((3, 64, 64), dtype=bool), "ids": np.arange(3)}
    space.put("floats", floats)
    space.put("masks", masks)
    assert np.array_equal(space.get("floats"), floats)
    got = space.get("masks")
    assert set(got) == {"masks", "ids"} and got["masks"].dtype == bool
    # booleans compress, floats are stored raw
    assert space.path("masks").stat().st_size < masks["masks"].nbytes / 4
    assert space.path("floats").stat().st_size >= floats.nbytes


def test_bytes_and_put_file(store):
    space = store.declare("generation/images")
    space.put("raw", b"\x00\x01")
    assert space.get("raw") == b"\x00\x01"
    calls = []

    def write(path):
        calls.append(path)
        path.write_text("png")

    first = space.put_file("img", write, extension=".png")
    second = space.put_file("img", write, extension=".png")
    assert first == second and first.read_text() == "png" and len(calls) == 1
    assert not list(first.parent.glob(".*tmp*"))


def test_one_write_per_key_unless_overwrite(store):
    space = store.declare("results/scores")
    assert space.put("k", 1)
    assert not space.put("k", 2)
    assert space.get("k") == 1
    assert space.put("k", 3, overwrite=True)
    assert space.get("k") == 3


def test_overwrite_removes_the_former_file(store):
    space = store.declare("results/records")
    space.put("k", np.zeros(3))
    former = space.path("k")
    space.put("k", ["x" * 100] * 1000, overwrite=True)
    assert not former.exists() and space.path("k").exists()


def test_missing_file_is_a_miss_and_the_entry_goes(store):
    space = store.declare("processing/masks")
    space.put("k", np.zeros(3))
    space.path("k").unlink()
    assert space.get("k", "default") == "default"
    assert not space.contains("k")


def test_meta_and_version(store):
    space = store.declare("results/scores")
    space.put("k", 1, version="2", meta={"model": "clip"})
    assert space.meta("k") == {"model": "clip"}
    assert store.entries()[0]["version"] == "2"


# ── spaces ────────────────────────────────────────────────────────────────────

def test_undeclared_space_is_refused(store):
    with pytest.raises(UndeclaredSpaceError, match="not declared"):
        store.put("results/scores", "k", 1)


def test_declaration_is_kept_in_the_index(tmp_path):
    Store(tmp_path / "s").declare("results/scores", cost="precious", description="d")
    other = Store(tmp_path / "s")
    assert other.declared()["results/scores"].cost == "precious"
    other.declare("results/scores", cost="cheap")
    assert Store(tmp_path / "s").declared()["results/scores"].cost == "cheap"


def test_bad_names_and_costs(store):
    with pytest.raises(ValueError):
        store.declare("scores")
    with pytest.raises(ValueError):
        store.declare("results/scores", cost="free")


def test_non_strict_store_writes_anywhere(tmp_path):
    loose = Store(tmp_path / "s", strict=False)
    loose.put("scratch/x", "k", 1)
    assert loose.get("scratch/x", "k") == 1


# ── accounts ──────────────────────────────────────────────────────────────────

def test_usage_counts(store):
    space = store.declare("results/scores")
    space.put("k", 1)
    space.get("k")
    space.get("k")
    space.get("absent")
    [row] = store.usage()
    assert (row["hits"], row["misses"], row["writes"]) == (2, 1, 1)
    assert store.entries()[0]["hits"] == 2


def test_file_digest_is_remembered_and_follows_content(store, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"a")
    first = store.file_digest(path)
    assert store.file_digest(path) == first
    time.sleep(0.01)
    path.write_bytes(b"b")
    assert store.file_digest(path) != first


# ── several processes ─────────────────────────────────────────────────────────

def _writer(root, worker):
    store = Store(root)
    space = store.space("results/scores")
    for index in range(50):
        space.put({"i": index}, {"worker": worker})
        space.get({"i": index})
    store.close()


def test_parallel_processes_write_each_key_once(tmp_path):
    root = tmp_path / "store"
    Store(root).declare("results/scores")
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_writer, args=(root, worker)) for worker in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    entries = Store(root).entries()
    assert len(entries) == 50
    assert sum(entry["hits"] for entry in entries) == 200


# ── gc ────────────────────────────────────────────────────────────────────────

def _fill(store):
    cheap = store.declare("processing/features", cost="cheap")
    expensive = store.declare("processing/masks", cost="expensive")
    precious = store.declare("results/scores", cost="precious")
    for index in range(3):
        cheap.put(index, np.zeros(1000, dtype=np.float64))      # ~8 KB each
        expensive.put(index, np.zeros(1000, dtype=np.float64))
        precious.put(index, np.zeros(1000, dtype=np.float64))
    return cheap, expensive, precious


def test_gc_evicts_cheap_first_and_never_precious(store):
    _fill(store)
    total = store.total_bytes()
    entry = store.entries()[0]["size"]
    plan = stores.garbage_plan(store, max_bytes=total - 4 * entry)
    assert [item["space"] for item in plan] == ["processing/features"] * 3 + \
        ["processing/masks"]
    plan = stores.garbage_plan(store, max_bytes=0)
    assert {item["space"] for item in plan} == {"processing/features", "processing/masks"}
    assert stores.collect_garbage(store, plan) == 6 * entry
    assert len(store.entries("results/scores")) == 3


def test_gc_keeps_pinned_entries(store):
    cheap, _, _ = _fill(store)
    cheap.pin(0, "experience-a")
    plan = stores.garbage_plan(store, max_bytes=0)
    assert ("processing/features", stores.key_digest(0)) not in \
        {(item["space"], item["digest"]) for item in plan}
    assert store.unpin_owner("experience-a") == 1


def test_gc_older_than(store):
    space = store.declare("processing/masks")
    space.put("old", 1)
    space.put("new", 2)
    with store._transaction() as connection:
        connection.execute("UPDATE entries SET accessed = ? WHERE key_json = ?",
                           (time.time() - 10 * 86400, '"old"'))
    plan = stores.garbage_plan(store, older_than_days=5)
    assert [item["key_json"] for item in plan] == ['"old"']


def test_named_precious_space_is_collectable(store):
    _fill(store)
    plan = stores.garbage_plan(store, spaces=["results/scores"], max_bytes=0)
    assert len(plan) == 3


# ── maintenance ───────────────────────────────────────────────────────────────

def test_check_and_repair(store):
    space = store.declare("processing/masks")
    space.put("a", np.zeros(3))
    space.put("b", np.zeros(4))
    space.path("a").unlink()
    orphan = store.files_root / "processing" / "masks" / "orphan.npz"
    orphan.write_bytes(b"x")
    temporary = store.files_root / "processing" / "masks" / ".x.npz.1.2.tmp"
    temporary.write_bytes(b"x")
    os.utime(temporary, (time.time() - 7200,) * 2)
    report = stores.check(store)
    assert (len(report["missing"]), len(report["orphans"]), len(report["temporaries"])) == \
        (1, 1, 1)
    stores.repair(store, report)
    assert not orphan.exists() and not temporary.exists()
    assert [entry["key_json"] for entry in store.entries()] == ['"b"']


def test_move(store, tmp_path):
    space = store.declare("processing/masks")
    space.put("k", np.ones(3))
    (store.managed("processing/masks") / "shard.npy").write_bytes(b"123")
    result = stores.move(store, tmp_path / "elsewhere")
    moved = Store(tmp_path / "elsewhere")
    assert result["entries"] == 1 and np.array_equal(moved.get("processing/masks", "k"),
                                                      np.ones(3))
    assert (moved.managed_root / "processing" / "masks" / "shard.npy").exists()
    assert store.root.exists()                      # source kept without delete_source
    with pytest.raises(FileExistsError):
        stores.move(store, tmp_path / "elsewhere")


def test_managed_folder_size_counts_hard_links_once(store):
    store.declare("training/problems")
    folder = store.managed("training/problems")
    (folder / "pool").mkdir()
    (folder / "pool" / "a").write_bytes(b"x" * 1000)
    os.link(folder / "pool" / "a", folder / "link")
    [row] = store.spaces()
    assert row["managed_bytes"] == 1000 and row["bytes"] == 1000


def test_adopt_a_former_folder(store, tmp_path):
    store.declare("processing/features")
    former = tmp_path / "old_cache" / "frozen"
    (former / "ns").mkdir(parents=True)
    (former / "ns" / "shard.npy").write_bytes(b"1")
    report = stores.adopt(store, "processing/features", former)
    assert report["moved"] == 1 and not former.exists()
    assert (store.managed("processing/features") / "ns" / "shard.npy").exists()


# ── compatibility with schema-2 indexes ────────────────────────────────────────

def test_a_schema_2_index_is_upgraded_in_place(tmp_path):
    root = tmp_path / "old_store"
    root.mkdir()
    connection = sqlite3.connect(root / "index.sqlite")
    connection.executescript("""
        CREATE TABLE entries (space TEXT NOT NULL, digest TEXT NOT NULL,
            key_json TEXT NOT NULL, kind TEXT NOT NULL, value TEXT, file TEXT,
            size INTEGER NOT NULL, created REAL NOT NULL, accessed REAL NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0, version TEXT, meta TEXT,
            PRIMARY KEY (space, digest));
        CREATE TABLE settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO settings VALUES ('schema', '2');
    """)
    key = {"scorer": "clip"}
    connection.execute("INSERT INTO entries VALUES (?, ?, ?, 'json', '0.41', NULL, 4, 0, 0, "
                       "0, NULL, NULL)", ("results/scores", stores.key_digest(key),
                                          stores.canonical_key(key)))
    connection.commit()
    connection.close()
    store = Store(root)
    assert store.get("results/scores", key) == 0.41
    assert store.setting("schema") == 3
    store.declare("results/scores", cost="precious")
    assert store.spaces()[0]["cost"] == "precious"


# ── entry point and CLI ───────────────────────────────────────────────────────

def test_open_uses_the_home_and_the_config(root, tmp_path):
    assert stores.open().root == (root / "store").resolve()
    (tmp_path / "config.toml").write_text(
        f'[store]\nroot = "{tmp_path / "s2"}"\nmax_size = "1.5GB"\n')
    assert stores.default_root() == tmp_path / "s2"
    assert stores.budget() == int(1.5 * (1 << 30))


def test_parse_size():
    assert stores.parse_size("450GB") == 450 * (1 << 30)
    assert stores.parse_size("200g") == 200 * (1 << 30)
    assert stores.parse_size(12) == 12
    with pytest.raises(ValueError):
        stores.parse_size("lots")


def test_cli_store(tmp_path, capsys):
    from useml.cli import main

    store = Store(tmp_path / "s")
    _fill(store)
    store.close()
    arguments = ["store", "--root", str(tmp_path / "s")]
    assert main(arguments + ["info"]) == 0
    assert main(arguments + ["spaces"]) == 0
    assert main(arguments + ["gc", "--max-size", "0"]) == 0
    out = capsys.readouterr().out
    assert "dry run: 6 entries" in out and "results/scores" in out
    assert main(arguments + ["gc", "--max-size", "0", "--apply"]) == 0
    assert len(stores.open(tmp_path / "s").entries()) == 3
