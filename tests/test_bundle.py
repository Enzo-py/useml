"""Tests for useml.DataBundle (Level-1 data contract)."""
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn

import useml
from useml.dataset.bundle import _TransformWrapper, _MaterializedDataset, _sha256


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    useml.init(str(vault))
    useml.new("bundle-project")
    yield tmp_path
    useml._session.vault = None
    useml._session._project = None
    useml._session.components = {}
    useml._session._is_dirty = False


class _TinyDS(torch.utils.data.Dataset):
    """20-sample in-memory dataset of random (x4, y_binary) pairs."""
    def __init__(self):
        torch.manual_seed(0)
        self.x = torch.randn(20, 4)
        self.y = torch.randint(0, 2, (20,))

    def __len__(self):
        return 20

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _scale(sample):
    x, y = sample
    return x * 0.5, y


def _cfg(**kw):
    defaults = dict(epochs=1, checkpoint_every=9999, loss="cross_entropy", batch_size=4)
    defaults.update(kw)
    return useml.Config(**defaults)


# ---------------------------------------------------------------------------
# 1. Hash stability
# ---------------------------------------------------------------------------

class TestHashes:
    def test_builtin_source_hash_is_deterministic(self):
        b1 = useml.DataBundle("a", source="mnist")
        b2 = useml.DataBundle("b", source="mnist")
        assert b1.source_hash() == b2.source_hash()

    def test_builtin_source_hash_differs_by_name(self):
        h_mnist = useml.DataBundle("a", source="mnist").source_hash()
        h_cifar = useml.DataBundle("b", source="cifar10").source_hash()
        assert h_mnist != h_cifar

    def test_custom_dataset_source_hash_is_deterministic(self):
        ds = _TinyDS()
        b1 = useml.DataBundle("a", source=ds)
        b2 = useml.DataBundle("b", source=ds)
        assert b1.source_hash() == b2.source_hash()

    def test_transform_hash_none_is_literal_none(self):
        b = useml.DataBundle("a", source="mnist", transform=None)
        assert b.transform_hash() == "none"

    def test_transform_hash_is_deterministic(self):
        b1 = useml.DataBundle("a", source="mnist", transform=_scale)
        b2 = useml.DataBundle("b", source="mnist", transform=_scale)
        assert b1.transform_hash() == b2.transform_hash()

    def test_transform_hash_differs_by_function(self):
        def double(s):
            x, y = s; return x * 2, y

        h_scale = useml.DataBundle("a", source="mnist", transform=_scale).transform_hash()
        h_double = useml.DataBundle("b", source="mnist", transform=double).transform_hash()
        assert h_scale != h_double

    def test_cache_key_combines_both_hashes(self):
        b = useml.DataBundle("a", source="mnist", transform=_scale)
        expected = _sha256(b.source_hash() + b.transform_hash())[:16]
        assert b.cache_key() == expected

    def test_cache_key_differs_when_transform_changes(self):
        def other(s):
            x, y = s; return x + 1, y

        k1 = useml.DataBundle("a", source="mnist", transform=_scale).cache_key()
        k2 = useml.DataBundle("b", source="mnist", transform=other).cache_key()
        assert k1 != k2


# ---------------------------------------------------------------------------
# 2. Transform source archival
# ---------------------------------------------------------------------------

class TestTransformSource:
    def test_no_transform_returns_none(self):
        b = useml.DataBundle("a", source="mnist")
        assert b.transform_source() is None

    def test_module_level_function_source_captured(self):
        b = useml.DataBundle("a", source="mnist", transform=_scale)
        src = b.transform_source()
        assert src is not None
        assert "def _scale" in src

    def test_inline_source_key_includes_bundle_name_and_fn_name(self):
        b = useml.DataBundle("my_ds", source="mnist", transform=_scale)
        key = b.inline_source_key()
        assert "my_ds" in key
        assert "_scale" in key
        assert key.endswith(".py")

    def test_inline_source_key_with_no_transform(self):
        b = useml.DataBundle("my_ds", source="mnist")
        key = b.inline_source_key()
        assert key.endswith(".py")


# ---------------------------------------------------------------------------
# 3. to_meta_dict
# ---------------------------------------------------------------------------

class TestMetaDict:
    def test_required_keys_present(self):
        b = useml.DataBundle("ds", source="mnist", transform=_scale, version="v1")
        d = b.to_meta_dict()
        for key in ("name", "source", "source_hash", "transform", "transform_hash",
                    "cache_key", "version"):
            assert key in d, f"missing key: {key}"

    def test_version_absent_when_none(self):
        b = useml.DataBundle("ds", source="mnist")
        assert "version" not in b.to_meta_dict()

    def test_transform_none_recorded(self):
        b = useml.DataBundle("ds", source="mnist")
        assert b.to_meta_dict()["transform"] is None

    def test_source_custom_dataset_uses_class_name(self):
        b = useml.DataBundle("ds", source=_TinyDS())
        assert b.to_meta_dict()["source"] == "_TinyDS"


# ---------------------------------------------------------------------------
# 4. _TransformWrapper
# ---------------------------------------------------------------------------

class TestTransformWrapper:
    def test_len_preserved(self):
        ds = _TinyDS()
        wrapped = _TransformWrapper(ds, _scale)
        assert len(wrapped) == len(ds)

    def test_transform_applied(self):
        ds = _TinyDS()
        wrapped = _TransformWrapper(ds, _scale)
        x_orig, _ = ds[0]
        x_wrap, _ = wrapped[0]
        assert torch.allclose(x_wrap, x_orig * 0.5)


# ---------------------------------------------------------------------------
# 5. _MaterializedDataset
# ---------------------------------------------------------------------------

class TestMaterializedDataset:
    def test_all_samples_materialised(self):
        ds = _TinyDS()
        mat = _MaterializedDataset(ds)
        assert len(mat) == len(ds)

    def test_samples_equal(self):
        ds = _TinyDS()
        mat = _MaterializedDataset(ds)
        for i in range(len(ds)):
            x_orig, y_orig = ds[i]
            x_mat, y_mat = mat[i]
            assert torch.equal(x_orig, x_mat)
            assert torch.equal(y_orig, y_mat)


# ---------------------------------------------------------------------------
# 6. loaders()
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_returns_two_dataloaders(self):
        ds = _TinyDS()
        bundle = useml.DataBundle("ds", source=ds, transform=_scale)
        cfg = _cfg()
        train_loader, val_loader = bundle.loaders(cfg)
        assert hasattr(train_loader, "__iter__")
        assert hasattr(val_loader, "__iter__")

    def test_transform_applied_in_loader(self):
        ds = _TinyDS()
        bundle_with = useml.DataBundle("ds", source=ds, transform=_scale)
        bundle_without = useml.DataBundle("ds_raw", source=ds)
        cfg = _cfg()
        loader_with, _ = bundle_with.loaders(cfg)
        loader_without, _ = bundle_without.loaders(cfg)
        # _scale multiplies x by 0.5, so max-abs of transformed batch < raw batch
        batch_raw, _ = next(iter(loader_without))
        batch_scaled, _ = next(iter(loader_with))
        assert batch_scaled.abs().max() < batch_raw.abs().max()

    def test_no_transform_still_produces_loaders(self):
        ds = _TinyDS()
        bundle = useml.DataBundle("ds", source=ds)
        cfg = _cfg()
        train_loader, val_loader = bundle.loaders(cfg)
        assert sum(len(b[0]) for b in train_loader) > 0

    def test_cache_creates_file(self, tmp_path):
        ds = _TinyDS()
        bundle = useml.DataBundle("ds", source=ds, transform=_scale, cache=True)
        cfg = useml.Config(
            epochs=1, loss="cross_entropy", batch_size=4,
            data_dir=str(tmp_path), checkpoint_every=9999,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle.loaders(cfg)

        cache_dir = tmp_path / ".bundle_cache"
        assert cache_dir.exists()
        cache_files = list(cache_dir.iterdir())
        assert len(cache_files) == 1
        assert cache_files[0].name == f"{bundle.cache_key()}.pt"

    def test_cache_reused_on_second_call(self, tmp_path):
        ds = _TinyDS()
        bundle = useml.DataBundle("ds", source=ds, transform=_scale, cache=True)
        cfg = useml.Config(
            epochs=1, loss="cross_entropy", batch_size=4,
            data_dir=str(tmp_path), checkpoint_every=9999,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle.loaders(cfg)  # first call — builds cache

        # Second call should produce no warnings (pure cache hit)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            bundle.loaders(cfg)
        assert len(record) == 0


# ---------------------------------------------------------------------------
# 7. Integration with useml.train() → snapshot metadata
# ---------------------------------------------------------------------------

class TestSnapshotIntegration:
    def test_metadata_yaml_contains_data_key(self, tmp_vault):
        import yaml

        ds = _TinyDS()
        bundle = useml.DataBundle("tiny", source=ds, transform=_scale, version="v1")
        cfg = _cfg(checkpoint_every=1)

        useml.train(_TinyNet, bundle, config=cfg)

        snap = useml.focus("bundle-project")[0]
        meta = yaml.safe_load((snap.path / "metadata.yaml").read_text())

        assert "data" in meta
        d = meta["data"]
        assert d["name"] == "tiny"
        assert d["version"] == "v1"
        assert d["transform"] == "_scale"
        assert "source_hash" in d
        assert "transform_hash" in d
        assert "cache_key" in d

    def test_transform_source_archived_in_snapshot(self, tmp_vault):
        ds = _TinyDS()
        bundle = useml.DataBundle("tiny", source=ds, transform=_scale)
        cfg = _cfg(checkpoint_every=1)

        useml.train(_TinyNet, bundle, config=cfg)

        snap = useml.focus("bundle-project")[0]
        transform_file = snap.path / "source" / bundle.inline_source_key()
        assert transform_file.exists()
        assert "def _scale" in transform_file.read_text()

    def test_no_data_key_without_bundle(self, tmp_vault):
        import yaml

        ds = _TinyDS()
        cfg = _cfg(checkpoint_every=1)
        useml.train(_TinyNet, ds, config=cfg)

        snap = useml.focus("bundle-project")[0]
        meta = yaml.safe_load((snap.path / "metadata.yaml").read_text())
        assert "data" not in meta

    def test_hashes_are_stable_across_runs(self, tmp_vault):
        import yaml

        ds = _TinyDS()
        bundle1 = useml.DataBundle("tiny", source=ds, transform=_scale)
        bundle2 = useml.DataBundle("tiny", source=ds, transform=_scale)
        cfg = _cfg(checkpoint_every=1)

        useml.train(_TinyNet, bundle1, config=cfg)
        useml.train(_TinyNet, bundle2, config=cfg)

        snaps = useml.focus("bundle-project").log()
        metas = [
            yaml.safe_load((s.path / "metadata.yaml").read_text())["data"]
            for s in snaps
        ]
        assert metas[0]["source_hash"] == metas[1]["source_hash"]
        assert metas[0]["transform_hash"] == metas[1]["transform_hash"]
        assert metas[0]["cache_key"] == metas[1]["cache_key"]

    def test_manual_commit_passes_bundle_meta(self, tmp_vault):
        import yaml

        ds = _TinyDS()
        bundle = useml.DataBundle("tiny", source=ds, transform=_scale)
        model = _TinyNet()

        useml.track("net", model)
        useml.commit(
            "manual with bundle",
            bundle_meta=bundle.to_meta_dict(),
        )

        snap = useml.focus("bundle-project")[0]
        meta = yaml.safe_load((snap.path / "metadata.yaml").read_text())
        assert "data" in meta
        assert meta["data"]["name"] == "tiny"
