import hashlib
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, Optional


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _fingerprint_dataset(dataset) -> str:
    """Lightweight fingerprint: length + a few sampled items (no full scan)."""
    import torch

    n = len(dataset)
    parts = [str(n)]
    probe_indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    for i in probe_indices:
        if 0 <= i < n:
            try:
                sample = dataset[i]
                item = sample[0] if isinstance(sample, (tuple, list)) else sample
                if hasattr(item, "numpy"):
                    raw = item.numpy().tobytes()
                else:
                    raw = str(item).encode()
                parts.append(hashlib.sha256(raw).hexdigest()[:8])
            except Exception:
                parts.append(f"err_{i}")
    return _sha256("|".join(parts))


class DataBundle:
    """Versioned data contract — pairs a dataset with its preprocessing transform.

    Records three hashes at every snapshot:
      - ``source_hash``: fingerprint of the raw data
      - ``transform_hash``: sha256 of the transform function's source code
      - ``cache_key``: sha256(source_hash + transform_hash), used for disk cache

    All metadata is embedded under ``data:`` in ``metadata.yaml`` for every
    snapshot that uses this bundle. The transform source code is archived
    verbatim in ``source/_bundle_<name>_<fn>.py``.

    Args:
        name: Human-readable identifier stored in every snapshot.
        source: Built-in name (``"mnist"``, ``"hf:..."``) or a
            ``torch.utils.data.Dataset`` instance.
        transform: Per-sample callable ``(sample) -> sample`` applied after
            the base dataset. ``None`` means identity (no preprocessing).
        version: Optional label (``"v1"``, ``"augmented"``). Informational only.
        cache: If ``True``, the fully-processed dataset is serialised to
            ``<data_dir>/.bundle_cache/<cache_key>.pt`` on first run and
            reloaded on subsequent runs. Use only when preprocessing is very
            slow — the cache can be large.
    """

    def __init__(
        self,
        name: str,
        source: Any,
        transform: Optional[Callable] = None,
        version: Optional[str] = None,
        cache: bool = False,
    ) -> None:
        self.name = name
        self.source = source
        self.transform = transform
        self.version = version
        self.cache = cache

        self._source_hash_cache: Optional[str] = None
        self._transform_hash_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Hashes
    # ------------------------------------------------------------------

    def source_hash(self) -> str:
        if self._source_hash_cache is None:
            if isinstance(self.source, str):
                self._source_hash_cache = _sha256(f"builtin:{self.source.lower()}")
            else:
                self._source_hash_cache = _fingerprint_dataset(self.source)
        return self._source_hash_cache

    def transform_hash(self) -> str:
        if self._transform_hash_cache is None:
            if self.transform is None:
                self._transform_hash_cache = "none"
            else:
                try:
                    src = inspect.getsource(self.transform)
                except (OSError, TypeError):
                    src = repr(self.transform)
                # Include __name__ so factory-produced closures with the same
                # body but different captured params hash differently when the
                # caller sets a descriptive __name__ (e.g. "gridify_16r_32t").
                name = getattr(self.transform, "__name__", "")
                self._transform_hash_cache = _sha256(src + name)
        return self._transform_hash_cache

    def cache_key(self) -> str:
        return _sha256(self.source_hash() + self.transform_hash())[:16]

    # ------------------------------------------------------------------
    # Transform source archival
    # ------------------------------------------------------------------

    def transform_source(self) -> Optional[str]:
        """Full source code of the transform for snapshot archival."""
        if self.transform is None:
            return None
        try:
            return inspect.getsource(self.transform)
        except (OSError, TypeError):
            return repr(self.transform)

    def inline_source_key(self) -> str:
        """Relative path used when writing transform source into snapshot/source/."""
        fn_name = getattr(self.transform, "__name__", "transform")
        return f"_bundle_{self.name}_{fn_name}.py"

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def loaders(self, config):
        """Return ``(train_loader, val_loader)`` with the transform applied."""
        from .loaders import _wrap_custom

        if self.cache:
            return self._cached_loaders(config)

        base_ds = self._get_base_dataset(config)
        if self.transform is not None:
            base_ds = _TransformWrapper(base_ds, self.transform)
        return _wrap_custom(base_ds, config)

    def _get_base_dataset(self, config):
        from ..errors import UnknownDatasetError
        from .loaders import _BUILTIN

        if isinstance(self.source, str):
            name = self.source.lower()
            if name in _BUILTIN:
                return _load_torchvision_raw(name, config)
            if name.startswith("hf:"):
                return _load_hf_raw(name[3:])
            raise UnknownDatasetError(
                f"Unknown dataset '{self.source}'. "
                f"Built-in: {list(_BUILTIN.keys())}. HuggingFace: 'hf:<name>'."
            )
        return self.source

    def _cached_loaders(self, config):
        import torch
        from .loaders import _wrap_custom

        cache_path = (
            Path(config.data_dir) / ".bundle_cache" / f"{self.cache_key()}.pt"
        )

        if cache_path.exists():
            processed = torch.load(cache_path, weights_only=False)
        else:
            warnings.warn(
                f"[DataBundle '{self.name}'] Building cache — this may take a while. "
                f"Cache will be saved to {cache_path}",
                stacklevel=2,
            )
            base_ds = self._get_base_dataset(config)
            if self.transform is not None:
                base_ds = _TransformWrapper(base_ds, self.transform)
            processed = _MaterializedDataset(base_ds)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(processed, cache_path)

        return _wrap_custom(processed, config)

    # ------------------------------------------------------------------
    # Snapshot metadata
    # ------------------------------------------------------------------

    def to_meta_dict(self) -> dict:
        """Dict embedded under ``data:`` in snapshot ``metadata.yaml``."""
        d = {
            "name": self.name,
            "source": (
                self.source if isinstance(self.source, str)
                else type(self.source).__name__
            ),
            "source_hash": self.source_hash(),
            "transform": (
                getattr(self.transform, "__name__", repr(self.transform))
                if self.transform else None
            ),
            "transform_hash": self.transform_hash(),
            "cache_key": self.cache_key(),
        }
        if self.version:
            d["version"] = self.version
        return d

    def __repr__(self) -> str:
        src = self.source if isinstance(self.source, str) else type(self.source).__name__
        fn = getattr(self.transform, "__name__", None) if self.transform else None
        return f"<DataBundle name='{self.name}' source={src!r} transform={fn!r}>"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

class _TransformWrapper:
    """Applies a per-sample transform lazily (no materialisation)."""

    def __init__(self, dataset, transform: Callable) -> None:
        self._ds = dataset
        self._fn = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        return self._fn(self._ds[idx])


class _MaterializedDataset:
    """Fully processed in-memory dataset (used only when cache=True)."""

    def __init__(self, dataset) -> None:
        self._samples = [dataset[i] for i in range(len(dataset))]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def _load_torchvision_raw(name: str, config):
    """Load a built-in torchvision dataset as a Dataset (train split only)."""
    try:
        import torchvision
        from torchvision import transforms
    except ImportError:
        from ..errors import TorchvisionNotInstalledError
        raise TorchvisionNotInstalledError()

    from .loaders import _BUILTIN
    info = _BUILTIN[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])
    DatasetCls = getattr(torchvision.datasets, info["class"])
    return DatasetCls(
        root=config.data_dir, train=True, download=True, transform=transform
    )


def _load_hf_raw(hf_name: str):
    """Load a HuggingFace dataset as a Dataset (train split only)."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        from ..errors import HuggingFaceNotInstalledError
        raise HuggingFaceNotInstalledError()

    import torch
    raw = hf_load(hf_name)
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    data = raw[split_name]

    class _HFRaw:
        def __len__(self):
            return len(data)

        def __getitem__(self, idx):
            row = data[idx]
            x_key = "pixel_values" if "pixel_values" in row else "image"
            x = torch.tensor(row[x_key], dtype=torch.float32)
            y = torch.tensor(row["label"], dtype=torch.long)
            return x, y

    return _HFRaw()
