import torch
from torch.utils.data import DataLoader, random_split

from ..errors import (
    TorchvisionNotInstalledError,
    HuggingFaceNotInstalledError,
    UnknownDatasetError,
    InvalidDatasetTypeError,
)


# Built-in dataset registry (torchvision)
_BUILTIN: dict = {
    "mnist": {
        "class": "MNIST",
        "mean": (0.1307,),
        "std":  (0.3081,),
    },
    "fashion_mnist": {
        "class": "FashionMNIST",
        "mean": (0.2860,),
        "std":  (0.3530,),
    },
    "cifar10": {
        "class": "CIFAR10",
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "class": "CIFAR100",
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
    },
}


def _require_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError:
        raise TorchvisionNotInstalledError(
            "torchvision is required for built-in datasets. "
            "Install it with: pip install torchvision"
        )


def load_dataset(dataset, config):
    """Return (train_loader, val_loader) from a name string or torch Dataset.

    Supported strings
    -----------------
    "mnist", "fashion_mnist", "cifar10", "cifar100"
    "hf:<hf_dataset_name>"   (requires: pip install datasets)

    Custom dataset
    --------------
    Pass any torch.utils.data.Dataset — it will be split automatically.
    """
    if isinstance(dataset, str):
        name = dataset.lower()
        if name in _BUILTIN:
            return _load_torchvision(name, config)
        if name.startswith("hf:"):
            return _load_huggingface(name[3:], config)
        raise UnknownDatasetError(
            f"Unknown dataset '{dataset}'. "
            f"Built-in: {list(_BUILTIN.keys())}. "
            f"HuggingFace: 'hf:<name>'."
        )
    elif hasattr(dataset, "__len__"):
        return _wrap_custom(dataset, config)
    else:
        raise InvalidDatasetTypeError(
            f"dataset must be a string or a torch Dataset, got {type(dataset).__name__}."
        )


def _load_torchvision(name: str, config):
    tv = _require_torchvision()
    from torchvision import transforms

    info = _BUILTIN[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])

    DatasetCls = getattr(tv.datasets, info["class"])

    train_full = DatasetCls(
        root=config.data_dir, train=True,  download=True, transform=transform
    )
    test_ds = DatasetCls(
        root=config.data_dir, train=False, download=True, transform=transform
    )

    train_ds, val_ds = _split(train_full, config.val_split, config.seed)

    train_loader = _make_loader(train_ds, config, shuffle=True)
    val_loader   = _make_loader(val_ds,   config, shuffle=False)

    return train_loader, val_loader


def _load_huggingface(hf_name: str, config):
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise HuggingFaceNotInstalledError(
            "The 'datasets' package is required for HuggingFace datasets. "
            "Install it with: pip install datasets"
        )

    raw = hf_load(hf_name)

    # Minimal wrapper: assumes 'image'/'pixel_values' + 'label' columns
    class HFWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self._data = hf_split

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            row = self._data[idx]
            x_key = "pixel_values" if "pixel_values" in row else "image"
            x = torch.tensor(row[x_key], dtype=torch.float32)
            y = torch.tensor(row["label"], dtype=torch.long)
            return x, y

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    train_full = HFWrapper(raw[split_name])
    train_ds, val_ds = _split(train_full, config.val_split, config.seed)

    return _make_loader(train_ds, config, shuffle=True), \
           _make_loader(val_ds,   config, shuffle=False)


def _wrap_custom(dataset, config):
    train_ds, val_ds = _split(dataset, config.val_split, config.seed)
    return _make_loader(train_ds, config, shuffle=True), \
           _make_loader(val_ds,   config, shuffle=False)


def _split(dataset, val_frac: float, seed: int):
    val_size   = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=g)


def _make_loader(ds, config, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=(config.device != "cpu"),
    )
