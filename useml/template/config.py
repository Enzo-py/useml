from __future__ import annotations

import hashlib
import inspect
from typing import Any, Optional

import torch


def _yaml_safe(value: Any) -> Any:
    """Best-effort conversion to a YAML-serialisable scalar."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


class Config:
    """Training configuration.

    Standard fields have fixed defaults.  Any extra keyword argument is stored
    as a custom field and serialised automatically to the snapshot YAML.

    Example::

        config = Config(epochs=10, lr=1e-3, latent_dim=16, kl_weight=1e-3)
        config.latent_dim   # 16
        config.to_dict()    # includes latent_dim and kl_weight
    """

    _STANDARD_FIELDS: dict = {
        "epochs":            20,
        "batch_size":        64,
        "lr":                1e-3,
        "optimizer":         "adam",
        "loss":              "cross_entropy",
        "device":            "auto",
        "val_split":         0.1,
        "num_workers":       0,
        "data_dir":          ".useml_data",
        "seed":              42,
        "checkpoint_every":  5,
        "checkpoint_metric": "val_loss",
    }

    def __init__(self, **kwargs: Any) -> None:
        for key, default in self._STANDARD_FIELDS.items():
            setattr(self, key, kwargs.pop(key, default))
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._custom_keys: list = list(kwargs.keys())

        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    # ------------------------------------------------------------------
    # Custom-field helpers
    # ------------------------------------------------------------------

    def extra(self) -> dict:
        """Returns custom (non-standard) fields as a dict."""
        return {k: getattr(self, k) for k in self._custom_keys}

    # ------------------------------------------------------------------
    # Loss introspection
    # ------------------------------------------------------------------

    def loss_name(self) -> str:
        import types as _types
        if isinstance(self.loss, str):
            return self.loss
        if isinstance(self.loss, type):
            return self.loss.__name__
        if isinstance(self.loss, _types.FunctionType):
            return self.loss.__name__
        return type(self.loss).__name__

    def loss_object(self) -> Optional[Any]:
        if isinstance(self.loss, str):
            return None
        return self.loss

    def loss_hash(self) -> Optional[str]:
        import types as _types
        obj = self.loss_object()
        if obj is None:
            return None
        if isinstance(obj, type):
            target = obj
        elif isinstance(obj, _types.FunctionType):
            target = obj
        else:
            target = type(obj)
        try:
            src = inspect.getsource(target)
            return hashlib.md5(src.encode()).hexdigest()
        except (OSError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Returns a YAML-serialisable dict of all fields (standard + custom)."""
        d: dict = {}
        for key in self._STANDARD_FIELDS:
            val = getattr(self, key)
            d[key] = self.loss_name() if key == "loss" else val
        for key in self._custom_keys:
            d[key] = _yaml_safe(getattr(self, key))
        return d

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{k}={getattr(self, k)!r}" for k in self._STANDARD_FIELDS]
        parts += [f"{k}={getattr(self, k)!r}" for k in self._custom_keys]
        return f"Config({', '.join(parts)})"
