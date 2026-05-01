from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class Config:
    # --- Training ---
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"          # "adam" | "adamw" | "sgd"
    loss: Any = "cross_entropy"      # str | nn.Module subclass | nn.Module instance | callable

    # --- Hardware ---
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "mps"

    # --- Data ---
    val_split: float = 0.1
    num_workers: int = 0
    data_dir: str = ".useml_data"
    seed: int = 42

    # --- Vault checkpointing ---
    checkpoint_every: int = 5
    checkpoint_metric: str = "val_loss"

    def __post_init__(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    # ------------------------------------------------------------------
    # Loss introspection
    # ------------------------------------------------------------------

    def loss_name(self) -> str:
        """Human-readable name of the loss for manifest/metadata."""
        import types as _types
        if isinstance(self.loss, str):
            return self.loss
        if isinstance(self.loss, type):
            return self.loss.__name__
        if isinstance(self.loss, _types.FunctionType):
            return self.loss.__name__
        return type(self.loss).__name__

    def loss_object(self) -> Optional[Any]:
        """Returns the loss class or instance if custom (not a built-in string)."""
        if isinstance(self.loss, str):
            return None
        return self.loss

    def loss_hash(self) -> Optional[str]:
        """MD5 of the loss source code, or None for built-in string losses."""
        import types as _types
        obj = self.loss_object()
        if obj is None:
            return None
        if isinstance(obj, type):
            target = obj                 # class → inspect the class
        elif isinstance(obj, _types.FunctionType):
            target = obj                 # plain function → inspect directly
        else:
            target = type(obj)           # nn.Module instance → inspect its class
        try:
            src = inspect.getsource(target)
            return hashlib.md5(src.encode()).hexdigest()
        except (OSError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Returns a YAML-serializable dict (no class objects)."""
        return {
            "epochs":            self.epochs,
            "batch_size":        self.batch_size,
            "lr":                self.lr,
            "optimizer":         self.optimizer,
            "loss":              self.loss_name(),
            "device":            self.device,
            "val_split":         self.val_split,
            "num_workers":       self.num_workers,
            "data_dir":          self.data_dir,
            "seed":              self.seed,
            "checkpoint_every":  self.checkpoint_every,
            "checkpoint_metric": self.checkpoint_metric,
        }
