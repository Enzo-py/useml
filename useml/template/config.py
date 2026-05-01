from __future__ import annotations

import hashlib
import inspect
import types as _types
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class Config:
    """Training configuration for a single model component.

    All fields have sensible defaults so ``Config()`` is valid out of the box.
    The ``loss`` field accepts multiple forms — see :attr:`loss` for details.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Samples per mini-batch.
        lr: Initial learning rate.
        optimizer: Optimiser name — ``"adam"``, ``"adamw"``, or ``"sgd"``.
        loss: Loss specification. Accepts:
            - A string key (``"cross_entropy"``, ``"mse"``, ``"bce"``, ``"l1"``).
            - An ``nn.Module`` subclass (instantiated automatically).
            - An ``nn.Module`` instance (used as-is).
            - A plain callable ``fn(pred, target) -> Tensor``.
        device: Target device — ``"cpu"``, ``"cuda"``, ``"mps"``, or
            ``"auto"`` (resolved at init time).
        val_split: Fraction of training data reserved for validation.
        num_workers: DataLoader worker processes.
        data_dir: Root directory for built-in dataset downloads.
        seed: Random seed for reproducible splits.
        checkpoint_every: Save a vault snapshot every N epochs (0 = disabled).
        checkpoint_metric: Metric name used to select the best checkpoint.
    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    loss: Any = "cross_entropy"
    device: str = "auto"
    val_split: float = 0.1
    num_workers: int = 0
    data_dir: str = ".useml_data"
    seed: int = 42
    checkpoint_every: int = 5
    checkpoint_metric: str = "val_loss"

    def __post_init__(self) -> None:
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
        """Returns a human-readable name for the configured loss.

        Returns:
            The string key for built-in losses, the class name for
            ``nn.Module`` subclasses or instances, or the function name for
            plain callables.
        """
        if isinstance(self.loss, str):
            return self.loss
        if isinstance(self.loss, type):
            return self.loss.__name__
        if isinstance(self.loss, _types.FunctionType):
            return self.loss.__name__
        return type(self.loss).__name__

    def loss_object(self) -> Optional[Any]:
        """Returns the loss object when it is a custom class or callable.

        Returns:
            The loss class, instance, or function; ``None`` for built-in
            string losses.
        """
        if isinstance(self.loss, str):
            return None
        return self.loss

    def loss_hash(self) -> Optional[str]:
        """Computes an MD5 fingerprint of the loss source code.

        Returns:
            Hexadecimal MD5 string, or ``None`` for built-in string losses
            or when the source is unavailable.
        """
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
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Returns a YAML-serialisable representation of this config.

        The ``loss`` field is replaced by its string name so that class
        objects and callables do not appear in the output.

        Returns:
            Plain dict with only JSON/YAML-safe values.
        """
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "loss": self.loss_name(),
            "device": self.device,
            "val_split": self.val_split,
            "num_workers": self.num_workers,
            "data_dir": self.data_dir,
            "seed": self.seed,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_metric": self.checkpoint_metric,
        }
