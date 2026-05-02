import inspect
import time
from typing import Any, Callable, Optional, Type

import torch
import torch.nn as nn

from .config import Config
from ..dataset import load_dataset
from ..errors import UnknownLossError, InvalidLossTypeError, UnknownOptimizerError


_BUILTIN_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse":           nn.MSELoss,
    "bce":           nn.BCEWithLogitsLoss,
    "l1":            nn.L1Loss,
}

_BUILTIN_OPTIMIZERS = {
    "adam":  lambda p, lr: torch.optim.Adam(p, lr=lr),
    "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr),
    "sgd":   lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
}


def _build_loss(config: Config) -> nn.Module:
    loss = config.loss

    if isinstance(loss, str):
        key = loss.lower()
        if key not in _BUILTIN_LOSSES:
            raise UnknownLossError(
                f"Unknown loss '{loss}'. Built-in choices: {list(_BUILTIN_LOSSES)}."
            )
        return _BUILTIN_LOSSES[key]()

    if isinstance(loss, nn.Module):
        return loss

    if isinstance(loss, type) and issubclass(loss, nn.Module):
        return loss()

    if callable(loss):
        _fn = loss
        class _WrappedLoss(nn.Module):
            def forward(self, pred, target):
                return _fn(pred, target)
        return _WrappedLoss()

    raise InvalidLossTypeError(
        f"config.loss must be a str, nn.Module subclass, nn.Module instance, "
        f"or callable. Got {type(loss).__name__}."
    )


def _build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    key = config.optimizer.lower()
    if key not in _BUILTIN_OPTIMIZERS:
        raise UnknownOptimizerError(
            f"Unknown optimizer '{config.optimizer}'. Choices: {list(_BUILTIN_OPTIMIZERS)}."
        )
    return _BUILTIN_OPTIMIZERS[key](model.parameters(), config.lr)


def _build_model(model_cls: Type[nn.Module], config: Config) -> nn.Module:
    sig = inspect.signature(model_cls.__init__)
    for param in list(sig.parameters.keys())[1:]:
        if param in ("config", "cfg"):
            return model_cls(**{param: config})
    return model_cls()


def _batch_size(batch: Any) -> int:
    first = batch[0] if isinstance(batch, (list, tuple)) else batch
    return first.size(0)


class Trainer:
    """Core training loop with vault checkpoint integration.

    Override :meth:`step` to customise the forward + loss computation for
    any training paradigm (autoencoder, self-supervised, distillation, …).

    Example — autoencoder::

        class AETrainer(useml.Trainer):
            def step(self, batch):
                x, _ = batch
                x = x.to(self.config.device)
                return self.criterion(self.model(x), x)   # target = input

    Or pass ``step_fn`` to :func:`run_training` / :func:`useml.train` for
    inline use without subclassing::

        def ae_step(model, batch, device):
            x, _ = batch
            x = x.to(device)
            return torch.nn.functional.mse_loss(model(x), x)

        useml.train(AutoEncoder, "mnist", config=config, step_fn=ae_step)
    """

    def __init__(self, model: nn.Module, config: Config) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = _build_optimizer(model, config)
        self._criterion: Optional[nn.Module] = None   # built lazily on first use
        self._session = self._get_active_session()
        self.bundle_meta: Optional[dict] = None              # set by run_training when a DataBundle is used
        self._bundle_transform_source: Optional[str] = None  # transform source code for archival
        self._bundle_transform_key: Optional[str] = None     # relative path inside source/

    @property
    def criterion(self) -> nn.Module:
        """Loss module — built on first access so step_fn users never trigger _build_loss."""
        if self._criterion is None:
            self._criterion = _build_loss(self.config)
        return self._criterion

    @criterion.setter
    def criterion(self, value: nn.Module) -> None:
        self._criterion = value

    def _get_active_session(self):
        try:
            import useml as _useml
            _ = _useml._session.project
            return _useml._session
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def step(self, batch: Any) -> torch.Tensor:
        """Compute the loss for one batch.

        Override this method to implement custom training logic.
        The model is already in the correct mode (train/eval) when this
        is called; gradients and optimizer steps are handled by the loop.

        Args:
            batch: Whatever the DataLoader yields — typically ``(x, y)``
                but can be any structure.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)
        return self.criterion(self.model(x), y)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self, train_loader, val_loader) -> dict:
        """Runs the full training loop.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.

        Returns:
            History dict with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        best_val = float("inf")
        history = {"train_loss": [], "val_loss": []}

        print(f"\n{'=' * 55}")
        print(
            f"  useml training — {self.config.epochs} epochs "
            f"on {self.config.device}  |  loss: {self.config.loss_name()}"
        )
        print(f"{'=' * 55}")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss   = self._run_epoch(val_loader,   train=False)
            elapsed    = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            best_val = min(best_val, val_loss)

            self._print_epoch(epoch, train_loss, val_loss, elapsed)
            self._maybe_checkpoint(epoch, val_loss)

        print(f"{'=' * 55}")
        print(f"  Best val_loss: {best_val:.4f}")
        print(f"{'=' * 55}\n")

        return history

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()
                loss = self.step(batch)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * _batch_size(batch)
                count += _batch_size(batch)
        return total / count if count else 0.0

    def _print_epoch(self, epoch: int, train: float, val: float, elapsed: float) -> None:
        w = len(str(self.config.epochs))
        print(
            f"  epoch {epoch:{w}d}/{self.config.epochs} │ "
            f"train {train:.4f} │ val {val:.4f} │ {elapsed:.1f}s"
        )

    def _maybe_checkpoint(self, epoch: int, val_loss: float) -> None:
        if not self._session:
            return
        if epoch % self.config.checkpoint_every != 0:
            return
        self._session.track("model", self.model, config=self.config)
        inline = (
            {self._bundle_transform_key: self._bundle_transform_source}
            if self._bundle_transform_source and self._bundle_transform_key
            else None
        )
        self._session.commit(
            message=f"epoch {epoch}",
            bundle_meta=self.bundle_meta,
            bundle_inline_source=inline,
            epoch=epoch,
            val_loss=round(val_loss, 6),
        )


def run_training(
    model_cls: Type[nn.Module],
    dataset: Any,
    config: Optional[Config] = None,
    step_fn: Optional[Callable] = None,
) -> dict:
    """Level-0 entry point — train a model in two lines.

    Args:
        model_cls: Model class (subclass of ``nn.Module``).
        dataset: Built-in name (``"mnist"``, ``"cifar10"``, …),
            ``"hf:<name>"``, or a ``torch.utils.data.Dataset``.
        config: Training configuration. Defaults to :class:`Config`.
        step_fn: Optional callable ``(model, batch, device) -> loss_tensor``
            that replaces the default ``criterion(model(x), y)`` computation.
            Use this to implement autoencoders, self-supervised losses, etc.
            without subclassing :class:`Trainer`.

    Returns:
        History dict with keys ``"train_loss"`` and ``"val_loss"``.

    Example::

        def ae_step(model, batch, device):
            x, _ = batch
            x = x.to(device)
            return F.mse_loss(model(x), x)

        useml.train(AutoEncoder, "mnist", config=config, step_fn=ae_step)
    """
    if config is None:
        config = Config()

    from ..dataset import DataBundle
    if isinstance(dataset, DataBundle):
        trainer_bundle_meta = dataset.to_meta_dict()
        trainer_transform_source = dataset.transform_source()
        trainer_transform_key = dataset.inline_source_key() if trainer_transform_source else None
    else:
        trainer_bundle_meta = None
        trainer_transform_source = None
        trainer_transform_key = None

    train_loader, val_loader = load_dataset(dataset, config)
    model = _build_model(model_cls, config)
    trainer = Trainer(model=model, config=config)
    trainer.bundle_meta = trainer_bundle_meta
    trainer._bundle_transform_source = trainer_transform_source
    trainer._bundle_transform_key = trainer_transform_key

    if step_fn is not None:
        _device = config.device
        _fn = step_fn
        trainer.step = lambda batch: _fn(trainer.model, batch, _device)

    return trainer.run(train_loader, val_loader)
