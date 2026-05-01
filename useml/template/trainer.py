import inspect
import time
from typing import Any, Optional, Type

import torch
import torch.nn as nn

from .config import Config
from .dataset import load_dataset


# ------------------------------------------------------------------ #
#  Internal builders                                                   #
# ------------------------------------------------------------------ #

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
    """Instantiate a loss from config.loss (str | class | instance | callable)."""
    loss = config.loss

    if isinstance(loss, str):
        key = loss.lower()
        if key not in _BUILTIN_LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. Built-in choices: "
                f"{list(_BUILTIN_LOSSES)}"
            )
        return _BUILTIN_LOSSES[key]()

    if isinstance(loss, nn.Module):
        return loss                              # already instantiated

    if isinstance(loss, type) and issubclass(loss, nn.Module):
        return loss()                            # instantiate the class

    if callable(loss):
        # Wrap a plain function so the training loop sees an nn.Module
        _fn = loss
        class _WrappedLoss(nn.Module):
            def forward(self, pred, target):
                return _fn(pred, target)
        return _WrappedLoss()

    raise TypeError(
        f"config.loss must be a str, nn.Module subclass, nn.Module instance, "
        f"or callable. Got {type(loss).__name__}."
    )


def _build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    key = config.optimizer.lower()
    if key not in _BUILTIN_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{config.optimizer}'. Choices: "
            f"{list(_BUILTIN_OPTIMIZERS)}"
        )
    return _BUILTIN_OPTIMIZERS[key](model.parameters(), config.lr)


def _build_model(model_cls: Type[nn.Module], config: Config) -> nn.Module:
    """Instantiate model, passing config if the constructor accepts it."""
    sig = inspect.signature(model_cls.__init__)
    for param in list(sig.parameters.keys())[1:]:   # skip 'self'
        if param in ("config", "cfg"):
            return model_cls(**{param: config})
    return model_cls()


# ------------------------------------------------------------------ #
#  Trainer                                                             #
# ------------------------------------------------------------------ #

class Trainer:
    """Core training loop with vault checkpoint integration.

    Checkpoints are saved to the currently focused project in the active
    useml session (useml.init + useml.new/focus must be called beforehand).
    If no session is active or no project is focused, training runs without
    saving snapshots.
    """

    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = _build_optimizer(model, config)
        self.criterion = _build_loss(config)
        self._session = self._get_active_session()

    def _get_active_session(self):
        try:
            import useml as _useml
            session = _useml._session
            # Only use the session if a vault is connected AND a project is focused
            _ = session.project  # raises RuntimeError if no focus
            return session
        except Exception:
            return None

    # ---------------------------------------------------------------- #

    def run(self, train_loader, val_loader) -> dict:
        best_val = float("inf")
        history = {"train_loss": [], "val_loss": []}

        print(f"\n{'='*55}")
        print(f"  useml training — {self.config.epochs} epochs "
              f"on {self.config.device}  |  loss: {self.config.loss_name()}")
        print(f"{'='*55}")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss   = self._val_epoch(val_loader)
            elapsed    = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss

            self._print_epoch(epoch, train_loss, val_loss, elapsed)
            self._maybe_checkpoint(epoch, val_loss)

        print(f"{'='*55}")
        print(f"  Best val_loss: {best_val:.4f}")
        print(f"{'='*55}\n")

        return history

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total, count = 0.0, 0
        device = self.config.device
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / count if count else 0.0

    def _val_epoch(self, loader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        device = self.config.device
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = self.criterion(self.model(x), y)
                total += loss.item() * x.size(0)
                count += x.size(0)
        return total / count if count else 0.0

    def _print_epoch(self, epoch: int, train: float, val: float, t: float):
        w = len(str(self.config.epochs))
        print(
            f"  epoch {epoch:{w}d}/{self.config.epochs} │ "
            f"train {train:.4f} │ val {val:.4f} │ {t:.1f}s"
        )

    def _maybe_checkpoint(self, epoch: int, val_loss: float) -> None:
        if not self._session:
            return
        if epoch % self.config.checkpoint_every != 0:
            return
        try:
            # Pass the Config instance so the vault can extract loss source
            self._session.track("model", self.model, config=self.config)
            self._session.commit(
                message=f"epoch {epoch}",
                epoch=epoch,
                val_loss=round(val_loss, 6),
            )
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def run_training(
    model_cls: Type[nn.Module],
    dataset: Any,
    config: Optional[Config] = None,
) -> dict:
    """Level-0 entry point — train a model in two lines.

    Snapshots are saved to the currently focused project in the active
    useml session. Call useml.init() + useml.new()/focus() beforehand to
    enable checkpointing; omitting them trains without saving.

    Parameters
    ----------
    model_cls : class
        Subclass of useml.Model (or any nn.Module).
    dataset : str or torch.utils.data.Dataset
        Built-in name ("mnist", "cifar10", …), "hf:<name>", or a Dataset.
    config : Config, optional
        Training configuration (loss, optimizer, epochs, …).

    Returns
    -------
    dict  {"train_loss": [...], "val_loss": [...]}
    """
    if config is None:
        config = Config()

    train_loader, val_loader = load_dataset(dataset, config)
    model = _build_model(model_cls, config)

    trainer = Trainer(model=model, config=config)
    return trainer.run(train_loader, val_loader)
