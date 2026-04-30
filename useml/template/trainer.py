import inspect
import time
from pathlib import Path
from typing import Any, Optional, Type, Union

import torch
import torch.nn as nn

from .config import Config
from .dataset import load_dataset


def _build_optimizer(model: nn.Module, config: Config):
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    raise ValueError(
        f"Unknown optimizer '{config.optimizer}'. Choose: adam, adamw, sgd"
    )


def _build_loss(config: Config):
    name = config.loss.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(
        f"Unknown loss '{config.loss}'. Choose: cross_entropy, mse, bce, l1"
    )


def _build_model(model_cls: Type[nn.Module], config: Config) -> nn.Module:
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.keys())  # includes 'self'
    for param in params[1:]:  # skip 'self'
        if param in ("config", "cfg"):
            return model_cls(**{param: config})
    return model_cls()


class Trainer:
    """Minimal training loop with vault checkpoint integration."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        vault_path: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = _build_optimizer(model, config)
        self.criterion = _build_loss(config)

        self._vault_path = vault_path
        self._project_name = project_name
        self._session = None

        if vault_path:
            self._init_vault(vault_path, project_name)

    def _init_vault(self, vault_path: str, project_name: str) -> None:
        try:
            import useml as _useml
            _useml.init(vault_path)
            _useml.new(project_name, auto_focus=True)
            self._session = _useml._session
        except Exception:
            self._session = None

    def run(self, train_loader, val_loader) -> dict:
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        print(f"\n{'='*55}")
        print(f"  useml training — {self.config.epochs} epochs on {self.config.device}")
        print(f"{'='*55}")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            self._print_epoch(epoch, train_loss, val_loss, elapsed)
            self._maybe_checkpoint(epoch, val_loss)

        print(f"{'='*55}")
        print(f"  Best val_loss: {best_val_loss:.4f}")
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

    def _print_epoch(self, epoch: int, train_loss: float, val_loss: float, elapsed: float) -> None:
        w = len(str(self.config.epochs))
        print(
            f"  epoch {epoch:{w}d}/{self.config.epochs} │ "
            f"train {train_loss:.4f} │ "
            f"val {val_loss:.4f} │ "
            f"{elapsed:.1f}s"
        )

    def _maybe_checkpoint(self, epoch: int, val_loss: float) -> None:
        if not self._session:
            return
        if epoch % self.config.checkpoint_every != 0:
            return

        try:
            self._session.track("model", self.model)
            self._session.commit(
                message=f"epoch {epoch}",
                epoch=epoch,
                val_loss=round(val_loss, 6),
            )
        except Exception:
            pass


def run_training(
    model_cls: Type[nn.Module],
    dataset: Any,
    config: Optional[Config] = None,
    vault_path: str = ".useml_vault",
) -> dict:
    """Entry point for Level-0 training.

    Parameters
    ----------
    model_cls : class
        Subclass of useml.Model (or any nn.Module).
    dataset : str or torch.utils.data.Dataset
        Built-in name ("mnist", "cifar10", …), "hf:<name>", or a Dataset.
    config : Config, optional
        Training configuration. Defaults to Config().
    vault_path : str
        Where to persist snapshots.

    Returns
    -------
    dict
        Training history {"train_loss": [...], "val_loss": [...]}.
    """
    if config is None:
        config = Config()

    train_loader, val_loader = load_dataset(dataset, config)

    model = _build_model(model_cls, config)

    project_name = model_cls.__name__
    trainer = Trainer(
        model=model,
        config=config,
        vault_path=vault_path,
        project_name=project_name,
    )

    return trainer.run(train_loader, val_loader)
