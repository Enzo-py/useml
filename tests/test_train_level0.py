"""Level-0 training pipeline tests.

Convergence on the AND logic gate (trivial, linearly separable):
  AND(0,0)=0  AND(0,1)=0  AND(1,0)=0  AND(1,1)=1
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import useml
from useml.template.config import Config
from useml.template.dataset import load_dataset
from useml.template.trainer import Trainer, run_training


# ====================================================================
# Shared dataset & model
# ====================================================================

class ANDDataset(Dataset):
    """AND gate repeated 25× so train/val splits are stable."""

    def __init__(self, repeat: int = 25):
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([0, 0, 0, 1])
        self.X = X.repeat(repeat, 1)
        self.y = y.repeat(repeat)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ANDNet(useml.Model):
    """Two-input, two-class linear classifier."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


def _cfg(**kwargs) -> Config:
    defaults = dict(
        epochs=300,
        lr=0.1,
        batch_size=16,
        optimizer="adam",
        loss="cross_entropy",
        val_split=0.1,
        seed=42,
        num_workers=0,
        checkpoint_every=9999,  # disabled by default in unit tests
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _predict_and11(model: nn.Module) -> int:
    """Predicted class for input AND(1, 1), device-agnostic."""
    device = next(model.parameters()).device
    x = torch.tensor([[1.0, 1.0]], device=device)
    with torch.no_grad():
        logits = model(x)
        return int(logits.argmax(dim=1).item())


# ====================================================================
# Training loop (no vault)
# ====================================================================

class TestTrainingLoop:

    def test_loss_decreases(self):
        """Training loss must be strictly lower at the end than at the start."""
        config = _cfg(epochs=100)
        train_loader, val_loader = load_dataset(ANDDataset(), config)

        trainer = Trainer(ANDNet(), config)
        history = trainer.run(train_loader, val_loader)

        first, last = history["train_loss"][0], history["train_loss"][-1]
        assert first > last, f"Loss did not decrease: {first:.4f} → {last:.4f}"

    def test_and_convergence(self):
        """After 300 epochs the model must predict 1 for AND(1, 1)."""
        config = _cfg(epochs=300)
        train_loader, val_loader = load_dataset(ANDDataset(), config)

        trainer = Trainer(ANDNet(), config)
        trainer.run(train_loader, val_loader)

        pred = _predict_and11(trainer.model)
        assert pred == 1, f"AND(1,1): expected 1, got {pred}"

    def test_val_loss_tracked(self):
        """History must contain one val_loss entry per epoch, all positive."""
        config = _cfg(epochs=10)
        train_loader, val_loader = load_dataset(ANDDataset(), config)

        trainer = Trainer(ANDNet(), config)
        history = trainer.run(train_loader, val_loader)

        assert "val_loss" in history
        assert len(history["val_loss"]) == 10
        assert all(v > 0 for v in history["val_loss"])

    def test_history_length_matches_epochs(self):
        """train_loss and val_loss lengths must equal Config.epochs."""
        config = _cfg(epochs=17)
        train_loader, val_loader = load_dataset(ANDDataset(), config)

        trainer = Trainer(ANDNet(), config)
        history = trainer.run(train_loader, val_loader)

        assert len(history["train_loss"]) == 17
        assert len(history["val_loss"]) == 17


# ====================================================================
# Vault integration (save & load)
# ====================================================================

class TestVaultIntegration:

    def test_checkpoint_dir_created(self, tmp_path):
        """run_training must create at least one snapshot on disk."""
        config = _cfg(epochs=30, checkpoint_every=10)
        run_training(ANDNet, ANDDataset(), config=config,
                     vault_path=str(tmp_path / "vault"))

        project_dir = tmp_path / "vault" / "ANDNet"
        assert project_dir.exists(), "Project directory missing"
        snaps = [d for d in project_dir.iterdir() if d.name.startswith("snap_")]
        assert len(snaps) >= 1, f"Expected ≥1 snapshot, got {len(snaps)}"

    def test_weights_file_present(self, tmp_path):
        """Every snapshot must contain weights/model.pth."""
        config = _cfg(epochs=10, checkpoint_every=5)
        run_training(ANDNet, ANDDataset(), config=config,
                     vault_path=str(tmp_path / "vault"))

        project_dir = tmp_path / "vault" / "ANDNet"
        snaps = [d for d in project_dir.iterdir() if d.name.startswith("snap_")]
        assert snaps, "No snapshots found"
        for snap in snaps:
            assert (snap / "weights" / "model.pth").exists(), \
                f"weights/model.pth missing in {snap.name}"

    def test_load_latest_after_training(self, tmp_path):
        """useml.load() must return a model with correct weights after run_training."""
        config = _cfg(epochs=300, checkpoint_every=100)
        run_training(ANDNet, ANDDataset(), config=config,
                     vault_path=str(tmp_path / "vault"))

        # run_training already set up the session — load directly
        loaded = useml.load("model")
        assert isinstance(loaded, ANDNet)
        assert _predict_and11(loaded) == 1, \
            "Loaded model failed to predict AND(1,1)=1"

    def test_load_with_explicit_from(self, tmp_path):
        """useml.load(_from='\\latest') must return a usable ANDNet."""
        config = _cfg(epochs=100, checkpoint_every=50)
        run_training(ANDNet, ANDDataset(), config=config,
                     vault_path=str(tmp_path / "vault"))

        loaded = useml.load("model", _from="\\latest")
        assert isinstance(loaded, ANDNet)

    def test_multiple_checkpoints_ordered(self, tmp_path):
        """\\latest must be the most recent checkpoint (highest epoch)."""
        config = _cfg(epochs=30, checkpoint_every=10)
        run_training(ANDNet, ANDDataset(), config=config,
                     vault_path=str(tmp_path / "vault"))

        project_dir = tmp_path / "vault" / "ANDNet"
        snaps = sorted(
            [d for d in project_dir.iterdir() if d.name.startswith("snap_")],
            key=lambda d: d.name,
        )
        # \\latest resolves to snapshots[0] (newest-first order)
        loaded_latest = useml.load("model", _from="\\latest")
        loaded_head0 = useml.load("model", _from=snaps[-1].name)

        p1 = _predict_and11(loaded_latest)
        p2 = _predict_and11(loaded_head0)
        assert p1 == p2, \
            "\\latest and the newest snapshot dir should give the same prediction"


# ====================================================================
# Public API surface
# ====================================================================

class TestPublicAPI:

    def test_useml_train_returns_history(self, tmp_path):
        """useml.train() must return a dict with train_loss and val_loss."""
        config = _cfg(epochs=10)
        history = useml.train(ANDNet, ANDDataset(), config=config,
                              vault_path=str(tmp_path / "vault"))

        assert isinstance(history, dict)
        assert "train_loss" in history and "val_loss" in history
        assert len(history["train_loss"]) == 10

    def test_useml_model_subclassable(self):
        """useml.Model must be usable as an nn.Module base class."""
        net = ANDNet()
        assert isinstance(net, nn.Module)
        out = net(torch.zeros(2, 2))
        assert out.shape == (2, 2)

    def test_config_device_auto_resolved(self):
        """Config(device='auto') must resolve to a concrete device string."""
        cfg = Config(device="auto")
        assert cfg.device in ("cpu", "cuda", "mps")
