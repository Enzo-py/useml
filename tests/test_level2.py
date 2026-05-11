"""Tests for Level-2 features: scheduler, metrics, checkpoint strategies, early stop."""
import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import useml
from useml.template.trainer import Trainer, run_training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    useml.init(str(vault))
    useml.new("level2-project")
    yield tmp_path
    useml._session.vault = None
    useml._session._project = None
    useml._session.components = {}
    useml._session._is_dirty = False


class _DS(torch.utils.data.Dataset):
    def __init__(self, n=40):
        torch.manual_seed(0)
        self.x = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x): return self.fc(x)


def _accuracy(model, batch, device):
    x, y = batch
    return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()


def _cfg(**kw):
    defaults = dict(
        epochs=2, batch_size=8, loss="cross_entropy",
        device="cpu", checkpoint_every=9999,
    )
    defaults.update(kw)
    return useml.Config(**defaults)


# ---------------------------------------------------------------------------
# 1. Custom metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metric_appears_in_history(self, tmp_vault):
        history = run_training(
            _Net, _DS(), config=_cfg(epochs=2),
            metrics={"accuracy": _accuracy},
        )
        assert "train_accuracy" in history
        assert "val_accuracy" in history
        assert len(history["train_accuracy"]) == 2
        assert len(history["val_accuracy"]) == 2

    def test_metric_value_is_in_zero_one(self, tmp_vault):
        history = run_training(
            _Net, _DS(), config=_cfg(epochs=1),
            metrics={"accuracy": _accuracy},
        )
        assert 0.0 <= history["val_accuracy"][0] <= 1.0

    def test_no_metrics_means_only_loss(self, tmp_vault):
        history = run_training(_Net, _DS(), config=_cfg(epochs=1))
        assert set(history.keys()) == {"train_loss", "val_loss"}

    def test_multiple_metrics(self, tmp_vault):
        def ones(model, batch, device):
            return torch.tensor(1.0)

        history = run_training(
            _Net, _DS(), config=_cfg(epochs=1),
            metrics={"accuracy": _accuracy, "ones": ones},
        )
        for key in ("train_accuracy", "val_accuracy", "train_ones", "val_ones"):
            assert key in history
        # ones should be exactly 1.0 (allowing floating point)
        assert abs(history["val_ones"][0] - 1.0) < 1e-6

    def test_metric_passed_to_commit_metadata(self, tmp_vault):
        import yaml

        run_training(
            _Net, _DS(), config=_cfg(epochs=1, checkpoint_every=1),
            metrics={"accuracy": _accuracy},
        )
        snap = useml.focus("level2-project")[0]
        meta = yaml.safe_load((snap.path / "metadata.yaml").read_text())
        assert "metrics" in meta
        assert "val_accuracy" in meta["metrics"]


# ---------------------------------------------------------------------------
# 2. Scheduler attribute
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_scheduler_steps_per_epoch_by_default(self, tmp_vault):
        config = _cfg(epochs=3, lr=1.0)
        train_loader, val_loader = _make_loaders(_DS(), config)
        model = _Net()
        trainer = Trainer(model, config)
        trainer.scheduler = StepLR(trainer.optimizer, step_size=1, gamma=0.5)
        trainer.run(train_loader, val_loader)
        # After 3 epochs of decay 0.5 → lr = 1.0 * 0.5^3 = 0.125
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(final_lr - 0.125) < 1e-6

    def test_reduce_lr_on_plateau_receives_metric(self, tmp_vault):
        config = _cfg(epochs=2, lr=1.0)
        train_loader, val_loader = _make_loaders(_DS(), config)
        trainer = Trainer(_Net(), config)
        trainer.scheduler = ReduceLROnPlateau(trainer.optimizer, factor=0.5, patience=0)
        # Should not error — receives val_loss
        trainer.run(train_loader, val_loader)

    def test_scheduler_via_train_kwarg(self, tmp_vault):
        # Factory form: receives optimizer
        history = run_training(
            _Net, _DS(), config=_cfg(epochs=2, lr=1.0),
            scheduler=lambda opt: StepLR(opt, step_size=1, gamma=0.1),
        )
        assert "train_loss" in history


# ---------------------------------------------------------------------------
# 3. Checkpoint strategies
# ---------------------------------------------------------------------------

class TestCheckpointStrategies:
    def test_every_n_default(self, tmp_vault):
        run_training(
            _Net, _DS(),
            config=_cfg(epochs=4, checkpoint_every=2, checkpoint_strategy="every_n"),
        )
        assert len(useml.focus("level2-project").log()) == 2

    def test_best_strategy_commits_only_on_improvement(self, tmp_vault):
        run_training(
            _Net, _DS(),
            config=_cfg(epochs=3, checkpoint_strategy="best"),
        )
        # At least one snapshot (epoch 1 always improves from None); subsequent
        # epochs may or may not improve.
        n = len(useml.focus("level2-project").log())
        assert 1 <= n <= 3

    def test_last_strategy_commits_exactly_once(self, tmp_vault):
        run_training(
            _Net, _DS(),
            config=_cfg(epochs=4, checkpoint_strategy="last"),
        )
        assert len(useml.focus("level2-project").log()) == 1

    def test_best_with_max_mode(self, tmp_vault):
        # Use accuracy as the tracked metric, with max mode
        run_training(
            _Net, _DS(),
            config=_cfg(epochs=3, checkpoint_strategy="best",
                        checkpoint_metric="accuracy", checkpoint_mode="max"),
            metrics={"accuracy": _accuracy},
        )
        n = len(useml.focus("level2-project").log())
        assert n >= 1


# ---------------------------------------------------------------------------
# 4. Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_disabled_by_default(self, tmp_vault):
        history = run_training(_Net, _DS(), config=_cfg(epochs=3))
        assert len(history["val_loss"]) == 3   # ran all 3 epochs

    def test_triggers_on_no_improvement(self, tmp_vault):
        # patience=0 → first epoch where loss doesn't improve → stop
        history = run_training(
            _Net, _DS(),
            config=_cfg(epochs=10, lr=0.0,  # lr=0 → loss never improves
                        early_stop_patience=1),
        )
        # First epoch establishes "best". Second epoch doesn't improve → counter=1 → stop
        assert len(history["val_loss"]) <= 3

    def test_does_not_trigger_when_loss_keeps_dropping(self, tmp_vault):
        # Normal training should not trigger early stopping with reasonable patience
        history = run_training(
            _Net, _DS(),
            config=_cfg(epochs=2, early_stop_patience=10),
        )
        assert len(history["val_loss"]) == 2


# ---------------------------------------------------------------------------
# 5. on_epoch_end hook
# ---------------------------------------------------------------------------

class TestOnEpochEndHook:
    def test_hook_called_each_epoch(self, tmp_vault):
        calls = []

        class MyTrainer(Trainer):
            def on_epoch_end(self, epoch, train_metrics, val_metrics):
                calls.append((epoch, val_metrics["loss"]))

        config = _cfg(epochs=3)
        train_loader, val_loader = _make_loaders(_DS(), config)
        trainer = MyTrainer(_Net(), config)
        trainer.run(train_loader, val_loader)

        assert [c[0] for c in calls] == [1, 2, 3]

    def test_default_hook_is_noop(self, tmp_vault):
        # Should not error
        config = _cfg(epochs=1)
        train_loader, val_loader = _make_loaders(_DS(), config)
        trainer = Trainer(_Net(), config)
        trainer.run(train_loader, val_loader)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders(ds, config):
    from useml.dataset import load_dataset
    return load_dataset(ds, config)
