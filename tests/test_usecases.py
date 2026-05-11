"""
Use-case integration tests.

Each test class maps to a use case in playground/usecases/ and verifies
the key API pattern works end-to-end on a tiny dataset.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import useml
from useml import Config
from useml.session.manager import _session
from useml.dataset import load_dataset
from useml.template.trainer import Trainer


# ── Shared helpers ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_session(tmp_path):
    useml.init(str(tmp_path / "vault"))
    useml.new("test")
    yield
    _session.vault = None
    _session._project = None
    _session.components = {}
    _session._is_dirty = False
    _session._stash = {}


def _clf_ds(n=200):
    return TensorDataset(torch.randn(n, 4), torch.randint(0, 2, (n,)))

def _reg_ds(n=200):
    X = torch.randn(n, 4)
    return TensorDataset(X, X.sum(1))

def _cfg(**kw):
    defaults = dict(epochs=2, batch_size=32, lr=1e-3, device="cpu",
                    checkpoint_every=9999, loss="cross_entropy")
    defaults.update(kw)
    return Config(**defaults)


class TinyClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x): return self.fc(x)


class TinyReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 4))
    def forward(self, x): return self.net(x)


class TinyLatent(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(4, latent_dim)
    def forward(self, x): return self.fc(x)


# ── UC001 — project.runs.new().train() ───────────────────────────────────────

class TestUC001StandardTrain:
    def test_history_shape(self):
        project = useml.focus("test")
        config = _cfg()
        history = project.runs.new(TinyClf, _clf_ds(), config=config).train()
        assert "train_loss" in history and "val_loss" in history
        assert len(history["train_loss"]) == config.epochs

    def test_checkpoint_created(self):
        project = useml.focus("test")
        project.runs.new(TinyClf, _clf_ds(), config=_cfg(checkpoint_every=1)).train()
        assert len(project.log()) > 0

    def test_reload_returns_correct_type(self):
        project = useml.focus("test")
        project.runs.new(TinyClf, _clf_ds(), config=_cfg(checkpoint_every=2)).train()
        m = project.runs.latest.load()
        assert isinstance(m, TinyClf)


# ── UC002 — custom TensorDataset ──────────────────────────────────────────────

class TestUC002CustomDataset:
    def test_custom_dataset_passes_through(self):
        project = useml.focus("test")
        history = project.runs.new(TinyReg, _reg_ds(), config=_cfg(loss="mse")).train()
        assert len(history["val_loss"]) == 2

    def test_model_reloadable(self):
        project = useml.focus("test")
        project.runs.new(TinyReg, _reg_ds(), config=_cfg(loss="mse", checkpoint_every=2)).train()
        m = project.runs.latest.load()
        assert isinstance(m, TinyReg)


# ── UC003 / UC006 — callable loss archived in snapshot ────────────────────────

class TestUC003CallableLoss:
    def test_callable_accepted(self):
        project = useml.focus("test")
        def my_huber(pred, target): return F.smooth_l1_loss(pred, target)
        history = project.runs.new(TinyReg, _reg_ds(), config=_cfg(loss=my_huber)).train()
        assert history["train_loss"][-1] > 0

    def test_loss_name_in_snapshot_config(self):
        import yaml
        project = useml.focus("test")
        def my_loss(pred, target): return F.mse_loss(pred, target)
        project.runs.new(TinyReg, _reg_ds(), config=_cfg(loss=my_loss, checkpoint_every=2)).train()
        snap = project.log()[0]
        cfg_yaml = yaml.safe_load((snap.path / "configs" / "tinyreg.yaml").read_text())
        assert cfg_yaml["loss"] == "my_loss"


# ── UC004 — two runs, one project ─────────────────────────────────────────────

class TestUC004TwoExperiments:
    def test_two_runs_produce_two_snapshots(self):
        project = useml.focus("test")
        ds = _clf_ds()
        for lr in (1e-3, 1e-4):
            c = _cfg(lr=lr, checkpoint_strategy="last")
            project.runs.new(TinyClf, ds, config=c).train()
        assert len(project.log()) == 2

    def test_each_snapshot_has_correct_message(self):
        project = useml.focus("test")
        ds = _clf_ds()
        for lr in (1e-3, 1e-4):
            c = _cfg(lr=lr, checkpoint_strategy="last")
            project.runs.new(TinyClf, ds, config=c).train()
        messages = [s.metadata.get("message", "") for s in project.log()]
        assert len(messages) == 2
        assert all("epoch" in m for m in messages)


# ── UC005 — step_fn + multi-model atomic commit ───────────────────────────────

class TestUC005StepFnAndAtomicCommit:
    def test_step_fn_is_called(self):
        project = useml.focus("test")
        calls = [0]
        def custom_step(model, batch, device):
            calls[0] += 1
            x, _ = batch
            return F.mse_loss(model(x.to(device)), x.to(device))

        project.runs.new(TinyAE, _clf_ds(), config=_cfg(loss="mse")).train(step_fn=custom_step)
        assert calls[0] > 0

    def test_atomic_commit_saves_both_components(self):
        project = useml.focus("test")
        c = _cfg(loss="mse")
        _session.track("encoder", TinyAE(), config=c)
        _session.track("decoder", TinyReg(), config=c)
        _session.commit("atomic", val_loss=0.1)

        snap = project.log()[0]
        assert "encoder" in snap.components
        assert "decoder" in snap.components

    def test_atomic_commit_both_reloadable(self):
        c = _cfg(loss="mse")
        _session.track("encoder", TinyAE(), config=c)
        _session.track("decoder", TinyReg(), config=c)
        _session.commit("atomic", val_loss=0.1)

        loaded_enc = _session.load("encoder", _from="\\latest")
        loaded_dec = _session.load("decoder", _from="\\latest")
        assert isinstance(loaded_enc, TinyAE)
        assert isinstance(loaded_dec, TinyReg)


# ── UC007 / UC010 — Config custom fields ──────────────────────────────────────

class TestUC007ConfigCustomFields:
    def test_custom_field_accessible(self):
        cfg = Config(epochs=2, device="cpu", latent_dim=16, kl_weight=1e-3)
        assert cfg.latent_dim == 16
        assert cfg.kl_weight == 1e-3

    def test_custom_field_in_to_dict(self):
        cfg = Config(epochs=2, device="cpu", T=200, beta_start=1e-4)
        d = cfg.to_dict()
        assert d["T"] == 200
        assert d["beta_start"] == 1e-4

    def test_standard_fields_still_present(self):
        cfg = Config(epochs=2, device="cpu", latent_dim=8)
        d = cfg.to_dict()
        assert "epochs" in d and d["epochs"] == 2
        assert "lr" in d

    def test_custom_fields_serialised_to_snapshot_yaml(self):
        import yaml
        project = useml.focus("test")
        cfg = _cfg(loss="mse", T=100, beta_start=0.001)
        _session.track("model", TinyReg(), config=cfg)
        _session.commit("test")

        snap = project.log()[0]
        snap_cfg = yaml.safe_load((snap.path / "configs" / "model.yaml").read_text())
        assert snap_cfg["T"] == 100
        assert snap_cfg["beta_start"] == 0.001


# ── UC008 — step_fn closure over external state (distillation pattern) ─────────

class TestUC008StepFnClosure:
    def test_step_fn_closes_over_external_model(self):
        project = useml.focus("test")
        teacher = TinyClf()
        teacher.eval()
        criterion = nn.CrossEntropyLoss()

        calls = [0]
        def distil_step(model, batch, device):
            calls[0] += 1
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_out = teacher(x)
            return criterion(model(x), t_out.argmax(1))

        project.runs.new(TinyClf, _clf_ds(), config=_cfg()).train(step_fn=distil_step)
        assert calls[0] > 0


# ── UC009 — manual two-optimizer loop + atomic commit ─────────────────────────

class TestUC009ManualTwoOptimizerLoop:
    def test_manual_track_commit_without_trainer(self):
        project = useml.focus("test")
        G, D = TinyAE(), TinyClf()
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(16, 4)
            g_loss = F.mse_loss(G(x), x)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            d_loss = F.cross_entropy(D(x), torch.randint(0, 2, (16,)))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        _session.track("G", G, config=None)
        _session.track("D", D, config=None)
        _session.commit("epoch 1", g_loss=g_loss.item(), d_loss=d_loss.item())

        snap = project.log()[0]
        assert "G" in snap.components
        assert "D" in snap.components

    def test_atomic_commit_both_g_and_d_reloadable(self):
        _session.track("G", TinyAE(), config=None)
        _session.track("D", TinyClf(), config=None)
        _session.commit("test")

        loaded_G = _session.load("G", _from="\\latest")
        loaded_D = _session.load("D", _from="\\latest")
        assert isinstance(loaded_G, TinyAE)
        assert isinstance(loaded_D, TinyClf)


# ── UC010 — load() passes matching config keys to constructor ─────────────────

class TestUC010LoadConstructorInjection:
    def test_load_injects_latent_dim(self):
        cfg = _cfg(loss="mse", latent_dim=8)
        _session.track("model", TinyLatent(latent_dim=8), config=cfg)
        _session.commit("test", val_loss=0.1)

        loaded = _session.load("model", _from="\\latest")
        assert loaded.latent_dim == 8

    def test_load_falls_back_to_no_arg_when_no_match(self):
        cfg = _cfg(loss="mse")  # no latent_dim key
        _session.track("model", TinyLatent(latent_dim=4), config=cfg)
        _session.commit("test", val_loss=0.1)

        loaded = _session.load("model", _from="\\latest")
        assert loaded.latent_dim == 4  # class default

    def test_two_snapshots_different_latent_dims(self):
        project = useml.focus("test")
        for ld in (8, 16):
            cfg = _cfg(loss="mse", latent_dim=ld)
            _session.track("model", TinyLatent(latent_dim=ld), config=cfg)
            _session.commit(f"ld={ld}", val_loss=0.1)

        snaps = project.log()  # [newest, oldest]
        loaded_16 = _session.load("model", _from=snaps[0].path.name)
        loaded_8  = _session.load("model", _from=snaps[1].path.name)
        assert loaded_16.latent_dim == 16
        assert loaded_8.latent_dim == 8


# ── UC012 — LR scheduler ──────────────────────────────────────────────────────

class TestUC012LRScheduler:
    """Scheduler assigned as a Trainer attribute — no subclassing required."""

    def test_step_lr_via_trainer_attribute(self):
        from torch.optim.lr_scheduler import StepLR

        config = _cfg(lr=1.0, epochs=3)
        train_loader, val_loader = load_dataset(_clf_ds(), config)
        trainer = Trainer(TinyClf(), config)
        trainer.scheduler = StepLR(trainer.optimizer, step_size=1, gamma=0.5)
        trainer.run(train_loader, val_loader)

        # After 3 step-per-epoch decays of 0.5: lr = 1.0 * 0.5^3 = 0.125
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(final_lr - 0.125) < 1e-6

    def test_scheduler_via_train_kwarg_factory(self):
        from torch.optim.lr_scheduler import StepLR

        project = useml.focus("test")
        history = project.runs.new(
            TinyClf, _clf_ds(), config=_cfg(epochs=2, lr=1.0)
        ).train(scheduler=lambda opt: StepLR(opt, step_size=1, gamma=0.1))
        assert "train_loss" in history and len(history["train_loss"]) == 2

    def test_reduce_lr_on_plateau_no_error(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        config = _cfg(epochs=2)
        train_loader, val_loader = load_dataset(_clf_ds(), config)
        trainer = Trainer(TinyClf(), config)
        trainer.scheduler = ReduceLROnPlateau(trainer.optimizer, patience=0)
        trainer.run(train_loader, val_loader)  # must not raise


# ── UC013 — Best-val checkpointing + early stopping ───────────────────────────

class TestUC013BestValCheckpoint:
    """checkpoint_strategy='best' saves only when validation improves."""

    def test_best_strategy_at_most_one_per_epoch(self):
        project = useml.focus("test")
        project.runs.new(
            TinyClf, _clf_ds(),
            config=_cfg(epochs=4, checkpoint_strategy="best"),
        ).train()
        n = len(project.log())
        assert 1 <= n <= 4

    def test_last_strategy_exactly_one_snapshot(self):
        project = useml.focus("test")
        project.runs.new(
            TinyClf, _clf_ds(),
            config=_cfg(epochs=5, checkpoint_strategy="last"),
        ).train()
        assert len(project.log()) == 1

    def test_early_stopping_shortens_training(self):
        project = useml.focus("test")
        history = project.runs.new(
            TinyClf, _clf_ds(),
            config=_cfg(epochs=20, lr=0.0, early_stop_patience=1),
        ).train()
        # Must stop well before 20 epochs
        assert len(history["val_loss"]) <= 4

    def test_best_with_custom_metric_max_mode(self):
        def accuracy(model, batch, device):
            x, y = batch
            return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()

        project = useml.focus("test")
        project.runs.new(
            TinyClf, _clf_ds(),
            config=_cfg(epochs=3, checkpoint_strategy="best",
                        checkpoint_metric="accuracy", checkpoint_mode="max"),
        ).train(metrics={"accuracy": accuracy})
        assert len(project.log()) >= 1


# ── UC014 — Custom metrics ────────────────────────────────────────────────────

class TestUC014CustomMetrics:
    """metrics dict: (model, batch, device) -> scalar, logged to history + snapshot."""

    def test_metric_in_history(self):
        def accuracy(model, batch, device):
            x, y = batch
            return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()

        project = useml.focus("test")
        history = project.runs.new(
            TinyClf, _clf_ds(), config=_cfg(epochs=2)
        ).train(metrics={"accuracy": accuracy})
        assert "train_accuracy" in history
        assert "val_accuracy" in history
        assert len(history["val_accuracy"]) == 2

    def test_metric_range_zero_one(self):
        def accuracy(model, batch, device):
            x, y = batch
            return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()

        project = useml.focus("test")
        history = project.runs.new(
            TinyClf, _clf_ds(), config=_cfg(epochs=1)
        ).train(metrics={"accuracy": accuracy})
        assert 0.0 <= history["val_accuracy"][0] <= 1.0

    def test_metric_written_to_snapshot_metadata(self):
        import yaml

        def accuracy(model, batch, device):
            x, y = batch
            return (model(x.to(device)).argmax(-1) == y.to(device)).float().mean()

        project = useml.focus("test")
        project.runs.new(
            TinyClf, _clf_ds(), config=_cfg(epochs=1, checkpoint_every=1)
        ).train(metrics={"accuracy": accuracy})
        snap = project.log()[0]
        meta = yaml.safe_load((snap.path / "metadata.yaml").read_text())
        assert "val_accuracy" in meta.get("metrics", {})


# ── UC015 — Trainer Pattern B (custom loop) ───────────────────────────────────

class TestUC015TrainerPatternB:
    """run.register() + update()/epoch_end()/epochs() for full-control loops."""

    def test_two_models_committed_atomically(self):
        """Two registered models end up in the same snapshot."""
        project = useml.focus("test")
        config = _cfg(epochs=2, checkpoint_every=1)
        G, D   = TinyAE(), TinyClf()
        opt_G  = torch.optim.Adam(G.parameters(), lr=1e-3)
        opt_D  = torch.optim.Adam(D.parameters(), lr=1e-3)

        run = project.runs.new(config=config)
        run.register("G", G, optimizer=opt_G)
        run.register("D", D, optimizer=opt_D)

        ds = _clf_ds()
        loader = torch.utils.data.DataLoader(ds, batch_size=16)

        for epoch in run.epochs(config.epochs):
            for batch in loader:
                x, y = batch
                g_loss = torch.nn.functional.mse_loss(G(x), x)
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                d_loss = torch.nn.functional.cross_entropy(D(x), y)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                run.update(n=x.size(0), g_loss=g_loss.item(), d_loss=d_loss.item())
            run.epoch_end(epoch)

        snap = project.log()[0]
        assert "G" in snap.components
        assert "D" in snap.components

    def test_history_keys_reflect_update_names(self):
        project = useml.focus("test")
        config  = _cfg(epochs=2, checkpoint_every=9999)
        run = project.runs.new(config=config)
        run.register("model", TinyClf())

        ds     = _clf_ds()
        loader = torch.utils.data.DataLoader(ds, batch_size=16)

        for epoch in run.epochs(config.epochs):
            for batch in loader:
                x, _ = batch
                run.update(n=x.size(0), my_metric=0.5)
            run.epoch_end(epoch)

        assert "train_my_metric" in run.history
        assert len(run.history["train_my_metric"]) == 2

    def test_no_val_loop_fallback_to_train_metrics_for_checkpoint(self):
        """Best-val checkpoint works even without a val loop (uses train metrics)."""
        project = useml.focus("test")
        config  = _cfg(
            epochs=3,
            checkpoint_strategy="best",
            checkpoint_metric="g_loss",
            checkpoint_mode="min",
        )
        G      = TinyAE()
        opt_G  = torch.optim.Adam(G.parameters(), lr=1e-3)
        run = project.runs.new(config=config)
        run.register("G", G, optimizer=opt_G)

        ds     = _clf_ds()
        loader = torch.utils.data.DataLoader(ds, batch_size=16)

        for epoch in run.epochs(config.epochs):
            for batch in loader:
                x, _ = batch
                loss = torch.nn.functional.mse_loss(G(x), x)
                opt_G.zero_grad(); loss.backward(); opt_G.step()
                run.update(n=x.size(0), g_loss=loss.item())
            run.epoch_end(epoch)

        assert len(project.log()) >= 1

    def test_register_returns_self_for_chaining(self):
        project = useml.focus("test")
        config  = _cfg()
        run = project.runs.new(config=config)
        result  = run.register("A", TinyClf()).register("B", TinyReg())
        assert result is run
        assert "A" in run._models
        assert "B" in run._models

    def test_pattern_a_backward_compat(self):
        """Trainer(model, config).run() still works unchanged."""
        config  = _cfg(epochs=2)
        trainer = Trainer(TinyClf(), config)
        tl, vl  = load_dataset(_clf_ds(), config)
        history = trainer.run(tl, vl)
        assert "train_loss" in history and "val_loss" in history


# ── UC016 — Training resume ───────────────────────────────────────────────────

class TestUC016TrainingResume:
    """Optimizer state persisted in snapshot; training continues from saved epoch."""

    def test_optimizer_state_saved_in_snapshot(self):
        """optimizers/<name>.pth must exist after a checkpoint."""
        project = useml.focus("test")
        project.runs.new(
            TinyClf, _clf_ds(), config=_cfg(epochs=1, checkpoint_every=1)
        ).train()
        snap = project.log()[0]
        assert (snap.path / "optimizers" / "tinyclf.pth").exists()

    def test_resume_restores_optimizer_state(self):
        """After resume, optimizer param groups match the saved run."""
        project = useml.focus("test")
        config = _cfg(epochs=2, lr=0.123, checkpoint_every=1)
        project.runs.new(TinyClf, _clf_ds(), config=config).train()

        # Fresh run for resume — register same component name
        run2 = project.runs.new(config=config)
        run2.register("tinyclf", TinyClf())
        run2.resume("\\latest")

        restored_lr = run2.optimizer.param_groups[0]["lr"]
        assert abs(restored_lr - 0.123) < 1e-9

    def test_resume_sets_epoch_counter(self):
        """epochs() iterator should start from saved_epoch + 1."""
        project = useml.focus("test")
        config = _cfg(epochs=3, checkpoint_every=1)
        project.runs.new(TinyClf, _clf_ds(), config=config).train()
        snap = project.log()[0]
        saved_epoch = snap.metadata.get("metrics", {}).get("epoch", 0)

        run2 = project.runs.new(config=config)
        run2.register("tinyclf", TinyClf())
        run2.resume("\\latest")

        assert run2._epoch == saved_epoch
        assert run2._epoch == 3

    def test_train_resume_from_runs_remaining_epochs(self):
        """run.train(resume_from=) only runs the remaining epochs."""
        project = useml.focus("test")
        config  = _cfg(epochs=4, checkpoint_strategy="last")
        config2 = _cfg(epochs=2, checkpoint_strategy="last")
        # First run: 2 epochs
        project.runs.new(TinyClf, _clf_ds(), config=config2).train()
        assert len(project.log()) == 1   # "last" → one snap

        # Resume: should run epochs 3 and 4 only
        history = project.runs.new(
            TinyClf, _clf_ds(), config=config
        ).train(resume_from="\\latest")
        assert len(history["val_loss"]) == 2

    def test_resume_pattern_b_custom_loop(self):
        """Pattern B: resume() + epochs() continues from saved checkpoint."""
        project = useml.focus("test")
        config = _cfg(epochs=4, checkpoint_every=2)
        G      = TinyAE()
        opt_G  = torch.optim.Adam(G.parameters(), lr=1e-3)
        run = project.runs.new(config=config)
        run.register("G", G, optimizer=opt_G)
        loader = torch.utils.data.DataLoader(_clf_ds(), batch_size=16)

        for epoch in run.epochs(config.epochs):
            for batch in loader:
                x, _ = batch
                loss = torch.nn.functional.mse_loss(G(x), x)
                opt_G.zero_grad(); loss.backward(); opt_G.step()
                run.update(n=x.size(0), loss=loss.item())
            run.epoch_end(epoch)

        # Resume into a new run — should yield 0 new epochs (already done)
        run2 = project.runs.new(config=config)
        run2.register("G", TinyAE(), optimizer=torch.optim.Adam(TinyAE().parameters()))
        run2.resume("\\latest")

        yielded = list(run2.epochs(config.epochs))
        assert len(yielded) == 0    # nothing left to run

    def test_best_strategy_continues_correctly_after_resume(self):
        """_best_ckpt_value is restored so 'best' strategy doesn't re-commit on first epoch."""
        project = useml.focus("test")
        config = _cfg(epochs=2, checkpoint_strategy="best")
        project.runs.new(TinyClf, _clf_ds(), config=config).train()

        run2 = project.runs.new(config=config)
        run2.register("tinyclf", TinyClf())
        run2.resume("\\latest")
        assert run2._best_ckpt_value is not None


# =============================================================================
# Side Track A — project.runs / project.models / Run
# =============================================================================


def _train_one_run(project, model_cls=TinyClf, **cfg_kw):
    """Helper: train a single run, saving exactly one snapshot via 'last'."""
    cfg_kw.setdefault("checkpoint_strategy", "last")
    cfg_kw.setdefault("loss",
                      "mse" if model_cls is TinyReg else "cross_entropy")
    cfg = _cfg(**cfg_kw)
    ds = _clf_ds() if model_cls is TinyClf else _reg_ds()
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))],
        generator=torch.Generator().manual_seed(0),
    )
    run = project.runs.new(model_cls, config=cfg)
    run.train(DataLoader(train_ds, batch_size=32),
              DataLoader(val_ds,   batch_size=32))
    return run


class TestSideTrackANewAPI:
    """The new project-centric API: runs, models, versions."""

    def test_useml_new_returns_project(self):
        """useml.new() now returns the Project (was None)."""
        from useml.vault.project import Project
        proj = useml.focus("test")
        assert isinstance(proj, Project)

    def test_useml_focus_returns_project(self):
        proj = useml.focus("test")
        assert proj is not None
        assert proj.path.name == "test"

    def test_project_runs_empty_initially(self):
        proj = useml.focus("test")
        assert len(proj.runs) == 0

    def test_project_models_empty_initially(self):
        proj = useml.focus("test")
        assert len(proj.models) == 0

    def test_runs_new_creates_run_with_class_name(self):
        proj = useml.focus("test")
        run = proj.runs.new(TinyClf, config=_cfg())
        # Auto-named after the class (lowercased)
        assert "tinyclf" in run._models

    def test_runs_new_with_explicit_name(self):
        proj = useml.focus("test")
        run = proj.runs.new(TinyClf, config=_cfg(), name="encoder")
        assert "encoder" in run._models
        assert "tinyclf" not in run._models

    def test_runs_new_creates_run_with_instance(self):
        proj = useml.focus("test")
        m = TinyClf()
        run = proj.runs.new(m, config=_cfg())
        assert "tinyclf" in run._models
        assert run._models["tinyclf"] is m

    def test_run_train_completes_and_saves(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        assert len(proj.runs) >= 1

    def test_versions_yaml_written_on_save(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        assert (proj.path / "versions.yaml").exists()
        idx = proj._read_versions_yaml()
        assert "tinyclf" in idx
        assert "v1" in idx["tinyclf"]

    def test_multiple_runs_increment_versions(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        idx = proj._read_versions_yaml()
        assert sorted(idx["tinyclf"].keys()) == ["v1", "v2"]

    def test_models_view_indexable(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        reg = proj.models["tinyclf"]
        assert len(reg) == 2
        assert "v1" in reg
        assert "v2" in reg

    def test_model_registry_latest(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        latest = proj.models["tinyclf"].latest
        assert latest.label == "v2"

    def test_model_registry_best(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        best = proj.models["tinyclf"].best("val_loss")
        # best by min val_loss → returns one of v1/v2
        assert best.label in {"v1", "v2"}
        # the "best" must have the minimum
        all_losses = [v.metrics.get("val_loss") for v in proj.models["tinyclf"]]
        assert best.metrics["val_loss"] == min(all_losses)

    def test_model_version_metrics_and_config(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        v = proj.models["tinyclf"]["v1"]
        assert "val_loss" in v.metrics
        assert "epoch" in v.metrics
        assert v.config["lr"] == 1e-3

    def test_model_version_load_with_instance(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        m = proj.models["tinyclf"]["v1"].load(TinyClf())
        assert isinstance(m, TinyClf)

    def test_model_version_load_auto_instantiate(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        m = proj.models["tinyclf"]["v1"].load()
        assert isinstance(m, TinyClf)

    def test_runs_latest_and_best(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        assert proj.runs.latest is not None
        assert proj.runs.best("val_loss") is not None

    def test_runs_iteration_newest_first(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)
        records = list(proj.runs)
        assert len(records) >= 2
        # newest first → first record has the largest id (timestamp)
        assert records[0].id > records[1].id

    def test_run_record_load_single_model(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        m = proj.runs.latest.load()  # only one model → no name needed
        assert isinstance(m, TinyClf)

    def test_run_record_load_by_name(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        m = proj.runs.latest.load("tinyclf", model=TinyClf())
        assert isinstance(m, TinyClf)

    def test_pattern_b_atomic_multi_model(self):
        """Pattern B: G + D registered, both saved atomically in one snapshot."""
        proj = useml.focus("test")

        class G_(nn.Module):
            def __init__(self):
                super().__init__(); self.fc = nn.Linear(4, 2)
            def forward(self, x): return self.fc(x)

        class D_(nn.Module):
            def __init__(self):
                super().__init__(); self.fc = nn.Linear(2, 1)
            def forward(self, x): return self.fc(x)

        cfg = _cfg(epochs=1, checkpoint_strategy="best",
                   checkpoint_metric="g_loss")
        run = proj.runs.new(config=cfg)
        run.register("G", G_())
        run.register("D", D_())
        for epoch in run.epochs(cfg.epochs):
            for _ in range(3):
                run.update(n=4, g_loss=0.5, d_loss=0.4)
            run.epoch_end(epoch)

        # Both models saved in same snapshot
        assert "G" in proj.models
        assert "D" in proj.models
        assert proj.models["G"].latest.snapshot_tag == proj.models["D"].latest.snapshot_tag

    def test_backward_compat_no_versions_yaml(self):
        """An existing project without versions.yaml: dynamic scan rebuilds it."""
        proj = useml.focus("test")
        _train_one_run(proj)
        _train_one_run(proj)

        # Wipe the index — simulate an old project
        (proj.path / "versions.yaml").unlink()
        assert not (proj.path / "versions.yaml").exists()

        # Access models → triggers rebuild
        view = proj.models
        assert "tinyclf" in view
        assert len(view["tinyclf"]) == 2
        assert (proj.path / "versions.yaml").exists()  # cached for next time

    def test_runs_filter_by_model(self):
        proj = useml.focus("test")
        _train_one_run(proj, TinyClf)
        _train_one_run(proj, TinyReg, loss="mse")
        clf_runs = list(proj.runs.filter(model="tinyclf"))
        reg_runs = list(proj.runs.filter(model="tinyreg"))
        assert len(clf_runs) == 1
        assert len(reg_runs) == 1

    def test_runs_leaderboard_returns_string(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        out = proj.runs.leaderboard("val_loss")
        assert isinstance(out, str)
        assert "val_loss" in out

    def test_project_show_doesnt_crash(self, capsys):
        proj = useml.focus("test")
        _train_one_run(proj)
        proj.show()
        captured = capsys.readouterr()
        assert "Project" in captured.out
        assert "tinyclf" in captured.out

    def test_model_registry_best_raises_on_unknown_metric(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        with pytest.raises(KeyError):
            proj.models["tinyclf"].best("nonexistent_metric")

    def test_model_registry_getitem_unknown_version_raises(self):
        proj = useml.focus("test")
        _train_one_run(proj)
        with pytest.raises(KeyError):
            proj.models["tinyclf"]["v999"]

    def test_run_resume_continues_from_saved_epoch(self):
        proj = useml.focus("test")
        cfg = _cfg(epochs=4, checkpoint_strategy="best")
        ds = _clf_ds()
        loader = DataLoader(ds, batch_size=32)
        # Initial run, epochs=2
        cfg2 = _cfg(epochs=2, checkpoint_strategy="best")
        run = proj.runs.new(TinyClf, config=cfg2)
        run.train(loader, loader)
        assert len(proj.runs) >= 1

        # Resume into a new run object → should pick up _epoch
        run2 = proj.runs.new(TinyClf, config=cfg)
        run2.resume("\\latest")
        assert run2._epoch >= 1   # picked up at least one epoch
