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


# ── UC001 — standard useml.train() ────────────────────────────────────────────

class TestUC001StandardTrain:
    def test_history_shape(self):
        config = _cfg()
        history = useml.train(TinyClf, _clf_ds(), config=config)
        assert list(history.keys()) == ["train_loss", "val_loss"]
        assert len(history["train_loss"]) == config.epochs

    def test_checkpoint_created(self):
        useml.train(TinyClf, _clf_ds(), config=_cfg(checkpoint_every=1))
        assert len(_session.project.log()) > 0

    def test_reload_returns_correct_type(self):
        useml.train(TinyClf, _clf_ds(), config=_cfg(checkpoint_every=2))
        model = useml.load("model", _from="\\latest")
        assert isinstance(model, TinyClf)


# ── UC002 — custom TensorDataset ──────────────────────────────────────────────

class TestUC002CustomDataset:
    def test_custom_dataset_passes_through(self):
        history = useml.train(TinyReg, _reg_ds(), config=_cfg(loss="mse"))
        assert len(history["val_loss"]) == 2

    def test_model_reloadable(self):
        useml.train(TinyReg, _reg_ds(), config=_cfg(loss="mse", checkpoint_every=2))
        model = useml.load("model", _from="\\latest")
        assert isinstance(model, TinyReg)


# ── UC003 / UC006 — callable loss archived in snapshot ────────────────────────

class TestUC003CallableLoss:
    def test_callable_accepted(self):
        def my_huber(pred, target): return F.smooth_l1_loss(pred, target)
        history = useml.train(TinyReg, _reg_ds(), config=_cfg(loss=my_huber))
        assert history["train_loss"][-1] > 0

    def test_loss_name_in_snapshot_config(self):
        import yaml
        def my_loss(pred, target): return F.mse_loss(pred, target)
        useml.train(TinyReg, _reg_ds(), config=_cfg(loss=my_loss, checkpoint_every=2))
        snap = _session.project.log()[0]
        cfg_yaml = yaml.safe_load((snap.path / "configs" / "model.yaml").read_text())
        assert cfg_yaml["loss"] == "my_loss"


# ── UC004 — two Trainer runs, one project ─────────────────────────────────────

class TestUC004TwoExperiments:
    def test_two_runs_produce_two_snapshots(self):
        ds = _clf_ds()
        for lr in (1e-3, 1e-4):
            c = _cfg(lr=lr)
            t = useml.Trainer(TinyClf(), c)
            tl, vl = load_dataset(ds, c)
            t.run(tl, vl)
            useml.track("model", t.model, config=c)
            useml.commit(f"lr={lr}", val_loss=0.5)

        assert len(_session.project.log()) == 2

    def test_each_snapshot_has_correct_message(self):
        ds = _clf_ds()
        for lr in (1e-3, 1e-4):
            c = _cfg(lr=lr)
            t = useml.Trainer(TinyClf(), c)
            tl, vl = load_dataset(ds, c)
            t.run(tl, vl)
            useml.track("model", t.model, config=c)
            useml.commit(f"lr={lr}", val_loss=0.5)

        messages = [s.metadata["message"] for s in _session.project.log()]
        assert "lr=0.001" in messages
        assert "lr=0.0001" in messages


# ── UC005 — step_fn + multi-model atomic commit ───────────────────────────────

class TestUC005StepFnAndAtomicCommit:
    def test_step_fn_is_called(self):
        calls = [0]
        def custom_step(model, batch, device):
            calls[0] += 1
            x, _ = batch
            return F.mse_loss(model(x.to(device)), x.to(device))

        useml.train(TinyAE, _clf_ds(), config=_cfg(loss="mse"), step_fn=custom_step)
        assert calls[0] > 0

    def test_atomic_commit_saves_both_components(self):
        c = _cfg(loss="mse")
        useml.track("encoder", TinyAE(), config=c)
        useml.track("decoder", TinyReg(), config=c)
        useml.commit("atomic", val_loss=0.1)

        snap = _session.project.log()[0]
        assert "encoder" in snap.components
        assert "decoder" in snap.components

    def test_atomic_commit_both_reloadable(self):
        c = _cfg(loss="mse")
        useml.track("encoder", TinyAE(), config=c)
        useml.track("decoder", TinyReg(), config=c)
        useml.commit("atomic", val_loss=0.1)

        loaded_enc = useml.load("encoder", _from="\\latest")
        loaded_dec = useml.load("decoder", _from="\\latest")
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
        cfg = _cfg(loss="mse", T=100, beta_start=0.001)
        useml.track("model", TinyReg(), config=cfg)
        useml.commit("test")

        snap = _session.project.log()[0]
        snap_cfg = yaml.safe_load((snap.path / "configs" / "model.yaml").read_text())
        assert snap_cfg["T"] == 100
        assert snap_cfg["beta_start"] == 0.001


# ── UC008 — step_fn closure over external state (distillation pattern) ─────────

class TestUC008StepFnClosure:
    def test_step_fn_closes_over_external_model(self):
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

        config = _cfg()
        useml.train(TinyClf, _clf_ds(), config=config, step_fn=distil_step)
        assert calls[0] > 0


# ── UC009 — manual two-optimizer loop + atomic commit ─────────────────────────

class TestUC009ManualTwoOptimizerLoop:
    def test_manual_track_commit_without_trainer(self):
        G, D = TinyAE(), TinyClf()
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(16, 4)
            g_loss = F.mse_loss(G(x), x)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            d_loss = F.cross_entropy(D(x), torch.randint(0, 2, (16,)))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        useml.track("G", G, config=None)
        useml.track("D", D, config=None)
        useml.commit("epoch 1", g_loss=g_loss.item(), d_loss=d_loss.item())

        snap = _session.project.log()[0]
        assert "G" in snap.components
        assert "D" in snap.components

    def test_atomic_commit_both_g_and_d_reloadable(self):
        useml.track("G", TinyAE(), config=None)
        useml.track("D", TinyClf(), config=None)
        useml.commit("test")

        loaded_G = useml.load("G", _from="\\latest")
        loaded_D = useml.load("D", _from="\\latest")
        assert isinstance(loaded_G, TinyAE)
        assert isinstance(loaded_D, TinyClf)


# ── UC010 — load() passes matching config keys to constructor ─────────────────

class TestUC010LoadConstructorInjection:
    def test_load_injects_latent_dim(self):
        cfg = _cfg(loss="mse", latent_dim=8)
        useml.track("model", TinyLatent(latent_dim=8), config=cfg)
        useml.commit("test", val_loss=0.1)

        loaded = useml.load("model", _from="\\latest")
        assert loaded.latent_dim == 8

    def test_load_falls_back_to_no_arg_when_no_match(self):
        cfg = _cfg(loss="mse")  # no latent_dim key
        useml.track("model", TinyLatent(latent_dim=4), config=cfg)
        useml.commit("test", val_loss=0.1)

        loaded = useml.load("model", _from="\\latest")
        assert loaded.latent_dim == 4  # class default

    def test_two_snapshots_different_latent_dims(self):
        for ld in (8, 16):
            cfg = _cfg(loss="mse", latent_dim=ld)
            useml.track("model", TinyLatent(latent_dim=ld), config=cfg)
            snap = useml.commit(f"ld={ld}", val_loss=0.1)

        snaps = _session.project.log()  # [newest, oldest]
        loaded_16 = useml.load("model", _from=snaps[0].path.name)
        loaded_8  = useml.load("model", _from=snaps[1].path.name)
        assert loaded_16.latent_dim == 16
        assert loaded_8.latent_dim == 8
