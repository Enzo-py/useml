from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from useml import config
from useml.config import ConfigError, Library


# ── classes the configs build ─────────────────────────────────────────────────

class Encoder:
    def __init__(self, name: str = "ViT-B", dim: int = 512, device=None):
        self.name, self.dim, self.device = name, dim, device
        Encoder.built += 1


Encoder.built = 0


class Matcher:
    def __init__(self, alpha: float = 0.5, solver: str = "sinkhorn", steps: int = 10):
        self.alpha, self.solver, self.steps = alpha, solver, steps


class Model:
    def __init__(self, encoder=None, matcher: Matcher | None = None, temperature: float = 1.0):
        self.encoder, self.matcher, self.temperature = encoder, matcher, temperature


class Scorer:
    def __init__(self, scorer_model: "Model", batch_size: int = 8):
        self.scorer_model, self.batch_size = scorer_model, batch_size


class Shared:
    """Injects its encoder into its matcher: the encoder must be written once."""

    def __init__(self, encoder=None, inner=None):
        self.encoder = encoder
        self.inner = inner if inner is not None else Model(encoder=encoder)


class Broken:
    def __init__(self, value: int = 1):
        raise RuntimeError("weights missing")


class Opaque:
    def __init__(self, handle=None):
        self._secret = handle


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "configs"
    library = Library(root, kinds={"model": "models", "scorer": "scorers"},
                      default_kind="model")
    for cls in (Encoder, Matcher, Model, Scorer, Shared, Broken, Opaque):
        library.register(cls)
    (root / "models").mkdir(parents=True)
    (root / "scorers").mkdir(parents=True)
    Encoder.built = 0
    return library


def write(lib, name, text, kind="models"):
    (lib.root / kind / f"{name}.yaml").write_text(textwrap.dedent(text))


# ── reading and building ──────────────────────────────────────────────────────

def test_tree_is_built_in_depth_and_omitted_keys_stay_default(lib):
    write(lib, "m", """
        meta: {display_name: "My model"}
        type: Model
        encoder: {type: Encoder, dim: 768}
        matcher: {type: Matcher, alpha: 0.2}
    """)
    model = lib.load("m")
    assert model.encoder.dim == 768 and model.encoder.name == "ViT-B"
    assert model.matcher.alpha == 0.2 and model.matcher.solver == "sinkhorn"
    assert model.config.display_name == "My model"


def test_load_is_memoised_and_reload_rebuilds(lib):
    write(lib, "e", "type: Encoder\n")
    assert lib.load("e") is lib.load("e")
    assert lib.load("e", reload=True) is not lib.load("e", reload=False) or Encoder.built == 2
    assert Encoder.built == 2


def test_misspelled_key_raises_with_close_name_and_address(lib):
    write(lib, "m", "type: Model\nmatcher: {type: Matcher, alpah: 0.1}\n")
    with pytest.raises(ConfigError, match=r"matcher\..*'alpah' \(close to alpha\)"):
        lib.load("m")


def test_unknown_type_suggests(lib):
    write(lib, "m", "type: Modle\n")
    with pytest.raises(ConfigError, match="Did you mean Model"):
        lib.load("m")


def test_unknown_meta_and_missing_type_raise(lib):
    write(lib, "a", "meta: {dispaly_name: x}\ntype: Encoder\n")
    write(lib, "b", "dim: 3\n")
    with pytest.raises(ConfigError, match="does not know"):
        lib.load_raw("a")
    with pytest.raises(ConfigError, match="`type:` is missing"):
        lib.load_raw("b")


def test_unknown_name_suggests(lib):
    write(lib, "resnet_ft", "type: Encoder\n")
    with pytest.raises(ConfigError, match="Did you mean resnet_ft"):
        lib.load("resnet_fth")


def test_constructor_error_carries_the_file(lib):
    write(lib, "broken", "type: Broken\n")
    with pytest.raises(ConfigError, match=r"broken.yaml.*Broken\(…\) raised RuntimeError"):
        lib.load("broken")


def test_list_without_type_stays_a_list(lib):
    assert lib.build_node([{"a": 1}, 2]) == [{"a": 1}, 2]


# ── refs ──────────────────────────────────────────────────────────────────────

def test_shared_reference_is_built_once(lib):
    write(lib, "enc", "type: Encoder\ndim: 64\n")
    write(lib, "m1", "type: Model\nencoder: {ref: enc}\n")
    write(lib, "m2", "type: Model\nencoder: {ref: enc}\n")
    assert lib.load("m1").encoder is lib.load("m2").encoder
    assert Encoder.built == 1


def test_reference_with_overrides(lib):
    write(lib, "enc", "type: Encoder\ndim: 64\n")
    write(lib, "m", "type: Model\nencoder: {ref: enc, dim: 128}\n")
    assert lib.load("m").encoder.dim == 128


def test_reference_with_type_raises(lib):
    write(lib, "m", "type: Model\nencoder: {ref: enc, type: Encoder}\n")
    with pytest.raises(ConfigError, match="cannot also declare"):
        lib.load("m")


def test_kind_resolution(lib):
    write(lib, "same", "type: Encoder\n")
    write(lib, "same", "type: Matcher\n", kind="scorers")
    write(lib, "only_scorer", "type: Matcher\n", kind="scorers")
    assert isinstance(lib.load("same"), Encoder)               # the default kind first
    assert isinstance(lib.load("same", "scorer"), Matcher)
    assert isinstance(lib.load("only_scorer"), Matcher)        # the only holder


def test_kinds_default_to_sub_folders(tmp_path):
    (tmp_path / "c" / "models").mkdir(parents=True)
    (tmp_path / "c" / "datasets").mkdir()
    assert list(Library(tmp_path / "c").kinds) == ["datasets", "models"]


# ── overrides ─────────────────────────────────────────────────────────────────

def test_dict_override_merges_into_the_node(lib):
    write(lib, "m", "type: Model\nmatcher: {type: Matcher, alpha: 0.2, steps: 50}\n")
    model = lib.load("m", matcher={"solver": "bcd"})
    assert (model.matcher.alpha, model.matcher.solver, model.matcher.steps) == (0.2, "bcd", 50)


def test_dotted_overrides(lib):
    write(lib, "m", "type: Model\nmatcher: {type: Matcher, alpha: 0.2}\n")
    model = lib.load("m", overrides={"matcher.solver": "bcd", "temperature": 3.0})
    assert model.matcher.solver == "bcd" and model.matcher.alpha == 0.2
    assert model.temperature == 3.0


def test_override_with_type_replaces_the_node(lib):
    write(lib, "m", "type: Model\nmatcher: {type: Matcher, alpha: 0.2, steps: 50}\n")
    model = lib.load("m", matcher={"type": "Matcher", "solver": "x"})
    assert model.matcher.steps == 10 and model.matcher.alpha == 0.5


def test_built_object_override_and_no_memo(lib):
    write(lib, "m", "type: Model\n")
    own = Matcher(alpha=0.9)
    first = lib.load("m", matcher=own)
    assert first.matcher is own and lib.load("m", matcher=own) is not first


def test_override_changes_hash_and_cited_name(lib):
    write(lib, "m", "type: Model\ntemperature: 1.0\n")
    plain = lib.load_raw("m")
    over = lib.load_raw("m", temperature=2.0)
    assert plain.hash != over.hash and over.cited_name == "m+ovr"
    assert over.label == "m [temperature=2]"
    assert lib.load_raw("m", temperature=2.0).hash == over.hash         # stable


def test_hash_ignores_formatting_and_meta(lib):
    write(lib, "a", "type: Model\ntemperature: 1.0\nmatcher: {type: Matcher, alpha: 0.2}\n")
    write(lib, "b", """
        meta: {note: "same thing"}
        matcher:
          alpha: 0.2
          type: Matcher
        temperature: 1.0
        type: Model
    """)
    assert lib.load_raw("a").hash == lib.load_raw("b").hash


def test_parse_override():
    assert config.parse_override("a.b=3") == ("a.b", 3)
    assert config.parse_override("x={type: M}") == ("x", {"type": "M"})
    assert config.parse_override("name=bcd") == ("name", "bcd")


# ── the object slot (`model=`) ────────────────────────────────────────────────

def test_model_fills_the_inferred_slot(lib):
    write(lib, "m", "type: Model\n")
    write(lib, "s", "type: Scorer\nscorer_model: {ref: m}\n", kind="scorers")
    write(lib, "m2", "type: Model\ntemperature: 7.0\n")
    scorer = lib.load("s", model="m2")              # a config NAME
    assert scorer.scorer_model.temperature == 7.0
    assert "model=    fills `scorer_model`" in lib.describe("s")


def test_slot_needs_a_positive_sign(lib):
    write(lib, "e", "type: Encoder\n")                  # name: str, dim: int → scalars
    with pytest.raises(ConfigError, match="finds no slot"):
        lib.load("e", model="x")


def test_slot_and_real_parameter_together_raise(lib):
    write(lib, "s", "type: Scorer\nscorer_model: {type: Model}\n", kind="scorers")
    with pytest.raises(ConfigError, match="name the same slot"):
        lib.load("s", model=Model(), scorer_model=Model())


# ── provenance and git ────────────────────────────────────────────────────────

def test_provenance_has_no_absolute_path(lib):
    write(lib, "m", "type: Model\n")
    line = lib.load_raw("m").provenance()
    assert str(lib.root) not in line and "cfg:" in line


def test_find_in_git_history(lib):
    root = lib.root.parent
    for command in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"]):
        subprocess.run(command, cwd=root, check=True)
    write(lib, "m", "type: Model\ntemperature: 1.0\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "first"], cwd=root, check=True)
    old_hash = lib.load_raw("m").hash
    write(lib, "m", "type: Model\ntemperature: 2.0\n")
    subprocess.run(["git", "commit", "-qam", "second"], cwd=root, check=True)
    [found] = lib.find(old_hash)
    assert found["name"] == "m" and found["path"].endswith("models/m.yaml")
    assert lib.find("00000000") == []


# ── writing and comparing ─────────────────────────────────────────────────────

def test_create_writes_and_reads_back(lib):
    config_ = lib.load_raw  # noqa: F841 - keep the name space clean
    created = Encoder("ViT-L", dim=1024)
    conf = __import__("useml.config._write", fromlist=["create"]).create(
        lib, "big", created, meta={"display_name": "Big"})
    rebuilt = lib.load("big")
    assert (rebuilt.name, rebuilt.dim) == ("ViT-L", 1024)
    assert conf.display_name == "Big"
    text = (lib.root / "models" / "big.yaml").read_text()
    assert "device" not in text                          # default None: not written


def test_create_does_not_overwrite(lib):
    from useml.config import _write

    _write.create(lib, "e", Encoder())
    with pytest.raises(ConfigError, match="already exists"):
        _write.create(lib, "e", Encoder(dim=3))
    _write.create(lib, "e", Encoder(dim=3), overwrite=True)
    assert lib.load("e", reload=True).dim == 3


def test_create_names_a_reference(lib):
    from useml.config import _write

    write(lib, "enc", "type: Encoder\ndim: 64\n")
    model = Model(encoder=lib.load("enc"))
    _write.create(lib, "m", model)
    assert "ref: enc" in (lib.root / "models" / "m.yaml").read_text()


def test_shared_object_is_written_once(lib):
    from useml.config import _write

    encoder = Encoder(dim=9)
    with pytest.warns(UserWarning, match="left at their default"):
        _write.create(lib, "shared", Shared(encoder=encoder))
    text = (lib.root / "models" / "shared.yaml").read_text()
    assert text.count("type: Encoder") == 1


def test_create_refuses_what_it_cannot_read_back(lib):
    from useml.config import _write

    with pytest.raises(ConfigError, match="cannot be found on the instance"):
        _write.create(lib, "o", Opaque(handle=3))
    assert not (lib.root / "models" / "o.yaml").exists()
    _write.create(lib, "o", Opaque(handle=3), handle=3, verify=False)
    assert lib.load("o")._secret == 3


def test_get_config_reads_the_live_object(lib):
    from useml.config import _write

    write(lib, "m", "type: Model\nmatcher: {type: Matcher}\n")
    model = lib.load("m")
    model.matcher.alpha = 0.9
    live = _write.get_config(lib, model)
    assert live.kwargs["matcher"]["alpha"] == 0.9
    assert _write.diff(lib, "m", model) == {"matcher.alpha": (0.5, 0.9)}


def test_diff_completes_defaults(lib):
    from useml.config import _write

    write(lib, "short", "type: Matcher\n")
    write(lib, "long", "type: Matcher\nalpha: 0.5\nsolver: sinkhorn\nsteps: 10\n")
    assert _write.diff(lib, "short", "long") == {}
    write(lib, "other", "type: Matcher\nsteps: 11\n")
    assert _write.diff(lib, "short", "other", build=False) == {"steps": (10, 11)}


# ── default library and import cost ───────────────────────────────────────────

def test_discovery_from_pyproject(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "configs" / "models").mkdir(parents=True)
    (project / "pkg").mkdir()
    (project / "pkg" / "__init__.py").write_text(
        "__all__ = ['Thing']\nclass Thing:\n    def __init__(self, x: int = 1):\n"
        "        self.x = x\n")
    (project / "configs" / "models" / "t.yaml").write_text("type: Thing\nx: 5\n")
    (project / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.useml.config]
        root = "configs"
        packages = ["pkg"]
        pythonpath = ["."]
    """))
    (project / "notebooks").mkdir()
    monkeypatch.chdir(project / "notebooks")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "pkg", raising=False)
    library = config.use()
    assert library.root == (project / "configs").resolve()
    assert config.load("t").x == 5
    assert config.ls() == {"models": ["t"]}


def test_importing_useml_config_does_not_import_heavy_things():
    code = ("import sys, useml.config as c; "
            "print(any(m in sys.modules for m in ('torch', 'numpy', 'yaml')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"USEML_NO_AUTO": "1", "PATH": ""}, check=True).stdout.strip()
    assert out == "False"


def test_cli_config(lib, monkeypatch, capsys):
    from useml.cli import main

    write(lib, "m", "type: Model\nmatcher: {type: Matcher}\n")
    monkeypatch.setattr(config, "_DEFAULT", lib)
    assert main(["config", "ls"]) == 0
    assert main(["config", "show", "m", "matcher.alpha=0.1"]) == 0
    out = capsys.readouterr().out
    assert "m+ovr" in out and "alpha: 0.1" in out
