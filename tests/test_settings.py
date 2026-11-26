"""`useml settings`: edits keep everything else in the file, and refuse what is wrong."""
from __future__ import annotations

import textwrap

import pytest

from useml import _settings as settings, paths

ORIGINAL = textwrap.dedent("""\
    # my machine
    root = "/data/me/useml"   # the big disk

    [monitor]
    # who I am
    machine = "gpu-a"
    share = [
      "status",
      "steps",
    ]

    [hub]
    parallel = false
""")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(ORIGINAL)
    monkeypatch.setenv("USEML_CONFIG", str(path))
    return path


def test_set_replaces_only_its_line(config_file):
    settings.edit("monitor.machine", "gpu-b")
    text = config_file.read_text()
    assert text == ORIGINAL.replace('machine = "gpu-a"', 'machine = "gpu-b"')
    assert config_file.with_name("config.toml.bak").read_text() == ORIGINAL


def test_multiline_array_is_replaced_whole(config_file):
    settings.edit("monitor.share", ["status", "errors"])
    text = config_file.read_text()
    assert 'share = ["status", "errors"]' in text and '"steps",' not in text
    assert "# who I am" in text and "[hub]" in text
    assert paths.config()["monitor"]["share"] == ["status", "errors"]


def test_new_keys_and_tables(config_file):
    settings.edit("monitor.log_lines", 50)
    settings.edit("store.max_size", "450GB")
    settings.edit("env.HF_HOME", "/data/hf")
    config = paths.config()
    assert config["monitor"]["log_lines"] == 50 and config["monitor"]["machine"] == "gpu-a"
    assert config["store"]["max_size"] == "450GB" and config["env"]["HF_HOME"] == "/data/hf"
    text = config_file.read_text()
    assert text.index("log_lines") < text.index("[hub]")          # added inside its table
    assert text.startswith("# my machine\nroot = \"/data/me/useml\"   # the big disk")


def test_top_level_key_and_unset(config_file):
    settings.edit("root", "/other")
    assert paths.config()["root"] == "/other"
    settings.edit("hub.parallel", remove=True)
    assert "parallel" not in paths.config().get("hub", {})
    settings.edit("hub.parallel", remove=True)                    # absent: no error


def test_a_file_that_does_not_exist_yet(tmp_path, monkeypatch):
    path = tmp_path / "new" / "config.toml"
    monkeypatch.setenv("USEML_CONFIG", str(path))
    settings.edit("monitor.machine", "laptop")
    assert paths.config() == {"monitor": {"machine": "laptop"}}


@pytest.mark.parametrize("key,text,message", [
    ("monitor.sahre", "x", "Did you mean monitor.share"),
    ("monitor.log_lines", "ten", "whole number"),
    ("hub.parallel", "maybe", "true or false"),
    ("monitor.share", '["status", "code"]', "unknown groups"),
    ("monitor.share", '"status"', "list of strings"),
    ("store.max_size", "lots", "cannot read the size"),
    ("env.9BAD", "x", "unknown setting"),
])
def test_wrong_values_are_refused(config_file, key, text, message):
    with pytest.raises(ValueError, match=message):
        settings.parse_value(key, text)
    assert config_file.read_text() == ORIGINAL


def test_values_are_toml_or_bare_strings():
    assert settings.parse_value("monitor.machine", "gpu-a") == "gpu-a"
    assert settings.parse_value("monitor.machine", '"with spaces"') == "with spaces"
    assert settings.parse_value("monitor.heartbeat", "30") == 30
    assert settings.parse_value("monitor.redact", '["a", "b"]') == ["a", "b"]
    assert settings.format_value('say "hi"') == '"say \\"hi\\""'


def test_cli(config_file, capsys):
    from useml.cli import main

    assert main(["settings", "set", "monitor.log_lines", "40"]) == 0
    assert main(["settings", "get", "monitor.log_lines"]) == 0
    assert main(["settings", "show"]) == 0
    assert main(["settings", "set", "monitor.log_lines", "x"]) == 2
    out = capsys.readouterr().out
    assert "monitor.log_lines = 40" in out and "40" in out and "monitor.share" in out


def test_a_label_with_an_apostrophe(config_file):
    # What the dashboard composes for a name like this: useml settings set monitor.label 'Bob'\''s laptop'
    settings.edit("monitor.label", settings.parse_value("monitor.label", "Bob's laptop"))
    assert paths.config()["monitor"]["label"] == "Bob's laptop"
