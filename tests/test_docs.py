"""The documentation must not lie: every command it shows parses, every guide it links
exists, the quickstart example runs, and it ships inside the package."""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from useml import cli

DOCS = Path(cli.__file__).resolve().parent / "docs"
GUIDES = sorted(DOCS.glob("*.md"))


def _code_blocks(text: str) -> list[str]:
    return re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL)


def _commands() -> list[tuple[str, str]]:
    found = []
    for guide in [*GUIDES, DOCS.parent.parent.parent / "README.md"]:
        if not guide.exists():
            continue
        for block in _code_blocks(guide.read_text()):
            for line in block.splitlines():
                line = line.split("  #")[0].strip()
                if line.startswith("useml "):
                    found.append((guide.name, line))
    return found


def _concrete(line: str) -> list[str]:
    """A documented command line → arguments argparse can check."""
    line = re.sub(r"<[^>]+>", "x", line)                 # <name> → x
    line = line.replace("[--move [--apply]]", "--move --apply")
    line = re.sub(r"\[([^\[\]]*)\]", "", line)            # drop optional [...]
    line = line.replace("--data auto|rows|none", "--data auto")
    args = shlex.split(line)[1:]
    if args[-1:] == ["--"] or "--" in args:
        args = args[:args.index("--")] + ["--", "true"]
    return args


@pytest.mark.parametrize("guide,line", _commands())
def test_every_documented_command_parses(guide, line):
    parser = cli.build_parser()
    try:
        parser.parse_args(_concrete(line))
    except SystemExit as error:
        pytest.fail(f"{guide}: `{line}` does not parse ({error})")


def test_every_relative_link_exists():
    for guide in GUIDES:
        for target in re.findall(r"\]\(([^)#]+\.md)\)", guide.read_text()):
            assert (DOCS / target).exists(), f"{guide.name} links to missing {target}"


def test_skill_has_frontmatter():
    text = (DOCS / "SKILL.md").read_text()
    assert text.startswith("---\nname: useml\ndescription: ")


def test_docs_command(capsys):
    assert cli.main(["docs"]) == 0
    listing = capsys.readouterr().out
    assert all(guide.stem in listing for guide in GUIDES)
    assert cli.main(["docs", "troubleshooting"]) == 0
    assert capsys.readouterr().out.startswith("# Troubleshooting")
    assert cli.main(["docs", "nope"]) == 1


def test_quickstart_example_runs(tmp_path, monkeypatch):
    """The experiment of the quickstart, copied from the guide, runs as written."""
    text = (DOCS / "quickstart.md").read_text()
    blocks = [b for b in _code_blocks(text) if "def make_numbers" in b or "exp.experience" in b]
    module, notebook = blocks
    package = tmp_path / "my_project"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "steps.py").write_text(module)
    script = notebook.replace('exp.get("demo/first/summary")',
                              'print(exp.get("demo/first/summary"))')
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(tmp_path),
           "USEML_EXP_ROOT": str(tmp_path / "experiences"), "USEML_NO_TMUX": "1",
           "USEML_CONFIG": str(tmp_path / "none.toml")}
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "{'label': 'first try', 'mean': 499.5, 'count': 1000}" in out.stdout
