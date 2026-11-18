from __future__ import annotations

import http.server
import io
import os
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from useml import hub
from useml.hub import _fetch, _scan


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files, honouring `Range: bytes=N-` (the resume path)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = 0
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            start = int(header[6:].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def served(tmp_path):
    folder = tmp_path / "www"
    folder.mkdir()

    def handler(*args, **kwargs):
        return _RangeHandler(*args, directory=str(folder), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield folder, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_parse_hf():
    assert _fetch.parse_hf("hf://openai/clip") == _fetch.HubURI("model", "openai/clip", None, None)
    assert _fetch.parse_hf("hf://datasets/a/b@v1/x/y.json") == \
        _fetch.HubURI("dataset", "a/b", "v1", "x/y.json")
    with pytest.raises(ValueError):
        _fetch.parse_hf("hf://gpt2")


def test_http_get_then_local_without_network(served, root):
    folder, base = served
    (folder / "a.txt").write_text("hello")
    path = hub.get(f"{base}/a.txt")
    assert path.read_text() == "hello" and root in path.parents
    (folder / "a.txt").unlink()                     # the server no longer has it
    assert hub.get(f"{base}/a.txt") == path
    [entry] = hub.ls()
    assert entry["uses"] == 2 and entry["size"] == 5


def test_http_resume_from_partial(served, root):
    folder, base = served
    (folder / "big.bin").write_bytes(b"0123456789")
    destination = _fetch._http_destination(f"{base}/big.bin")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_name(destination.name + ".part").write_bytes(b"01234")
    assert hub.get(f"{base}/big.bin").read_bytes() == b"0123456789"


def test_get_with_dest_and_extract(served, tmp_path, root):
    folder, base = served
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("inner/file.txt", "x")
    (folder / "data.zip").write_bytes(buffer.getvalue())
    out = hub.get(f"{base}/data.zip", dest=tmp_path / "data" / "data.zip",
                  extract_archive=True)
    assert (out / "inner" / "file.txt").read_text() == "x"


def test_extract_refuses_escaping_members(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="escapes"):
        hub.extract(archive)
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_tar_once(tmp_path):
    source = tmp_path / "f.txt"
    source.write_text("y")
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="f.txt")
    out = hub.extract(archive)
    assert (out / "f.txt").read_text() == "y" and (out / ".extracted").exists()
    assert hub.extract(archive) == out


def test_unknown_scheme():
    with pytest.raises(ValueError, match="unknown scheme"):
        hub.get("ftp://x/y")


def test_adopt_a_manual_download(tmp_path, root):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "f").write_text("z")
    hub.adopt("gdrive://abc", manual)
    assert hub.get("gdrive://abc") == manual.resolve()


def test_expired_token_is_named(tmp_path):
    inner = RuntimeError('401 Client Error. OAuth token has expired')
    outer = OSError("x is not a valid model identifier")
    outer.__cause__ = inner
    with pytest.raises(OSError, match="EXPIRED"):
        with hub.hub_errors_explained("x"):
            raise outer


def test_other_os_errors_pass_through():
    with pytest.raises(OSError, match="disk full"):
        with hub.hub_errors_explained("x"):
            raise OSError("disk full")


# ── stray caches ──────────────────────────────────────────────────────────────

def test_scan_and_move_merge_without_overwrite(fake_home, root):
    stray = fake_home / ".cache" / "huggingface" / "hub"
    (stray / "models--a--b").mkdir(parents=True)
    (stray / "models--a--b" / "w.bin").write_bytes(b"1" * 10)
    (stray / "shared.txt").write_text("old")
    target = root / "hub" / "huggingface" / "hub"
    target.mkdir(parents=True)
    (target / "shared.txt").write_text("new")

    [found] = [entry for entry in hub.scan() if entry["variable"] == "HF_HOME"]
    assert found["bytes"] == 13 and found["target"] == str(root / "hub" / "huggingface")

    plan = hub.move(fake_home / ".cache" / "huggingface")
    assert not plan["applied"] and (stray / "models--a--b").exists()

    done = hub.move(fake_home / ".cache" / "huggingface", apply=True)
    assert (target / "models--a--b" / "w.bin").exists()
    assert (target / "shared.txt").read_text() == "new"          # never overwritten
    assert (stray / "shared.txt").read_text() == "old"           # left, reported
    assert done["conflicts"] == [str(stray / "shared.txt")]


def test_move_without_root_refuses(fake_home):
    (fake_home / ".cache" / "huggingface").mkdir(parents=True)
    with pytest.raises(ValueError, match="no root"):
        hub.move(fake_home / ".cache" / "huggingface")


def test_cli_paths_and_ls(root, capsys):
    from useml.cli import main

    assert main(["paths"]) == 0
    assert main(["hub", "ls"]) == 0
    assert "nothing downloaded" in capsys.readouterr().out
