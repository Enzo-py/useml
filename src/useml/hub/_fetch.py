"""`get(uri)` — the one function that downloads. It returns a local path.

    get("hf://openai/clip-vit-large-patch14")                  # a whole model repo
    get("hf://openai/clip-vit-large-patch14", allow=["*.json", "*.safetensors"])
    get("hf://openai/clip-vit-large-patch14@main/config.json")  # one file, one revision
    get("hf://datasets/ylecun/mnist")                      # a dataset repo
    get("https://example.org/annotations.zip", extract=True)  # a file, then its content
    get("gdrive://1AbC…")                                       # a Google Drive file
    get("gdrive-folder://1AbC…")                                # a Google Drive folder

The `hf://` form follows Hugging Face's own: `hf://[datasets/|spaces/]<org>/<name>[@rev]
[/<path in repo>]`.

LOCAL FIRST. A URI already fetched is answered from disk without touching the network: the
registry for HTTP and Drive, `local_files_only=True` for Hugging Face. `refresh=True`
asks again.

WHERE. Hugging Face files stay in the Hugging Face cache (`HF_HOME`, under the root once
`useml.paths.apply()` ran): transformers, diffusers and open_clip then find them without
any change. Everything else goes to `<home>/hub/files/`, or to `dest` when given.

INTERRUPTED. An HTTP file is written to `<name>.part`, resumed with a Range request on the
next call, then renamed: a half file is never returned as a whole one.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ._registry import Registry

_REGISTRY: Registry | None = None


def registry() -> Registry:
    """The registry of this machine's home (re-opened if the home changed)."""
    global _REGISTRY
    path = paths.home() / "hub" / "registry.sqlite"
    if _REGISTRY is None or _REGISTRY.path != path:
        _REGISTRY = Registry(path)
    return _REGISTRY


def files_directory() -> Path:
    return paths.directory("hub", "files")


# ── parsing ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HubURI:
    repo_type: str          # "model", "dataset" or "space"
    repo_id: str
    revision: str | None
    filename: str | None    # None: the whole repository


def parse_hf(uri: str) -> HubURI:
    rest = uri[len("hf://"):].strip("/")
    parts = rest.split("/")
    repo_type = "model"
    if parts and parts[0] in ("datasets", "spaces", "models"):
        repo_type = {"datasets": "dataset", "spaces": "space", "models": "model"}[parts[0]]
        parts = parts[1:]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{uri!r}: expected hf://[datasets/]<org>/<name>[@rev][/<file>]")
    name, _, revision = parts[1].partition("@")
    filename = "/".join(parts[2:]) or None
    return HubURI(repo_type, f"{parts[0]}/{name}", revision or None, filename)


# ── errors worth explaining ───────────────────────────────────────────────────

@contextlib.contextmanager
def hub_errors_explained(what: str):
    """Re-raise a Hugging Face 401 as what it is. A stored token is sent with EVERY
    request, so an EXPIRED token turns a public repo into a 401, which `transformers`
    reports as "not a valid model identifier": a typo hunt for a name that is fine."""
    try:
        yield
    except OSError as error:
        chain, cause = [str(error)], error
        for _ in range(8):
            cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
            if cause is None:
                break
            chain.append(str(cause))
        text = "\n".join(chain)
        if "401" not in text and "Unauthorized" not in text:
            raise
        expired = "expired" in text.lower()
        raise OSError(
            f"{what}: the Hugging Face Hub answered 401 Unauthorized"
            + (" because the stored token has EXPIRED.\n" if expired else ".\n")
            + "This is an authentication failure, not a bad repository id. A stored token "
              "is sent with every request, so an expired one fails even on a public "
              "repository.\nFix it either way:\n"
              "  hf auth login          # refresh the token, or\n"
              "  hf auth logout         # drop it: a public repository needs none\n"
              "and check HF_TOKEN in the environment, which overrides the stored one."
        ) from error


# ── the backends ──────────────────────────────────────────────────────────────

def _get_hf(uri: str, *, allow=None, ignore=None, refresh: bool, dest) -> Path:
    try:
        import huggingface_hub
    except ImportError as error:
        raise ImportError("hf:// needs huggingface_hub: pip install 'useml[hf]'") from error
    target = parse_hf(uri)
    common = {"repo_id": target.repo_id, "repo_type": target.repo_type,
              "revision": target.revision}
    if dest is not None:
        common["local_dir"] = str(dest)

    def fetch(local_only: bool) -> str:
        if target.filename is not None:
            return huggingface_hub.hf_hub_download(
                filename=target.filename, local_files_only=local_only, **common)
        return huggingface_hub.snapshot_download(
            allow_patterns=allow, ignore_patterns=ignore, local_files_only=local_only,
            **common)

    if not refresh:
        try:
            return Path(fetch(True))
        except Exception:
            pass                                 # not (completely) on disk: go online
    with hub_errors_explained(uri):
        return Path(fetch(False))


def _http_destination(url: str) -> Path:
    name = Path(urllib.parse.urlparse(url).path).name or "download"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return files_directory() / "http" / f"{digest}-{name}"


def _get_http(url: str, destination: Path, *, chunk: int = 1 << 20) -> Path:
    """Stream to `<destination>.part`, resuming it with a Range request, then rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "useml"})
    if done:
        request.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = done and getattr(response, "status", 200) == 206
        total = response.headers.get("Content-Length")
        total = int(total) + (done if resumed else 0) if total else None
        with open(partial, "ab" if resumed else "wb") as file:
            written = done if resumed else 0
            bar = _progress(total, written, destination.name)
            while True:
                block = response.read(chunk)
                if not block:
                    break
                file.write(block)
                written += len(block)
                bar(written)
    if total is not None and partial.stat().st_size != total:
        raise OSError(f"{url}: received {partial.stat().st_size} bytes of {total}; the "
                      f"partial file is kept and the next call resumes it")
    os.replace(partial, destination)
    return destination


def _progress(total, start, name):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return lambda _written: None
    bar = tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=name,
               file=sys.stderr, leave=False)
    state = {"last": start}

    def update(written):
        bar.update(written - state["last"])
        state["last"] = written
        if total is not None and written >= total:
            bar.close()
    return update


def _get_gdrive(identifier: str, *, folder: bool, destination: Path) -> Path:
    try:
        import gdown
    except ImportError as error:
        raise ImportError("gdrive:// needs gdown: pip install 'useml[gdrive]'") from error
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if folder:
            import inspect

            options = {"id": identifier, "output": str(destination), "quiet": False}
            if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
                options["remaining_ok"] = True
            gdown.download_folder(**options)
            return destination
        written = gdown.download(id=identifier, output=str(destination) + os.sep,
                                 quiet=False)
    except Exception as error:
        raise RuntimeError(
            f"Google Drive download of {identifier} failed: {error}\n"
            "Drive rate-limits automated downloads after repeated access. Wait ~24 h, or "
            f"download it by hand and place it under {destination}, then call "
            f"useml.hub.adopt(<uri>, <path>).") from error
    if written is None:
        raise RuntimeError(f"Google Drive refused {identifier} (rate limit or permissions)")
    return Path(written)


# ── archives ──────────────────────────────────────────────────────────────────

def extract(archive, destination=None) -> Path:
    """Unpack a .zip / .tar(.gz|.bz2|.xz) into `destination` (default `<archive>.d`), once:
    a `.extracted` marker makes the next call free. No member may leave the destination."""
    archive = Path(archive)
    destination = Path(destination) if destination else archive.with_name(archive.name + ".d")
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    staging = destination.with_name(destination.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    root = staging.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if root not in (root / member).resolve().parents and \
                        (root / member).resolve() != root:
                    raise ValueError(f"{archive}: member {member!r} escapes the destination")
            bundle.extractall(staging)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(staging, filter="data")
            else:                                          # pragma: no cover - old Python
                for member in bundle.getmembers():
                    if root not in (root / member.name).resolve().parents:
                        raise ValueError(f"{archive}: member {member.name!r} escapes")
                bundle.extractall(staging)
    else:
        shutil.rmtree(staging)
        raise ValueError(f"{archive}: not a zip or tar archive")
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(staging, destination)
    marker.touch()
    return destination


# ── the entry point ───────────────────────────────────────────────────────────

def get(uri: str, *, dest=None, allow=None, ignore=None, extract_archive: bool = False,
        refresh: bool = False) -> Path:
    """Fetch `uri` (see the module) and return its local path.

    `dest` — where to put it instead of the default place. `allow` / `ignore` — glob
    patterns of the files of a Hugging Face repository to fetch / skip. `extract_archive`
    — unpack a downloaded archive and return the folder. `refresh` — ignore what is on disk.
    """
    key = uri if dest is None else f"{uri} -> {Path(dest).resolve()}"
    if allow or ignore:
        key += f" allow={sorted(allow or [])} ignore={sorted(ignore or [])}"
    if extract_archive:
        key += " extracted"
    book = registry()
    if not refresh and not uri.startswith("hf://"):
        known = book.lookup(key)
        if known is not None:
            return known

    scheme = urllib.parse.urlparse(uri).scheme
    if uri.startswith("hf://"):
        kind, path = "hf", _get_hf(uri, allow=allow, ignore=ignore, refresh=refresh,
                                   dest=dest)
    elif scheme in ("http", "https"):
        destination = Path(dest) if dest is not None else _http_destination(uri)
        if destination.exists() and destination.is_dir():
            destination = destination / (Path(urllib.parse.urlparse(uri).path).name
                                         or "download")
        kind, path = "http", _get_http(uri, destination)
    elif scheme in ("gdrive", "gdrive-folder"):
        identifier = uri.split("://", 1)[1].strip("/")
        destination = Path(dest) if dest is not None else (
            files_directory() / "gdrive" / identifier)
        kind = "gdrive"
        path = _get_gdrive(identifier, folder=scheme == "gdrive-folder",
                           destination=destination)
    else:
        raise ValueError(f"{uri!r}: unknown scheme; use hf://, http(s)://, gdrive:// or "
                         f"gdrive-folder://")
    if extract_archive:
        path = extract(path)
    book.record(key, kind, path)
    return path


def adopt(uri: str, path) -> Path:
    """Record files obtained by hand (a manual download) as the answer to `uri`."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    registry().record(uri, "manual", path)
    return path


def ls(*, refresh_sizes: bool = False) -> list[dict]:
    """Every recorded download, most recently used first."""
    return registry().entries(refresh_sizes=refresh_sizes)
