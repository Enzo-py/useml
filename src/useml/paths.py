"""Where every cache goes: ONE root per machine, set once, outside any repository.

    # ~/.config/useml/config.toml   (or the file named by $USEML_CONFIG)
    root = "/data/me/useml"

    [env]                                   # optional: a cache that lives elsewhere
    HF_HOME = "/data/me/hf_cache"

    [hub]
    parallel = false                        # sequential downloads (the default)

`$USEML_ROOT` overrides `root`. Without a root (a laptop), library caches are left where
they are and useml's own files go to `~/.cache/useml`.

LAYOUT under the root:

    <root>/hub/huggingface     HF_HOME          weights and datasets of the Hub
    <root>/hub/torch           TORCH_HOME       torch.hub weights
    <root>/hub/nltk_data       NLTK_DATA
    <root>/hub/files           what `useml.hub.get` fetches over HTTP / Google Drive
    <root>/cache/<name>        pip, matplotlib, triton, torchinductor, CUDA, vLLM, XDG
    <root>/tmp                 TMPDIR
    <root>/logs                network guard and monitor logs

WHAT `apply()` NEVER DOES: move a cache that is already chosen. A variable already set to
a folder OUTSIDE the home and /tmp is kept (an `HF_HOME` on a big disk holding tens of GB
of weights: moving it would download everything again). The goal is that no cache writes
into the home or /tmp, not to relocate what is already elsewhere. `useml hub scan` lists
the caches left in the home and moves them on request.

THREE LEAKS TOWARDS THE HOME, closed here:

  • VARIABLES FOR ONE PART of the Hugging Face cache (`HF_HUB_CACHE`, `TRANSFORMERS_CACHE`,
    `HF_XET_CACHE`, …) win over `HF_HOME`. Set to an EMPTY or missing folder under the
    home or /tmp, such a variable follows `HF_HOME`. Set to an ALREADY POPULATED cache, it
    is KEPT wherever it is, with a warning: moving it would download everything again.
  • `huggingface_hub` IMPORTED BEFORE useml (a notebook starting with `import torch,
    open_clip`) has read `HF_HOME` at its import. Its downloads re-read
    `constants.HF_HUB_CACHE` at every call, so these constants are REALIGNED on the root.
  • The login token (`hf auth login`) written under the former `HF_HOME` stays readable:
    `HF_TOKEN_PATH` points to it while the new `HF_HOME` has none.

Diagnosis: `useml paths` (or `python -m useml paths`).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT_VARIABLE = "USEML_ROOT"
CONFIG_VARIABLE = "USEML_CONFIG"

#: environment variable → its folder under the root
CACHE_VARIABLES = {
    "HF_HOME": "hub/huggingface",
    "TORCH_HOME": "hub/torch",
    "NLTK_DATA": "hub/nltk_data",
    "XDG_CACHE_HOME": "cache/xdg",
    "PIP_CACHE_DIR": "cache/pip",
    "UV_CACHE_DIR": "cache/uv",
    "MPLCONFIGDIR": "cache/matplotlib",
    "TRITON_CACHE_DIR": "cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "CUDA_CACHE_PATH": "cache/nv_compute",
    "VLLM_CACHE_ROOT": "cache/vllm",
    "TMPDIR": "tmp",
}

#: Hugging Face variables that, when set, replace `HF_HOME` for one part of the cache →
#: their folder under `HF_HOME`
HUGGING_FACE_PART_VARIABLES = {
    "HF_HUB_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "TRANSFORMERS_CACHE": "hub",
    "SENTENCE_TRANSFORMERS_HOME": "hub",
    "HF_XET_CACHE": "xet",
    "HF_ASSETS_CACHE": "assets",
    "HUGGINGFACE_ASSETS_CACHE": "assets",
    "HF_MODULES_CACHE": "modules",
    "HF_DATASETS_CACHE": "datasets",
}

_applied_root: Path | None = None
#: shell values before `apply`, for the diagnosis
_original_values: dict[str, str | None] = {}
_kept_variables: dict[str, str] = {}


# ── the machine configuration ─────────────────────────────────────────────────

def config_path() -> Path:
    """`$USEML_CONFIG`, else `$XDG_CONFIG_HOME/useml/config.toml`, else
    `~/.config/useml/config.toml`."""
    explicit = os.environ.get(CONFIG_VARIABLE, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    return Path(base).expanduser() / "useml" / "config.toml"


def config() -> dict:
    """The parsed machine configuration; {} if the file does not exist. Read at each call:
    a file written after import counts."""
    path = config_path()
    if not path.is_file():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                       # pragma: no cover - depends on Python
        import tomli as tomllib
    with open(path, "rb") as file:
        try:
            return tomllib.load(file)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"useml: cannot read {path}: {error}") from error


def root() -> Path | None:
    """The root of every cache: `$USEML_ROOT`, else `root` in the config file, else None."""
    explicit = os.environ.get(ROOT_VARIABLE, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    value = config().get("root")
    return Path(os.path.expandvars(str(value))).expanduser() if value else None


def home() -> Path:
    """Where useml keeps its OWN files: the root, or `~/.cache/useml` without one."""
    configured = root()
    return configured if configured is not None else Path.home() / ".cache" / "useml"


def directory(*parts: str, create: bool = True) -> Path:
    """`home()/<parts>`, created by default."""
    path = home().joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configured_choices() -> dict[str, str]:
    """The `[env]` table of the config file: caches the machine places elsewhere."""
    table = config().get("env", {}) or {}
    return {str(name): os.path.expandvars(str(value)) for name, value in table.items()}


# ── applying it ───────────────────────────────────────────────────────────────

def is_under_home_or_temporary(value) -> bool:
    path = Path(value).expanduser().resolve()
    protected = [Path.home().resolve(), Path("/tmp").resolve(), Path("/var/tmp").resolve()]
    return any(path == folder or folder in path.parents for folder in protected)


def apply() -> Path | None:
    """Point every cache variable at the root (idempotent). Returns the root, or None."""
    global _applied_root
    configured = root()
    if configured is None or configured == _applied_root:
        return configured
    for variable in (*CACHE_VARIABLES, *HUGGING_FACE_PART_VARIABLES):
        _original_values.setdefault(variable, os.environ.get(variable))
    choices = configured_choices()
    for variable, sub_directory in CACHE_VARIABLES.items():
        current = os.environ.get(variable)
        if current and not is_under_home_or_temporary(current):
            _kept_variables[variable] = current
            continue
        target = Path(choices[variable]).expanduser() if variable in choices else (
            configured / sub_directory)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    for variable, value in choices.items():        # extra variables of the machine
        if variable not in CACHE_VARIABLES:
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
            os.environ[variable] = value
    tempfile.tempdir = None                        # re-read TMPDIR at the next call
    _redirect_hugging_face_parts()
    _keep_login_token()
    _applied_root = configured
    _realign_if_read_too_early()
    return configured


def _is_populated(value) -> bool:
    """The folder exists and already holds something (downloaded weights)."""
    path = Path(value).expanduser()
    try:
        return path.is_dir() and any(
            entry.name not in {".locks", "version.txt"} for entry in path.iterdir())
    except OSError:
        return False


def _redirect_hugging_face_parts() -> None:
    """A variable for one part of the HF cache follows `HF_HOME` only if it points under
    the home or /tmp AND to an empty or missing folder."""
    hugging_face_home = Path(os.environ["HF_HOME"])
    for variable, sub_directory in HUGGING_FACE_PART_VARIABLES.items():
        current = os.environ.get(variable)
        if not current:
            continue                      # not set: derived from HF_HOME, already right
        if not is_under_home_or_temporary(current):
            _kept_variables[variable] = current
            continue
        if _is_populated(current):
            _kept_variables[variable] = current
            print(f"⚠ useml: {variable}={current} is under the home but already holds a "
                  f"cache: kept there, to avoid downloading everything again. "
                  f"`useml hub scan --move` can move it.", file=sys.stderr)
            continue
        os.environ[variable] = str(hugging_face_home / sub_directory)


def _keep_login_token() -> None:
    if os.environ.get("HF_TOKEN_PATH") or os.environ.get("HF_TOKEN"):
        return
    if (Path(os.environ["HF_HOME"]) / "token").exists():
        return
    former = Path.home() / ".cache" / "huggingface" / "token"
    if former.exists():
        os.environ["HF_TOKEN_PATH"] = str(former)


def _realign_if_read_too_early() -> None:
    """`huggingface_hub` imported BEFORE useml keeps the paths read at its import. Its
    downloads re-read `constants.HF_HUB_CACHE` at every call: these constants are fixed.
    Only `HF_MODULES_CACHE` (remote code, `trust_remote_code`) is frozen in `sys.path` at
    its first use; that one needs a fresh kernel, and we say so."""
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is None:
        return
    hugging_face_home = os.environ["HF_HOME"]
    if Path(constants.HF_HOME).expanduser().resolve() == Path(hugging_face_home).resolve():
        return
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or \
        os.path.join(hugging_face_home, "hub")
    assets = os.environ.get("HF_ASSETS_CACHE") or os.path.join(hugging_face_home, "assets")
    xet = os.environ.get("HF_XET_CACHE") or os.path.join(hugging_face_home, "xet")
    paths = {
        "HF_HOME": hugging_face_home,
        "default_cache_path": os.path.join(hugging_face_home, "hub"),
        "default_assets_cache_path": os.path.join(hugging_face_home, "assets"),
        "default_xet_cache_path": os.path.join(hugging_face_home, "xet"),
        "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub,
        "HF_ASSETS_CACHE": assets,
        "HUGGINGFACE_ASSETS_CACHE": assets,
        "HF_XET_CACHE": xet,
    }
    if os.environ.get("HF_TOKEN_PATH"):
        paths["HF_TOKEN_PATH"] = os.environ["HF_TOKEN_PATH"]
    for name, value in paths.items():
        if hasattr(constants, name):
            setattr(constants, name, value)
    modules = os.environ.get("HF_MODULES_CACHE") or os.path.join(hugging_face_home, "modules")
    frozen = []
    for module_name in ("transformers.utils.hub", "transformers.dynamic_module_utils",
                        "diffusers.utils.constants", "diffusers.utils.dynamic_modules_utils"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "HF_MODULES_CACHE"):
            if getattr(module, "HF_MODULES_CACHE") in sys.path:
                frozen.append(module_name)
            setattr(module, "HF_MODULES_CACHE", modules)
    print(f"useml: huggingface_hub was imported before useml; its cache paths are "
          f"realigned on {hugging_face_home}.", file=sys.stderr)
    if frozen:
        print("⚠ useml: remote code (trust_remote_code) was already loaded from the former "
              "HF_HOME; restart the kernel for it to move too.", file=sys.stderr)


# ── diagnosis ─────────────────────────────────────────────────────────────────

def hub_cache_in_use() -> str:
    """The hub folder Hugging Face uses (the one holding `models--*`)."""
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        return constants.HF_HUB_CACHE
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            return os.environ[variable]
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache"), "huggingface")
    return os.path.join(hf_home, "hub")


def _repository_count(folder) -> int:
    try:
        return sum(1 for entry in Path(folder).expanduser().iterdir()
                   if entry.name.startswith(("models--", "datasets--")))
    except OSError:
        return 0


def describe() -> str:
    """One line per variable: where it points, and whether it was kept as it was."""
    configured = root()
    if configured is None:
        return (f"no root (set ${ROOT_VARIABLE} or `root = \"…\"` in {config_path()}); "
                f"useml's own files go to {home()}")
    lines = [f"root: {configured}   (config: {config_path()})"]
    for variable in CACHE_VARIABLES:
        kept = "  (kept: already outside home and /tmp)" if variable in _kept_variables else ""
        lines.append(f"  {variable:<24} {os.environ.get(variable, '—')}{kept}")
    return "\n".join(lines)


def diagnose() -> str:
    """Where Hugging Face weights go, and where the downloaded ones are."""
    configured = apply()
    lines = [describe(), "", "shell value → value now:"]
    for variable in ("HF_HOME", *HUGGING_FACE_PART_VARIABLES, "XDG_CACHE_HOME"):
        before = _original_values.get(variable, os.environ.get(variable))
        lines.append(f"  {variable:<28} {before or '—'}  →  {os.environ.get(variable) or '—'}")
    candidates = {
        "IN USE": hub_cache_in_use(),
        "~/.cache/huggingface/hub": str(Path.home() / ".cache" / "huggingface" / "hub"),
    }
    if configured is not None:
        candidates["<root>/hub/huggingface/hub"] = str(configured / "hub" / "huggingface" / "hub")
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = _original_values.get(variable)
        if value:
            candidates[f"shell {variable}"] = (
                os.path.join(value, "hub") if variable == "HF_HOME" else value)
    lines += ["", "models / datasets already downloaded:"]
    for label, folder in candidates.items():
        lines.append(f"  {label:<28} {_repository_count(folder):>5}   {folder}")
    return "\n".join(lines)
