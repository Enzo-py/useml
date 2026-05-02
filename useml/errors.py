"""useml error catalog.

Every exception raised by useml is defined here with a unique code,
a human-readable cause, and an actionable fix. Import the exception
classes you need; do not instantiate _ErrorDef directly.

Codes
-----
UML-1xx  Session (vault connection, project focus, dirty state)
UML-2xx  Vault / Project (creation, identity)
UML-3xx  Snapshot (overwrite, missing, weights)
UML-4xx  Import / workdir (nothing mounted, module not found)
UML-5xx  Training (loss, optimizer)
UML-6xx  Dataset (missing package, unknown name, wrong type)
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Catalog definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ErrorDef:
    code: str
    message: str
    cause: str
    fix: str


_CATALOG: dict[str, _ErrorDef] = {
    # Session
    "UML-101": _ErrorDef(
        code="UML-101",
        message="Not connected to a vault.",
        cause="useml.init() has not been called, so no vault path is known.",
        fix="Call useml.init('path/to/vault') before any vault operation.",
    ),
    "UML-102": _ErrorDef(
        code="UML-102",
        message="No project is currently focused.",
        cause="useml.new() or useml.focus() has not been called.",
        fix="Call useml.new('name') to create a project or useml.focus('name') to resume one.",
    ),
    "UML-103": _ErrorDef(
        code="UML-103",
        message="You have unsaved changes in RAM.",
        cause="Components were tracked but not committed before switching projects.",
        fix="Call useml.commit('message') to save, or useml.stash() to park changes.",
    ),
    # Vault / Project
    "UML-201": _ErrorDef(
        code="UML-201",
        message="A project with this name already exists in the vault.",
        cause="useml.new() was called with a name that matches an existing project directory.",
        fix="Use useml.focus('name') to resume the existing project, or choose a different name.",
    ),
    "UML-202": _ErrorDef(
        code="UML-202",
        message="Cannot compare a Project to an unsupported type.",
        cause="Project.__eq__ received a value that is neither a string nor a Project.",
        fix="Compare projects only to strings (project name) or other Project instances.",
    ),
    # Snapshot
    "UML-301": _ErrorDef(
        code="UML-301",
        message="Snapshot directory is not empty — refusing to overwrite.",
        cause="A snapshot with the same timestamp already exists on disk.",
        fix="This is an internal guard; if it fires, check for clock skew or concurrent writes.",
    ),
    "UML-302": _ErrorDef(
        code="UML-302",
        message="Snapshot not found.",
        cause="The requested snapshot tag or folder name does not exist in the project.",
        fix="Run useml.show() to list available snapshots, then pass a valid tag.",
    ),
    "UML-303": _ErrorDef(
        code="UML-303",
        message="Invalid snapshot tag.",
        cause="The tag did not match any known format (\\latest, \\head~N, or a folder name).",
        fix="Use '\\\\latest', '\\\\head~N' (N ≥ 0), or the exact snapshot folder name.",
    ),
    "UML-304": _ErrorDef(
        code="UML-304",
        message="Weights file not found for the requested component.",
        cause="The snapshot was saved without weights for this component, or the file was deleted.",
        fix="Re-commit the model, or load from a snapshot that contains the component.",
    ),
    "UML-305": _ErrorDef(
        code="UML-305",
        message="Failed to load weights — code/weights mismatch.",
        cause="The model architecture changed after the snapshot was saved.",
        fix="Mount the snapshot (useml.mount()) to import the original class, then call useml.load().",
    ),
    "UML-306": _ErrorDef(
        code="UML-306",
        message="Snapshot has no source/ directory.",
        cause="The snapshot was saved before source archiving was introduced, or source saving failed.",
        fix="Re-commit the model with the current version of useml to capture sources.",
    ),
    "UML-307": _ErrorDef(
        code="UML-307",
        message="Cannot instantiate model class — constructor requires arguments not found in snapshot config.",
        cause="The model __init__ has required positional arguments that are missing from the snapshot config YAML. "
              "This happens when architecture params (latent_dim, n_r_bins, …) were not stored in Config.",
        fix="Add the missing params as custom Config fields before committing: "
            "Config(..., latent_dim=16). useml.load() will then forward them to the constructor automatically.",
    ),
    # Import / workdir
    "UML-401": _ErrorDef(
        code="UML-401",
        message="No snapshot is mounted.",
        cause="A useml.workdir.* import was attempted before calling useml.mount().",
        fix="Call useml.mount('\\\\latest') (or another tag) before importing from useml.workdir.",
    ),
    "UML-402": _ErrorDef(
        code="UML-402",
        message="Module not found in the mounted snapshot.",
        cause="The requested module path does not exist in the snapshot's source/ directory.",
        fix="Check the module name with useml.debug_imports(), then correct the import path.",
    ),
    "UML-403": _ErrorDef(
        code="UML-403",
        message="Cannot import class from the current working directory.",
        cause="The module path stored in the snapshot manifest is no longer importable.",
        fix="Ensure the module is on sys.path, or mount the snapshot with useml.mount().",
    ),
    # Training
    "UML-501": _ErrorDef(
        code="UML-501",
        message="Unknown loss function key.",
        cause="The string passed to Config(loss=...) is not in the built-in loss registry.",
        fix="Use one of the built-in keys ('cross_entropy', 'mse', 'bce', 'l1'), or pass an nn.Module.",
    ),
    "UML-502": _ErrorDef(
        code="UML-502",
        message="Invalid loss type.",
        cause="Config.loss received a value that is not a string, nn.Module subclass, instance, or callable.",
        fix="Pass a string key, an nn.Module subclass, an nn.Module instance, or a callable.",
    ),
    "UML-503": _ErrorDef(
        code="UML-503",
        message="Unknown optimizer key.",
        cause="The string passed to Config(optimizer=...) is not in the built-in optimizer registry.",
        fix="Use one of the built-in keys: 'adam', 'adamw', 'sgd'.",
    ),
    # Dataset
    "UML-601": _ErrorDef(
        code="UML-601",
        message="torchvision is not installed.",
        cause="A built-in dataset was requested but torchvision is missing from the environment.",
        fix="Run: pip install torchvision",
    ),
    "UML-602": _ErrorDef(
        code="UML-602",
        message="The 'datasets' package is not installed.",
        cause="An 'hf:' dataset was requested but the HuggingFace datasets library is missing.",
        fix="Run: pip install datasets",
    ),
    "UML-603": _ErrorDef(
        code="UML-603",
        message="Unknown dataset name.",
        cause="The string passed to train(dataset=...) is not a built-in name and does not start with 'hf:'.",
        fix="Use a built-in name ('mnist', 'cifar10', …), 'hf:<name>', or a torch Dataset instance.",
    ),
    "UML-604": _ErrorDef(
        code="UML-604",
        message="Invalid dataset type.",
        cause="dataset must be a string or a torch.utils.data.Dataset, but received something else.",
        fix="Pass a string name or a torch Dataset instance to train() / load_dataset().",
    ),
}


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class UseMlError(Exception):
    """Base class for all useml exceptions.

    Subclasses set ``code`` to look up cause and fix from the catalog.
    The formatted message always starts with ``[UML-NNN]`` so it can be
    grepped against ERRORS.md.
    """

    code: str = "UML-000"

    def __init__(self, message: str = "") -> None:
        entry = _CATALOG.get(self.code)
        if entry:
            body = message or entry.message
            full = (
                f"[{self.code}] {body}\n"
                f"  Cause : {entry.cause}\n"
                f"  Fix   : {entry.fix}"
            )
        else:
            full = message
        super().__init__(full)
        self.user_message = message


# ---------------------------------------------------------------------------
# Session errors  (UML-1xx)
# ---------------------------------------------------------------------------

class NotConnectedError(UseMlError, RuntimeError):
    """Raised when a vault operation is attempted without calling useml.init()."""
    code = "UML-101"


class NoFocusError(UseMlError, RuntimeError):
    """Raised when a project-scoped operation is attempted with no project focused."""
    code = "UML-102"


class UncommittedChangesError(UseMlError, RuntimeError):
    """Raised when switching projects while tracked components are unsaved."""
    code = "UML-103"


# ---------------------------------------------------------------------------
# Vault / Project errors  (UML-2xx)
# ---------------------------------------------------------------------------

class ProjectAlreadyExistsError(UseMlError, FileExistsError):
    """Raised when useml.new() targets a project name that already exists."""
    code = "UML-201"


class ProjectTypeError(UseMlError, TypeError):
    """Raised when Project.__eq__ receives an incompatible type."""
    code = "UML-202"


# ---------------------------------------------------------------------------
# Snapshot errors  (UML-3xx)
# ---------------------------------------------------------------------------

class SnapshotOverwriteError(UseMlError, FileExistsError):
    """Raised when a snapshot save would overwrite an existing directory."""
    code = "UML-301"


class SnapshotNotFoundError(UseMlError, FileNotFoundError):
    """Raised when a snapshot tag resolves to nothing on disk."""
    code = "UML-302"


class InvalidSnapshotTagError(UseMlError, ValueError):
    """Raised when a snapshot tag string does not match any valid format."""
    code = "UML-303"


class WeightsNotFoundError(UseMlError, FileNotFoundError):
    """Raised when a component's .pth file is missing from the snapshot."""
    code = "UML-304"


class WeightsLoadError(UseMlError, RuntimeError):
    """Raised when state_dict loading fails due to architecture mismatch."""
    code = "UML-305"


class ModelInstantiationError(UseMlError, TypeError):
    """Raised when load() cannot construct the model class from the snapshot config."""
    code = "UML-307"


class NoSourceDirectoryError(UseMlError, FileNotFoundError):
    """Raised when mount() targets a snapshot that has no source/ directory."""
    code = "UML-306"


# ---------------------------------------------------------------------------
# Import / workdir errors  (UML-4xx)
# ---------------------------------------------------------------------------

class NothingMountedError(UseMlError, ImportError):
    """Raised when useml.workdir.* is accessed without a mounted snapshot."""
    code = "UML-401"


class SnapshotModuleNotFoundError(UseMlError, ImportError):
    """Raised when a module path is not found inside the mounted snapshot."""
    code = "UML-402"


class WorkdirImportError(UseMlError, ImportError):
    """Raised when a class cannot be imported from the current working directory."""
    code = "UML-403"


# ---------------------------------------------------------------------------
# Training errors  (UML-5xx)
# ---------------------------------------------------------------------------

class UnknownLossError(UseMlError, ValueError):
    """Raised when Config.loss is a string key not in the built-in registry."""
    code = "UML-501"


class InvalidLossTypeError(UseMlError, TypeError):
    """Raised when Config.loss is not a string, nn.Module, or callable."""
    code = "UML-502"


class UnknownOptimizerError(UseMlError, ValueError):
    """Raised when Config.optimizer is not in the built-in registry."""
    code = "UML-503"


# ---------------------------------------------------------------------------
# Dataset errors  (UML-6xx)
# ---------------------------------------------------------------------------

class TorchvisionNotInstalledError(UseMlError, ImportError):
    """Raised when a built-in dataset requires torchvision but it is missing."""
    code = "UML-601"


class HuggingFaceNotInstalledError(UseMlError, ImportError):
    """Raised when an 'hf:' dataset requires the datasets package but it is missing."""
    code = "UML-602"


class UnknownDatasetError(UseMlError, ValueError):
    """Raised when the dataset string is not a known built-in or 'hf:' prefix."""
    code = "UML-603"


class InvalidDatasetTypeError(UseMlError, TypeError):
    """Raised when the dataset argument is neither a string nor a torch Dataset."""
    code = "UML-604"


# ---------------------------------------------------------------------------
# ERRORS.md generator
# ---------------------------------------------------------------------------

def generate_errors_md() -> str:
    """Returns the full content of ERRORS.md as a string."""
    sections: dict[str, list[_ErrorDef]] = {}
    prefixes = {
        "UML-1": "Session (UML-1xx)",
        "UML-2": "Vault / Project (UML-2xx)",
        "UML-3": "Snapshot (UML-3xx)",
        "UML-4": "Import / Workdir (UML-4xx)",
        "UML-5": "Training (UML-5xx)",
        "UML-6": "Dataset (UML-6xx)",
    }
    for prefix, title in prefixes.items():
        sections[title] = [e for e in _CATALOG.values() if e.code.startswith(prefix)]

    lines = [
        "# useml Error Catalog",
        "",
        "Every error raised by useml has a unique code of the form `UML-NNN`.",
        "Search this file by code to understand the cause and how to fix it.",
        "",
    ]
    for title, entries in sections.items():
        lines += [f"## {title}", ""]
        for e in entries:
            lines += [
                f"### `{e.code}` — {e.message}",
                "",
                f"**Cause** : {e.cause}",
                "",
                f"**Fix**   : {e.fix}",
                "",
            ]
    return "\n".join(lines)
