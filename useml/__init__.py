"""useml — minimalist ML experiment framework."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("useml")
except PackageNotFoundError:
    __version__ = "dev-local"

from .session.manager import _session
from .vault.project import Project
from .errors import ProjectAlreadyExistsError, ModelInstantiationError, UseMlError, NotConnectedError
from .template import Config, Loss, Model
from .dataset import DataBundle

from . import workdir


def init(vault_path: str = "vault") -> None:
    """Connect to a vault directory.

    Args:
        vault_path: Root directory where projects are stored.
    """
    _session.connect(vault_path)


def _require_vault(fn_name: str) -> None:
    if _session.vault is None:
        raise NotConnectedError(
            f"useml.{fn_name}() called before useml.init(). "
            f"Run useml.init('path/to/vault') first."
        )


def new(project_name: str, force: bool = False) -> Project:
    """Create a new project in the vault and return it.

    Args:
        project_name: Name for the new project.
        force: Discard any unsaved in-RAM state without raising.

    Returns:
        The newly created :class:`~useml.vault.project.Project` instance —
        the entry point for ``project.runs``, ``project.models``, etc.

    Raises:
        NotConnectedError: If :func:`init` has not been called.
        ProjectAlreadyExistsError: If a project with that name already exists.
    """
    _require_vault("new")
    if _session.vault.exists(project_name):
        raise ProjectAlreadyExistsError(
            f"Project '{project_name}' already exists. "
            f"Use useml.focus('{project_name}') to open it."
        )
    _session.set_focus(project_name, force=force)
    return _session._project


def focus(project_name: str, force: bool = False) -> Project:
    """Open an existing project and return it.

    Args:
        project_name: Name of the project to open.
        force: Discard any unsaved in-RAM state without raising.

    Returns:
        The focused :class:`~useml.vault.project.Project` instance.

    Raises:
        NotConnectedError: If :func:`init` has not been called.
    """
    _require_vault("focus")
    _session.set_focus(project_name, force=force)
    return _session._project


__all__ = [
    # Setup
    "init",
    "new",
    "focus",
    # Classes
    "Config",
    "Model",
    "Loss",
    "DataBundle",
    # Errors
    "ProjectAlreadyExistsError",
    "ModelInstantiationError",
    "NotConnectedError",
    "UseMlError",
]
