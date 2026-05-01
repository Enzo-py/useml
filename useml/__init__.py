"""useml — minimalist framework for machine learning experiment versioning."""

from typing import Any, List, Optional

from .session.manager import _session
from .vault.project import Project, ProjectAlreadyExistsError
from .vault.snapshot import Snapshot
from .imports import NothingMountedError
from .template import Config, Loss, Model, Trainer, run_training

from . import workdir


def init(vault_path: str = "vault") -> None:
    """Connects the session to a storage vault directory.

    Args:
        vault_path: Root directory where projects are stored.
    """
    _session.connect(vault_path)


def projects() -> List[Project]:
    """Lists all projects available in the current vault.

    Returns:
        All Project instances found on disk.
    """
    all_projects = _session.get_projects()
    if not all_projects:
        print("Vault is empty. Use useml.new('name') to start a project.")
    else:
        print("UseML Projects List:")
        for proj in all_projects:
            print(f"  {proj}")
    return all_projects


def focus(project_name: str, force: bool = False) -> Project:
    """Sets the active project focus.

    Args:
        project_name: Name of the project to focus on.
        force: When True, discards unsaved components without raising.

    Returns:
        The focused Project instance.
    """
    _session.set_focus(project_name, force=force)
    return _session._project


def new(project_name: str, force: bool = False, auto_focus: bool = False) -> None:
    """Creates and focuses a new project.

    Args:
        project_name: Name for the new project.
        force: When True, discards unsaved components without raising.
        auto_focus: When True, silently focuses an existing project instead
            of raising :exc:`ProjectAlreadyExistsError`.

    Raises:
        ProjectAlreadyExistsError: If a project with the same name already
            exists and *auto_focus* is False.
    """
    if not auto_focus and _session.vault.exists(project_name):
        raise ProjectAlreadyExistsError(
            f"Project '{project_name}' already exists in vault "
            f"({_session.vault})."
        )
    _session.set_focus(project_name, force=force)


def stash() -> None:
    """Pushes the current project state into the RAM stash and resets focus."""
    _session.stash()


def track(
    name: str,
    model: Any,
    config: Optional[Any] = None,
    optimizer: Optional[Any] = None,
) -> None:
    """Registers a model and optional config/optimiser for the current session.

    Args:
        name: Unique name for the component.
        model: PyTorch model or compatible object.
        config: Hyperparameters as a plain dict or a :class:`Config` instance.
            When a Config is provided, the loss source is archived automatically
            on the next :func:`commit`.
        optimizer: Optional associated optimiser.
    """
    _session.track(name, model, config, optimizer)


def commit(message: str, **metrics: Any) -> Snapshot:
    """Saves a snapshot of all tracked components to the focused project.

    Args:
        message: Description of the changes or experiment state.
        **metrics: Quantitative results (e.g. ``loss=0.1``, ``accuracy=0.95``).

    Returns:
        The created Snapshot instance.
    """
    return _session.commit(message, **metrics)


def show() -> None:
    """Prints a summary of the current session, project, and tracked components."""
    if not _session.vault:
        print("Session Status: Disconnected (call useml.init() first)")
        return

    print("\n--- UseML Dashboard ---")
    if _session._is_dirty:
        print("⚠️  WARNING: You have unsaved components in RAM (not committed to disk).")

    print(f"Vault:   {_session.vault.path}")
    try:
        proj = _session.project
        print(f"Project: {proj.path.name} ({len(proj)} snapshots)")
        print(f"Tracked: {list(_session.components.keys()) or 'None'}")
        if len(proj) > 0:
            latest = proj[0]
            print(
                f"Latest:  {latest.metadata['message']} "
                f"[{latest.path.name}]"
            )
    except RuntimeError:
        print("Project: [No focus — use useml.focus() or useml.new()]")
    print("-----------------------\n")


def load(name: str, _from: Optional[str] = None) -> Any:
    """Loads a saved model with its weights from a snapshot.

    Args:
        name: Name of the model component to load.
        _from: Snapshot tag to load weights from. Defaults to the currently
            mounted snapshot, or ``"\\\\latest"`` if none is mounted. Accepts
            the same tags as :func:`mount`.

    Returns:
        PyTorch module with loaded weights.
    """
    return _session.load(name, _from=_from)


def mount(snapshot_tag: str) -> None:
    """Mounts a snapshot for ``useml.workdir.*`` imports.

    Args:
        snapshot_tag: One of:
            - ``"\\\\latest"`` — most recent snapshot
            - ``"\\\\head~N"`` — N commits before latest
            - ``"\\\\current"`` / ``"\\\\workdir"`` — unmount
            - A literal snapshot folder name
    """
    return _session.mount(snapshot_tag)


def train(
    model_cls: Any,
    dataset: Any,
    config: Optional[Config] = None,
) -> dict:
    """Level-0 entry point: trains a model in two lines.

    Snapshots are saved to the currently focused project when a session is
    active (``useml.init()`` + ``useml.new()`` / ``useml.focus()``). Omitting
    those calls trains without saving.

    Args:
        model_cls: Model class — subclass of :class:`Model` or any
            ``nn.Module``.
        dataset: Built-in name (``"mnist"``, ``"fashion_mnist"``,
            ``"cifar10"``, ``"cifar100"``), ``"hf:<hf_name>"`` for
            HuggingFace, or a ``torch.utils.data.Dataset``.
        config: Training configuration. Defaults to :class:`Config` with all
            defaults.

    Returns:
        History dict: ``{"train_loss": [...], "val_loss": [...]}``.

    Example:
        >>> import useml
        >>> useml.init("vault")
        >>> useml.new("my-experiment")
        >>> class Net(useml.Model):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.Linear(784, 10)
        ...     def forward(self, x):
        ...         return self.fc(x.view(x.size(0), -1))
        >>> useml.train(Net, "mnist")
    """
    return run_training(model_cls, dataset, config=config)


def debug_imports() -> None:
    """Prints a debug summary of imports visible in ``__main__`` and via workdir.

    Shows all import statements found in ``__main__`` (or active notebook
    cells) and all module names importable via ``useml.workdir.*`` from the
    currently mounted snapshot.

    Raises:
        NothingMountedError: If no snapshot is mounted when listing workdir
            modules.
    """
    _session.imports.debug()


__all__ = [
    # Core vault API
    "init",
    "projects",
    "focus",
    "new",
    "stash",
    "track",
    "commit",
    "show",
    "load",
    "mount",
    "debug_imports",
    "NothingMountedError",
    # Level-0 template
    "train",
    "Loss",
    "Model",
    "Config",
    "Trainer",
]
