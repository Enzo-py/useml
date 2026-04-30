# useml/__init__.py
"""
UseML: A minimalist framework for machine learning experiment versioning.
"""

from typing import Any, Dict, List, Optional, Generator

from .session.manager import _session
from .vault.project import Project, ProjectAlreadyExistsError
from .vault.snapshot import Snapshot
from .imports import NothingMountedError
from .template import Config, Model, Trainer, run_training

from . import workdir  # Enable useml.workdir.* imports


def init(vault_path: str = "vault") -> None:
    """Connects the session to a storage vault directory.
    
    Args:
        vault_path (str): The root directory where projects are stored.
    """
    _session.connect(vault_path)


def projects() -> List[Project]:
    """Lists all available projects in the current vault.
    
    Returns:
        List[Project]: A list of project instances found on disk.
    """
    all_projects = _session.get_projects()
    if not all_projects:
        print("Vault is empty. Use useml.new('name') to start a project.")
    else:
        print("UseML Projects List:")
        for proj in all_projects:
            print(f"  {proj}")
    return all_projects


def focus(project_name: str, force: bool = False) -> None:
    """Sets focus. Use force=True to discard unsaved RAM components."""
    _session.set_focus(project_name, force=force)
    return _session._project


def new(project_name: str, force: bool = False, auto_focus=False) -> None:
    """Creates a new project context. Use force=True to discard unsaved RAM."""
    if not auto_focus and _session.vault.exists(project_name):
        raise ProjectAlreadyExistsError(f"A project with the same name (identifier) is already defined in this vault ({_session.vault}).")
    
    _session.set_focus(project_name, force=force)


def stash() -> None:
    """Stashes current project state in RAM."""
    _session.stash()


def track(
    name: str, 
    model: Any, 
    config: Optional[Dict[str, Any]] = None, 
    optimizer: Optional[Any] = None
) -> None:
    """Registers a model and its optional config/optimizer for the current session.
    
    Args:
        name (str): Unique name for the component.
        model (Any): PyTorch model or similar object.
        config (Dict): Hyperparameters or configuration.
        optimizer (Any): Associated optimizer.
    """
    _session.track(name, model, config, optimizer)


def commit(message: str, **metrics: Any) -> Snapshot:
    """Saves a snapshot of all tracked components to the focused project.
    
    Args:
        message (str): Description of the changes or experiment state.
        **metrics: Quantitative results (e.g., loss=0.1, accuracy=0.95).
        
    Returns:
        Snapshot: The created snapshot instance.
    """
    return _session.commit(message, **metrics)


def show() -> None:
    """Displays a summary of the current session, project, and tracked components."""
    if not _session.vault:
        print("Session Status: Disconnected (Call useml.init() first)")
        return

    print(f"\n--- UseML Dashboard ---")
    if _session._is_dirty: # <--- Ajout du warning attendu par le test
        print("⚠️  WARNING: You have unsaved components in RAM (not committed to disk).")

    print(f"Vault:   {_session.vault.path}")
    
    try:
        proj = _session.project
        print(f"Project: {proj.path.name} ({len(proj)} snapshots)")
        print(f"Tracked: {list(_session.components.keys()) or 'None'}")
        if len(proj) > 0:
            latest = proj[0]
            print(f"Latest:  {latest.metadata['message']} [{latest.path.name}]")
    except RuntimeError:
        print("Project: [No focus - Use useml.focus() or useml.new()]")
    print(f"-----------------------\n")

def load(name: str, _from: str = None):
    """Loads a saved model with its weights.
    
    Args:
        name: Name of the model to load.
        _from: Snapshot to load weights from (default: mounted or \\latest).
    
    Returns:
        PyTorch module with loaded weights.
    """
    return _session.load(name, _from=_from)


def mount(snapshot_tag: str):
    """Mounts a snapshot for useml.workdir.* imports.

    Args:
        snapshot_tag: Snapshot identifier (\\latest, \\head~N, \\workdir, or folder name).
    """
    return _session.mount(snapshot_tag)


def train(
    model_cls,
    dataset,
    config=None,
    vault_path: str = ".useml_vault",
) -> dict:
    """Level-0 entry point: train a model in two lines.

    Parameters
    ----------
    model_cls : class
        Subclass of useml.Model (or any nn.Module).
    dataset : str or torch.utils.data.Dataset
        "mnist", "fashion_mnist", "cifar10", "cifar100",
        "hf:<hf_name>", or a torch.utils.data.Dataset.
    config : Config, optional
        Training configuration. Defaults to Config().
    vault_path : str
        Directory to persist experiment snapshots.

    Returns
    -------
    dict
        {"train_loss": [...], "val_loss": [...]}

    Example
    -------
    >>> import useml
    >>> class Net(useml.Model):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(784, 10)
    ...     def forward(self, x):
    ...         return self.fc(x.view(x.size(0), -1))
    >>> useml.train(Net, "mnist")
    """
    return run_training(model_cls, dataset, config=config, vault_path=vault_path)


def debug_imports() -> None:
    """Prints a debug summary of imports visible in __main__ and available via useml.workdir.*.

    Shows:
    - All import statements found in __main__ (or active notebook cells).
    - All module names importable via 'from useml.workdir.<name> import ...' from the
      currently mounted snapshot.  Raises NothingMountedError if no snapshot is mounted.
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
    "Model",
    "Config",
    "Trainer",
]
