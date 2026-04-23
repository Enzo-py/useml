# useml/__init__.py
"""
UseML: A minimalist framework for machine learning experiment versioning.
"""

from typing import Any, Dict, List, Optional, Generator

from .session.manager import _session
from .vault.project import Project, ProjectAlreadyExistsError
from .vault.snapshot import Snapshot


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


def new(project_name: str, force: bool = False) -> None:
    """Creates a new project context. Use force=True to discard unsaved RAM."""
    if _session.vault.exists(project_name):
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


def resume() -> None:
    """Restores the latest available state into all tracked components."""
    _session.resume()


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
            print(f"Latest:  {latest['message']} [{latest.path.name}]")
    except RuntimeError:
        print("Project: [No focus - Use useml.focus() or useml.new()]")
    print(f"-----------------------\n")

def mount(tag) -> Generator:
    return _session.mount(tag)

__all__ = [
    "init",
    "projects",
    "focus",
    "new",
    "track",
    "commit",
    "resume",
    "show",
]
