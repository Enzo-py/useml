import logging
import sys

from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Generator

from ..vault.core import Vault
from ..vault.project import Project, ProjectState
from .component import Component

logger = logging.getLogger(__name__)

class SessionError(Exception):
    """Base exception for all session-related errors."""
    pass

class UncommittedSessionError(SessionError):
    """Raised when trying to switch context with unsaved components in RAM."""
    pass

class NoSessionFocusError(SessionError):
    """Raised when trying to access current session project when none is focused."""

class Session:
    """Manages the global state of the UseML workspace.

    This manager handles the connection to a storage Vault and maintains
    the focus on a specific Project. It acts as the bridge between functional
    API calls and the underlying storage hierarchy.
    """

    def __init__(self) -> None:
        """Initializes a blank Session."""
        self.vault: Optional[Vault] = None
        self._project: Optional[Project] = None
        self.components: Dict[str, Component] = {}
        self._is_dirty: bool = False
        self._stash: Dict[str, ProjectState] = {}

    def connect(self, vault_path: Union[str, Path]) -> None:
        """Connects the session to a storage vault.

        Args:
            vault_path (Union[str, Path]): Path to the root storage directory.
        """
        self.vault = Vault(Path(vault_path))
        logger.info(f"Connected to vault: {self.vault.path}")

    def get_projects(self) -> List[Project]:
        """Lists all projects available in the connected vault.

        Returns:
            List[Project]: A list of existing project instances.

        Raises:
            RuntimeError: If the session is not connected to a vault.
        """
        if not self.vault:
            raise RuntimeError("Not connected to a vault. Call useml.init('path') first.")
        return self.vault.projects()

    @property
    def project(self) -> Project:
        if self._project is None:
            raise NoSessionFocusError("No project in focus. Call useml.focus('name') or useml.new('name') first.")
        return self._project

    def set_focus(self, project_name: str, force: bool = False) -> None:
        # 1. On vérifie d'abord si c'est le même projet via l'attribut interne _project
        if self._project is not None and self._project == project_name:
            return

        # 2. On regarde si le projet est dans le stash (RAM)
        if project_name in self._stash:
            # Si on a un projet actuel dirty, on bloque avant de restaurer le stash
            if self._is_dirty and not force:
                raise UncommittedSessionError(
                    f"Project '{project_name}' has unsaved changes in RAM. "
                    f"Please use useml.commit() or useml.stash()."
                )
            
            state = self._stash.pop(project_name)
            self._project = state.project
            self.components = state.components
            self._is_dirty = state.is_dirty
            return

        # 3. Sécurité Dirty State pour un nouveau focus (Disk)
        if self._is_dirty and not force:
            raise UncommittedSessionError(
                f"Project '{project_name}' has unsaved changes in RAM. "
                f"Please use useml.commit() or useml.stash()."
            )

        # 4. Chargement classique depuis le Vault
        self._project = self.vault.get_project(project_name)
        self.components = {}
        self._is_dirty = False

    def stash(self) -> None:
        """Puts the current project state into the RAM stash and clears active focus."""
        if self._project is None:
            return
            
        name = self._project.path.name
        self._stash[name] = ProjectState(
            project=self._project,
            components=self.components.copy(),
            is_dirty=self._is_dirty
        )
        
        # Reset total des indicateurs internes
        self._project = None  # Crucial pour test_stash_and_restore_flow
        self.components = {}
        self._is_dirty = False

    def track(
        self,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Registers a component for the current session."""
        self.components[name] = Component(
            name=name, model=model, config=config, optimizer=optimizer
        )
        self._is_dirty = True

    def commit(self, message: str, **metrics: Any) -> Any:
        """Saves a snapshot to the focused project."""
        snap = self.project.commit(
            message=message, components=self.components, **metrics
        )

        self._is_dirty = False
        return snap
    
    def untrack(self, name: str) -> None:
        if name not in self.components:
            return

        # Est-ce que cet élément existait dans le dernier snapshot sur disque ?
        exists_on_disk = False
        if self._project and self._project.snapshots:
            latest = self._project.snapshots[-1]
            exists_on_disk = name in latest.components # Il nous faut cette info dans Snapshot

        del self.components[name]

        if exists_on_disk:
            # On a retiré un truc qui était sauvegardé -> Dirty (il faut valider la suppression)
            self._is_dirty = True
        else:
            # On a retiré un truc jamais sauvegardé -> On recalcule si on reste dirty
            # On n'est dirty que s'il reste d'autres composants "orphelins" ou modifiés
            self._is_dirty = self._check_if_still_dirty_after_removal()

    def _check_if_still_dirty_after_removal(self) -> bool:
        # Logique simplifiée : si on n'a plus rien en RAM, c'est forcément propre
        if not self.components:
            return False
        # Sinon, on garde l'état actuel (car d'autres éléments peuvent être dirty)
        return self._is_dirty

    def resume(self) -> None:
        """Restores tracked components from the latest snapshot in focus."""
        if len(self.project) == 0:
            logger.info(f"Project '{self.project.path.name}' has no snapshots.")
            return

        latest = self.project[0]
        for name, comp in self.components.items():
            try:
                latest.load_component(comp)
            except FileNotFoundError:
                logger.warning(f"Component '{name}' not found in latest snapshot.")

    @contextmanager
    def mount(self, tag: str):
        snapshot_path = self.project.get_snapshot_path(tag)
        source_dir = snapshot_path / "source"
        
        if not source_dir.exists():
            raise FileNotFoundError(f"No source code found for snapshot {tag}")

        original_sys_path = list(sys.path)
        # On capture les modules chargés AVANT le mount
        initial_modules = dict(sys.modules)
        
        local_modules_stems = {p.stem for p in source_dir.glob("*.py")}
        local_modules_stems.update({p.name for p in source_dir.iterdir() if p.is_dir()})

        try:
            # Nettoyage entrée
            for name in list(sys.modules.keys()):
                if name.split('.')[0] in local_modules_stems:
                    del sys.modules[name]
                    
            sys.path.insert(0, str(source_dir))
            yield snapshot_path

        finally:
            # 1. On restaure le path original
            sys.path = original_sys_path
            
            # 2. On identifie ce qui a été chargé pendant le mount
            current_modules = list(sys.modules.keys())
            
            for name in current_modules:
                root_pkg = name.split('.')[0]
                if root_pkg in local_modules_stems:
                    # Si le module n'était pas là avant : on le tue (isolation)
                    if name not in initial_modules:
                        if name in sys.modules:
                            # Optionnel : mod.__dict__.clear() ici pour "poisoning"
                            del sys.modules[name]
                    
                    # Si le module était là avant : on doit le restaurer !
                    else:
                        # On récupère l'objet module actuel et l'original
                        current_mod = sys.modules.get(name)
                        original_mod = initial_modules[name]
                        
                        # Si l'objet a changé (ré-importé pendant le mount)
                        # on remet l'original dans sys.modules
                        sys.modules[name] = original_mod
                        
                        # CRUCIAL : Si l'utilisateur a une variable locale 'm', 
                        # elle pointe toujours vers 'current_mod'.
                        # On doit donc ré-injecter le contenu de l'original dedans.
                        if current_mod is not original_mod:
                            current_mod.__dict__.clear()
                            current_mod.__dict__.update(original_mod.__dict__)

_session = Session()
