# useml/session/manager.py (sections à améliorer)

import hashlib
import importlib
import inspect
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import torch

from ..vault.core import Vault
from ..vault.project import Project, ProjectState
from ..vault.snapshot import Snapshot
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
    pass


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
        self._mounted_snapshot: Optional[str] = None
        self._mounted_sys_path: Optional[str] = None

    @property
    def workdir(self):
        import useml.workdir as workdir_module
        return workdir_module

    @property
    def imports(self):
        """Returns the ImportManager for this session."""
        from useml.imports import ImportManager
        return ImportManager(self)

    def connect(self, vault_path: Union[str, Path]) -> None:
        """Connects the session to a storage vault.

        Args:
            vault_path: Path to the root storage directory.
        """
        self.vault = Vault(Path(vault_path))
        logger.info(f"Connected to vault: {self.vault.path}")

    def get_projects(self) -> List[Project]:
        """Lists all projects available in the connected vault.

        Returns:
            A list of existing project instances.

        Raises:
            RuntimeError: If the session is not connected to a vault.
        """
        if not self.vault:
            raise RuntimeError(
                "Not connected to a vault. Call useml.init('path') first."
            )
        return self.vault.projects()

    @property
    def project(self) -> Project:
        """Returns the currently focused project.
        
        Returns:
            The focused Project instance.
            
        Raises:
            NoSessionFocusError: If no project is currently focused.
        """
        if self._project is None:
            raise NoSessionFocusError(
                "No project in focus. Call useml.focus('name') or "
                "useml.new('name') first."
            )
        return self._project

    def set_focus(self, project_name: str, force: bool = False) -> None:
        """Sets the active project focus.
        
        Args:
            project_name: Name of the project to focus on.
            force: If True, discard unsaved changes without raising an error.
            
        Raises:
            UncommittedSessionError: If there are unsaved changes and force=False.
        """
        # Check if already focused on the same project
        if self._project is not None and self._project == project_name:
            return

        # Check if project is in RAM stash
        if project_name in self._stash:
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

        # Dirty state guard for new focus from disk
        if self._is_dirty and not force:
            raise UncommittedSessionError(
                f"Project '{project_name}' has unsaved changes in RAM. "
                f"Please use useml.commit() or useml.stash()."
            )

        # Load project from vault
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
        
        # Reset internal state
        self._project = None
        self.components = {}
        self._is_dirty = False

    def track(
        self,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Registers a component for the current session.
        
        Args:
            name: Unique identifier for the component.
            model: PyTorch model or similar object.
            config: Optional hyperparameters or configuration.
            optimizer: Optional associated optimizer.
        """
        self.components[name] = Component(
            name=name, model=model, config=config, optimizer=optimizer
        )
        self._is_dirty = True

    def commit(self, message: str, **metrics: Any) -> Any:
        """Saves a snapshot to the focused project.
        
        Args:
            message: Description of the changes or experiment state.
            **metrics: Quantitative results (e.g., loss=0.1, accuracy=0.95).
            
        Returns:
            The created snapshot instance.
        """
        snap = self.project.commit(
            message=message, components=self.components, **metrics
        )
        self._is_dirty = False
        return snap
    
    def untrack(self, name: str) -> None:
        """Removes a component from tracking.
        
        Args:
            name: Name of the component to untrack.
        """
        if name not in self.components:
            return

        # Check if component exists in the latest snapshot on disk
        exists_on_disk = False
        if self._project and self._project.log():
            latest = self._project.log()[0]
            exists_on_disk = name in latest.components

        del self.components[name]

        if exists_on_disk:
            # Removed a saved component -> dirty state
            self._is_dirty = True
        else:
            # Removed an unsaved component -> recalculate dirty state
            self._is_dirty = self._check_if_still_dirty_after_removal()

    def mount(self, snapshot_tag: str) -> None:
        """Mounts a snapshot for useml-managed code access.
        
        This does NOT modify global sys.path. It only affects:
        - useml.load() behavior
        - useml.workdir.* imports
        
        Regular Python imports (from models import X) are unaffected.
        
        Args:
            snapshot_tag: Snapshot identifier. Supports:
                - "\\latest": Most recent snapshot
                - "\\head~N": N commits before latest (e.g., "\\head~2")
                - "\\current": Current working directory (unmount)
                - Direct snapshot folder name
                
        Raises:
            NoSessionFocusError: If no project is currently focused.
            FileNotFoundError: If the snapshot or its source directory doesn't exist.
        """
        # Clear all cached workdir modules so the next import re-runs the loader
        for name in list(sys.modules):
            if name.startswith("useml.workdir.") or name.startswith("_useml_workdir_internal"):
                del sys.modules[name]

        if self._mounted_sys_path:
            self._clear_project_modules()
        
        # Unmount — both \current and \workdir are accepted
        if snapshot_tag in ("\\current", "\\workdir"):
            self._mounted_snapshot = None
            self._mounted_sys_path = None
            logger.info("Unmounted snapshot (back to current workdir)")
            return

        # Resolve snapshot path
        snapshot_path = self._resolve_snapshot_path(snapshot_tag)
        source_dir = snapshot_path / "source"

        if not source_dir.exists():
            raise FileNotFoundError(
                f"No source directory found for snapshot: {snapshot_tag}"
            )

        # Store mount info (don't modify global sys.path or sys.modules)
        self._mounted_snapshot = snapshot_tag
        self._mounted_sys_path = str(source_dir)
        
        logger.info(f"Mounted snapshot: {snapshot_tag}")

    def load(
        self, 
        model_name: str, 
        _from: Optional[str] = None
    ) -> torch.nn.Module:
        """Loads a saved model with its weights.
        
        Args:
            model_name: Name of the model to load.
            _from: Snapshot to load weights from. If None, uses currently mounted
                   snapshot or "\\latest". Supports same tags as mount().
                   
        Returns:
            PyTorch module with loaded weights.
            
        Raises:
            NoSessionFocusError: If no project is currently focused.
            FileNotFoundError: If model weights are not found.
            RuntimeError: If weights fail to load due to code/weights mismatch.
            
        Examples:
            >>> # Load with full code isolation
            >>> useml.mount("\\latest")
            >>> model = useml.load("model1")
            
            >>> # Load weights into current code
            >>> useml.mount("\\current")
            >>> model = useml.load("model1", _from="\\latest")
        """
        # Determine weights source
        weights_snapshot_tag = (
            _from or self._mounted_snapshot or "\\latest"
        )
        snapshot_path = self._resolve_snapshot_path(weights_snapshot_tag)
        
        # Load snapshot to get component metadata
        snapshot = Snapshot(snapshot_path)
        
        if model_name not in snapshot.components:
            raise FileNotFoundError(
                f"Model '{model_name}' not found in "
                f"snapshot {weights_snapshot_tag}"
            )
        
        component_meta = snapshot.components[model_name]
        
        # Load weights file
        weights_file = snapshot_path / component_meta["weights"]
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_file}"
            )
        
        state_dict = torch.load(
            weights_file, 
            map_location="cpu", 
            weights_only=True
        )
        
        # Get code hash from manifest
        model_code_hash = component_meta.get("code_hash", None)
        
        # Import model class based on isolation mode
        module_path = component_meta["module_path"]
        class_name = component_meta["class_name"]
        
        if self._mounted_snapshot and not _from:
            # Full isolation: use code from mounted snapshot
            ModelClass = self._import_from_mounted(module_path, class_name)
        else:
            # Current code: use workdir code
            ModelClass = self._import_from_workdir(module_path, class_name)
            
            # Check code compatibility
            current_code_hash = self._get_code_hash(ModelClass)
            if model_code_hash and current_code_hash != model_code_hash:
                logger.warning(
                    f"Code for '{model_name}' has changed!\n"
                    f"  Old hash: {model_code_hash}\n"
                    f"  New hash: {current_code_hash}\n"
                    f"  This may cause weight loading failures."
                )
        
        # Instantiate and load weights
        model = ModelClass()
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load '{model_name}' weights "
                f"(code/weights mismatch):\n{e}"
            )
        
        return model

    # ===== Private Helper Methods =====

    def _clear_project_modules(self) -> None:
        """Clears modules loaded from previously mounted snapshot.
        
        Only removes modules that were loaded from the old snapshot path,
        preserving modules from other sources.
        """
        import sys

        if not self._mounted_sys_path:
            return

        path = self._mounted_sys_path

        for name, mod in list(sys.modules.items()):
            try:
                mod_file = getattr(mod, "__file__", "")
                if mod_file and mod_file.startswith(path):
                    del sys.modules[name]
            except Exception:
                # Module might not be deletable, skip it
                pass

    def _check_if_still_dirty_after_removal(self) -> bool:
        """Checks if session is still dirty after component removal.
        
        Returns:
            True if session has unsaved changes, False otherwise.
        """
        if not self.components:
            return False
        return self._is_dirty

    def _resolve_snapshot_path(self, snapshot_tag: str) -> Path:
        """Resolves a snapshot tag to an absolute path.
        
        Args:
            snapshot_tag: Tag to resolve. Supports:
                - "\\latest": Most recent snapshot
                - "\\head~N": N commits before latest
                - Direct snapshot folder name
            
        Returns:
            Absolute path to the snapshot directory.
            
        Raises:
            NoSessionFocusError: If no project is focused.
            ValueError: If the tag format is invalid or snapshot not found.
        """
        if self._project is None:
            raise NoSessionFocusError(
                "No project in focus. Call useml.focus() first."
            )
        
        snapshots = self._project.log()
        
        if snapshot_tag == "\\latest":
            if not snapshots:
                raise ValueError("No snapshots found in project")
            return snapshots[0].path
        
        if snapshot_tag.startswith("\\head~"):
            try:
                offset = int(snapshot_tag[6:])
            except ValueError:
                raise ValueError(
                    f"Invalid snapshot tag format: {snapshot_tag}. "
                    f"Use \\head~N where N is an integer."
                )
            
            if offset < 0:
                raise ValueError(f"Offset must be non-negative: {offset}")
            
            if offset >= len(snapshots):
                raise ValueError(
                    f"Offset {offset} exceeds available snapshots "
                    f"({len(snapshots)} total)"
                )
            
            return snapshots[offset].path
        
        # Direct snapshot folder name
        snapshot_path = self._project.path / "snapshots" / snapshot_tag
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot not found: {snapshot_tag}")
        
        return snapshot_path

    def _import_from_mounted(
        self, 
        module_path: str, 
        class_name: str
    ) -> type:
        """Imports a class from currently mounted snapshot code.
        
        Args:
            module_path: Module path (e.g., "models.mymodel").
            class_name: Class name (e.g., "MyModel").
            
        Returns:
            The imported class object.
            
        Raises:
            ImportError: If the class cannot be imported from mounted snapshot.
        """
        import sys

        # Remove conflicting global modules to force reload from snapshot
        to_delete = [
            k for k in list(sys.modules.keys())
            if k == module_path or k.startswith(module_path + ".")
        ]
        for k in to_delete:
            del sys.modules[k]

        try:
            # Import via useml.workdir hook
            module = importlib.import_module(
                f"useml.workdir.{module_path}"
            )
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_path} "
                f"in mounted snapshot: {e}"
            )

    def _import_from_workdir(
        self, 
        module_path: str, 
        class_name: str
    ) -> type:
        """Imports a class from current working directory code.
        
        Args:
            module_path: Module path (e.g., "models.mymodel" or "__main__").
            class_name: Class name (e.g., "MyModel").
            
        Returns:
            The imported class object.
            
        Raises:
            ImportError: If the class cannot be imported from workdir.
        """
        try:
            if module_path == "__main__":
                import __main__
                module = __main__
            else:
                module = importlib.import_module(module_path)
            
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_path} "
                f"in workdir: {e}"
            )

    def _get_code_hash(self, cls: type) -> str:
        """Computes the MD5 hash of a class's source code.
        
        Args:
            cls: The class to hash.
            
        Returns:
            Hexadecimal MD5 hash string, or "unknown" if source unavailable.
        """
        try:
            source = inspect.getsource(cls)
            return hashlib.md5(source.encode()).hexdigest()
        except OSError:
            # Source unavailable (built-in class, compiled, etc.)
            return "unknown"


_session = Session()
