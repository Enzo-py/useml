import hashlib
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from ..errors import (
    NotConnectedError,
    NoFocusError,
    UncommittedChangesError,
    SnapshotNotFoundError,
    InvalidSnapshotTagError,
    WeightsNotFoundError,
    WeightsLoadError,
    NoSourceDirectoryError,
    SnapshotModuleNotFoundError,
    WorkdirImportError,
)
from ..vault.core import Vault
from ..vault.project import Project, ProjectState
from ..vault.snapshot import Snapshot
from .component import Component

logger = logging.getLogger(__name__)

# Backward-compatible aliases (tests import these names from this module)
NoSessionFocusError = NoFocusError
UncommittedSessionError = UncommittedChangesError


class Session:
    """Global workspace that bridges the public API and the vault layer.

    Holds the connection to a Vault, the currently focused Project, the set of
    tracked Components, and the mounted snapshot for workdir imports.
    """

    def __init__(self) -> None:
        self.vault: Optional[Vault] = None
        self._project: Optional[Project] = None
        self.components: Dict[str, Component] = {}
        self._is_dirty: bool = False
        self._stash: Dict[str, ProjectState] = {}
        self._mounted_snapshot: Optional[str] = None
        self._mounted_sys_path: Optional[str] = None

    @property
    def workdir(self):
        import useml.workdir as _workdir
        return _workdir

    @property
    def imports(self):
        """Returns the ImportManager for the active session."""
        from useml.imports import ImportManager
        return ImportManager(self)

    def connect(self, vault_path: Union[str, Path]) -> None:
        """Connects the session to a storage vault.

        Args:
            vault_path: Path to the root storage directory.
        """
        self.vault = Vault(Path(vault_path))
        logger.info("Connected to vault: %s", self.vault.path)

    def get_projects(self) -> List[Project]:
        """Lists all projects in the connected vault.

        Returns:
            All existing Project instances.

        Raises:
            NotConnectedError: If the session is not connected to a vault.
        """
        if not self.vault:
            raise NotConnectedError(
                "Not connected to a vault. Call useml.init('path') first."
            )
        return self.vault.projects()

    @property
    def project(self) -> Project:
        """The currently focused project.

        Raises:
            NoFocusError: If no project is focused.
        """
        if self._project is None:
            raise NoFocusError(
                "No project in focus. Call useml.focus('name') or "
                "useml.new('name') first."
            )
        return self._project

    def set_focus(self, project_name: str, force: bool = False) -> None:
        """Switches focus to the named project.

        Args:
            project_name: Name of the project to focus on.
            force: When True, discards unsaved changes without raising.

        Raises:
            UncommittedChangesError: If there are unsaved changes and force is False.
        """
        if self._project is not None and self._project == project_name:
            return

        if project_name in self._stash:
            if self._is_dirty and not force:
                raise UncommittedChangesError(
                    "You have unsaved changes in RAM. "
                    "Call useml.commit() or useml.stash() first."
                )
            state = self._stash.pop(project_name)
            self._project = state.project
            self.components = state.components
            self._is_dirty = state.is_dirty
            return

        if self._is_dirty and not force:
            raise UncommittedChangesError(
                "You have unsaved changes in RAM. "
                "Call useml.commit() or useml.stash() first."
            )

        self._project = self.vault.get_project(project_name)
        self.components = {}
        self._is_dirty = False

    def stash(self) -> None:
        """Pushes the current project state into the RAM stash and resets focus."""
        if self._project is None:
            return
        name = self._project.path.name
        self._stash[name] = ProjectState(
            project=self._project,
            components=self.components.copy(),
            is_dirty=self._is_dirty,
        )
        self._project = None
        self.components = {}
        self._is_dirty = False

    def track(
        self,
        name: str,
        model: Any,
        config: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Registers a component for the current session.

        Args:
            name: Unique identifier for the component.
            model: PyTorch model or compatible object.
            config: Hyperparameters as a plain dict or a ``useml.Config``.
            optimizer: Optional associated optimiser.
        """
        self.components[name] = Component(
            name=name, model=model, config=config, optimizer=optimizer
        )
        self._is_dirty = True

    def commit(self, message: str, **metrics: Any) -> Snapshot:
        """Saves a snapshot of all tracked components to the focused project.

        Args:
            message: Description of the changes or experiment state.
            **metrics: Quantitative results (e.g. ``loss=0.1``).

        Returns:
            The created Snapshot instance.
        """
        snap = self.project.commit(
            message=message, components=self.components, **metrics
        )
        self._is_dirty = False
        return snap

    def untrack(self, name: str) -> None:
        """Removes a component from tracking.

        Args:
            name: Name of the component to remove.
        """
        if name not in self.components:
            return

        exists_on_disk = False
        if self._project and self._project.log():
            exists_on_disk = name in self._project.log()[0].components

        del self.components[name]

        if exists_on_disk:
            self._is_dirty = True
        elif not self.components:
            self._is_dirty = False

    def mount(self, snapshot_tag: str) -> None:
        """Mounts a snapshot for ``useml.workdir.*`` imports.

        Args:
            snapshot_tag: Snapshot identifier — ``"\\latest"``, ``"\\head~N"``,
                ``"\\current"`` / ``"\\workdir"`` to unmount, or a folder name.

        Raises:
            NoFocusError: If no project is focused.
            NoSourceDirectoryError: If the snapshot has no ``source/`` directory.
        """
        if self._mounted_sys_path:
            self._clear_mounted_modules()

        for name in list(sys.modules):
            if name.startswith("_useml_workdir_internal"):
                del sys.modules[name]

        if snapshot_tag in ("\\current", "\\workdir"):
            self._mounted_snapshot = None
            self._mounted_sys_path = None
            logger.info("Unmounted snapshot (back to current workdir)")
            return

        snapshot_path = self._resolve_snapshot_path(snapshot_tag)
        source_dir = snapshot_path / "source"

        if not source_dir.exists():
            raise NoSourceDirectoryError(
                f"No source directory found for snapshot '{snapshot_tag}'."
            )

        self._mounted_snapshot = snapshot_tag
        self._mounted_sys_path = str(source_dir)
        logger.info("Mounted snapshot: %s", snapshot_tag)

    def load(
        self,
        model_name: str,
        _from: Optional[str] = None,
    ) -> torch.nn.Module:
        """Loads a saved model with its weights from a snapshot.

        Args:
            model_name: Name of the model component to load.
            _from: Snapshot tag to load weights from. Defaults to the
                currently mounted snapshot, or ``"\\latest"`` if none is mounted.

        Returns:
            PyTorch module with loaded weights.

        Raises:
            NoFocusError: If no project is focused.
            WeightsNotFoundError: If the model or weights file is not found.
            WeightsLoadError: If weights cannot be loaded due to a mismatch.
        """
        tag = _from or self._mounted_snapshot or "\\latest"
        snapshot_path = self._resolve_snapshot_path(tag)
        snapshot = Snapshot(snapshot_path)

        if model_name not in snapshot.components:
            raise WeightsNotFoundError(
                f"Model '{model_name}' not found in snapshot '{tag}'."
            )

        meta = snapshot.components[model_name]
        weights_file = snapshot_path / meta["weights"]
        if not weights_file.exists():
            raise WeightsNotFoundError(
                f"Weights file missing: {weights_file}"
            )

        state_dict = torch.load(
            weights_file, map_location="cpu", weights_only=True
        )

        module_path = meta["module_path"]
        class_name = meta["class_name"]

        if self._mounted_snapshot and not _from:
            ModelClass = self._import_from_snapshot(module_path, class_name)
        else:
            ModelClass = self._import_from_workdir(module_path, class_name)
            saved_hash = meta.get("code_hash")
            current_hash = self._get_code_hash(ModelClass)
            if saved_hash and current_hash != saved_hash:
                logger.warning(
                    "Code for '%s' has changed (old=%s… new=%s…). "
                    "Weight loading may fail.",
                    model_name,
                    saved_hash[:8],
                    current_hash[:8],
                )

        model = ModelClass()
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise WeightsLoadError(
                f"Failed to load '{model_name}' weights "
                f"(code/weights mismatch): {exc}"
            ) from exc

        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_mounted_modules(self) -> None:
        if not self._mounted_sys_path:
            return
        path = self._mounted_sys_path
        for name, mod in list(sys.modules.items()):
            mod_file = getattr(mod, "__file__", "") or ""
            if mod_file.startswith(path):
                try:
                    del sys.modules[name]
                except Exception:
                    pass

    def _resolve_snapshot_path(self, tag: str) -> Path:
        """Resolves a snapshot tag to a filesystem path.

        Args:
            tag: Snapshot identifier.

        Returns:
            Absolute path to the snapshot directory.

        Raises:
            NoFocusError: If no project is focused.
            SnapshotNotFoundError: If the tag resolves to nothing on disk.
            InvalidSnapshotTagError: If the tag format is malformed.
        """
        if self._project is None:
            raise NoFocusError(
                "No project in focus. Call useml.focus() first."
            )

        snapshots = self._project.log()

        if tag == "\\latest":
            if not snapshots:
                raise SnapshotNotFoundError("No snapshots found in project.")
            return snapshots[0].path

        if tag.startswith("\\head~"):
            try:
                offset = int(tag[6:])
            except ValueError:
                raise InvalidSnapshotTagError(
                    f"Invalid tag '{tag}'. Expected '\\head~N' where N is an integer."
                )
            if offset < 0:
                raise InvalidSnapshotTagError(
                    f"Offset must be non-negative, got {offset}."
                )
            if offset >= len(snapshots):
                raise SnapshotNotFoundError(
                    f"Offset {offset} out of range ({len(snapshots)} snapshots available)."
                )
            return snapshots[offset].path

        snap_path = self._project.path / tag
        if not snap_path.exists():
            raise SnapshotNotFoundError(f"Snapshot not found: '{tag}'.")
        return snap_path

    def _import_from_snapshot(self, module_path: str, class_name: str) -> type:
        """Imports a class from the currently mounted snapshot.

        Args:
            module_path: Dotted module path.
            class_name: Name of the class to import.

        Returns:
            The imported class object.

        Raises:
            SnapshotModuleNotFoundError: If the class cannot be found in the snapshot.
        """
        for key in list(sys.modules):
            if key == module_path or key.startswith(module_path + "."):
                del sys.modules[key]
        try:
            module = importlib.import_module(f"useml.workdir.{module_path}")
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise SnapshotModuleNotFoundError(
                f"Cannot import {class_name} from '{module_path}' "
                f"in mounted snapshot: {exc}"
            ) from exc

    def _import_from_workdir(self, module_path: str, class_name: str) -> type:
        """Imports a class from the current working directory.

        Args:
            module_path: Dotted module path, or ``"__main__"``.
            class_name: Name of the class to import.

        Returns:
            The imported class object.

        Raises:
            WorkdirImportError: If the class cannot be found.
        """
        try:
            if module_path == "__main__":
                import __main__ as module
            else:
                module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise WorkdirImportError(
                f"Cannot import {class_name} from '{module_path}': {exc}"
            ) from exc

    def _get_code_hash(self, cls: type) -> str:
        try:
            source = inspect.getsource(cls)
            return hashlib.md5(source.encode()).hexdigest()
        except OSError:
            return "unknown"


_session = Session()
