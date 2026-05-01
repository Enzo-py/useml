import datetime
import hashlib
import importlib.metadata
import inspect
import sys
from pathlib import Path
from typing import Dict, List

from .snapshot import Snapshot
from ..session.component import Component
from .code_extractor import _get_source_assets


def _is_notebook() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


class ProjectError(Exception):
    """Base exception for project-related errors."""


class ProjectAlreadyExistsError(Exception):
    """Raised when a project with the same name already exists in the vault."""


class ProjectState:
    """In-RAM snapshot of a paused project, held by the session stash."""

    def __init__(
        self,
        project: "Project",
        components: Dict[str, Component],
        is_dirty: bool,
    ) -> None:
        self.project = project
        self.components = components
        self.is_dirty = is_dirty


class Project:
    """Collection of snapshots stored under a single project directory.

    A Project directory lives inside the Vault and contains one subdirectory
    per snapshot. The log is reconstructed by scanning the filesystem, so no
    index file is required.
    """

    def __init__(self, path: Path) -> None:
        """Initialises the project, creating the directory if absent.

        Args:
            path: Filesystem path for this project.
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        message: str,
        components: Dict[str, Component],
        **metrics,
    ) -> Snapshot:
        """Creates a reproducible snapshot of the tracked components.

        Args:
            message: Human-readable description of this snapshot.
            components: Named components to serialise.
            **metrics: Quantitative results attached to this snapshot
                (e.g. ``loss=0.1``, ``accuracy=0.95``).

        Returns:
            The newly created Snapshot instance.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        snap_path = self.path / f"snap_{timestamp}"

        try:
            version = importlib.metadata.version("useml")
        except importlib.metadata.PackageNotFoundError:
            version = "dev-local"

        inline_sources = _get_source_assets(components)
        manifest = {"components": self._build_component_manifest(components)}
        meta = {
            "message": message,
            "timestamp": timestamp,
            "iso_date": now.isoformat(),
            "useml_version": version,
            "environment": "notebook" if _is_notebook() else "script",
        }
        if metrics:
            meta["metrics"] = metrics

        snapshot = Snapshot(snap_path)
        snapshot.save(
            components=components,
            manifest=manifest,
            meta=meta,
            archive_source=False,
            inline_sources=inline_sources,
        )
        return snapshot

    def log(self) -> List[Snapshot]:
        """Returns all valid snapshots, newest first.

        A directory qualifies as a snapshot when it starts with ``snap_`` and
        contains a ``manifest.yaml`` file.

        Returns:
            Snapshots sorted by name in descending order (newest first).
        """
        if not self.path.exists():
            return []
        snapshots = [
            Snapshot(d)
            for d in self.path.iterdir()
            if (
                d.is_dir()
                and d.name.startswith("snap_")
                and (d / "manifest.yaml").exists()
            )
        ]
        snapshots.sort(key=lambda s: s.path.name, reverse=True)
        return snapshots

    def get_snapshot_path(self, snap_id: str) -> Path:
        """Returns the filesystem path for a given snapshot identifier.

        Args:
            snap_id: Snapshot directory name (e.g. ``snap_20260501_…``).

        Returns:
            Full path under this project directory.
        """
        return self.path / snap_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_component_manifest(
        self, components: Dict[str, Component]
    ) -> Dict:
        entries = {}
        for name, comp in components.items():
            model_cls = comp.model.__class__
            if comp.useml_config is not None:
                loss_class = comp.useml_config.loss_name()
                loss_hash = comp.useml_config.loss_hash()
            else:
                loss_class = None
                loss_hash = None
            entries[name] = {
                "class_name": model_cls.__name__,
                "module_path": model_cls.__module__,
                "weights": f"weights/{name}.pth",
                "config": f"configs/{name}.yaml" if comp.config else None,
                "code_hash": self._get_code_hash(model_cls),
                "loss_class": loss_class,
                "loss_hash": loss_hash,
            }
        return entries

    def _get_code_hash(self, cls: type) -> str:
        try:
            source = inspect.getsource(cls)
            return hashlib.md5(source.encode()).hexdigest()
        except OSError:
            return "unknown"

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> Snapshot:
        return self.log()[index]

    def __len__(self) -> int:
        return len(self.log())

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.path.name == other
        if isinstance(other, Project):
            return self.path.name == other.path.name
        raise TypeError(
            f"Cannot compare Project with {type(other).__name__}"
        )

    def __repr__(self) -> str:
        return (
            f"<useml.Project name='{self.path.name}' snapshots={len(self)}>"
        )
