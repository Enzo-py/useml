import datetime
import hashlib
import importlib.metadata
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .snapshot import Snapshot
from ..session.component import Component
from .code_extractor import _get_source_assets
from ..errors import ProjectAlreadyExistsError, ProjectTypeError

# Backward-compatible alias
ProjectError = ProjectAlreadyExistsError


def _is_notebook() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


class ProjectState:
    """Represents a paused project state held in RAM."""

    def __init__(self, project, components, is_dirty):
        self.project = project
        self.components = components
        self.is_dirty = is_dirty


class Project:
    """Manages a collection of snapshots within a specific project directory.

    A Project is a subdirectory of the Vault. It versions model weights and
    maintains a chronological log of commits by scanning the filesystem.
    """

    def __init__(self, path: Path) -> None:
        """Initializes the Project instance.

        Args:
            path: The filesystem path where snapshots are stored.
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        message: str,
        components: Dict[str, Component],
        bundle_meta: Optional[dict] = None,
        bundle_inline_source: Optional[dict] = None,
        **metrics,
    ) -> Snapshot:
        """Creates a reproducible snapshot of tracked components.

        Args:
            message: Human-readable description of this checkpoint.
            components: Named Component objects to serialise.
            **metrics: Quantitative results attached to the snapshot metadata.

        Returns:
            The created Snapshot instance.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        snap_id = f"snap_{timestamp}"
        snap_path = self.path / snap_id

        try:
            version = importlib.metadata.version("useml")
        except importlib.metadata.PackageNotFoundError:
            version = "dev-local"

        inline_sources = _get_source_assets(components)

        component_manifest = {}
        for name, comp in components.items():
            model_class = comp.model.__class__

            if comp.useml_config is not None:
                loss_class = comp.useml_config.loss_name()
                loss_hash = comp.useml_config.loss_hash()
            else:
                loss_class = None
                loss_hash = None

            component_manifest[name] = {
                "class_name": model_class.__name__,
                "module_path": model_class.__module__,
                "weights": f"weights/{name}.pth",
                "config": f"configs/{name}.yaml" if comp.config else None,
                "code_hash": self._get_code_hash(model_class),
                "loss_class": loss_class,
                "loss_hash": loss_hash,
            }

        manifest = {"components": component_manifest}

        meta = {
            "message": message,
            "timestamp": timestamp,
            "iso_date": now.isoformat(),
            "useml_version": version,
            "environment": "notebook" if _is_notebook() else "script",
        }
        if metrics:
            meta["metrics"] = metrics
        if bundle_meta:
            meta["data"] = bundle_meta

        if bundle_inline_source:
            inline_sources = {**inline_sources, **bundle_inline_source}

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
        """Returns all valid snapshots, sorted newest-first.

        Returns:
            List of Snapshot instances.
        """
        if not self.path.exists():
            return []

        snapshots = [
            Snapshot(d)
            for d in self.path.iterdir()
            if d.is_dir()
            and d.name.startswith("snap_")
            and (d / "manifest.yaml").exists()
        ]
        snapshots.sort(key=lambda s: s.path.name, reverse=True)
        return snapshots

    def _get_code_hash(self, cls: type) -> str:
        try:
            source = inspect.getsource(cls)
            return hashlib.md5(source.encode()).hexdigest()
        except OSError:
            return "unknown"

    def __getitem__(self, index: int) -> Snapshot:
        return self.log()[index]

    def __len__(self) -> int:
        return len(self.log())

    def __repr__(self) -> str:
        return f"<useml.Project name='{self.path.name}' snapshots={len(self)}>"

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            return self.path.name == value
        if isinstance(value, Project):
            return self.path.name == value.path.name
        raise ProjectTypeError(
            f"Cannot compare a Project to {type(value).__name__}."
        )
