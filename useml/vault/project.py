import sys
import datetime
import importlib.metadata

from pathlib import Path
from typing import Dict, List

from .snapshot import Snapshot
from ..session.component import Component
from .code_extractor import _get_source_assets

class ProjectError(Exception):
    """Base exception for all project-related errors."""
    pass

class ProjectAlreadyExistsError(Exception):
    """A project with the same name (identifier) is already defined in this vault."""
    pass

def _is_notebook() -> bool:
    """Détecte si le code tourne dans un kernel Jupyter/IPython."""
    return 'ipykernel' in sys.modules or 'IPython' in sys.modules

class ProjectState:
    """Represents a paused project state in RAM."""
    def __init__(self, project, components, is_dirty):
        self.project = project
        self.components = components
        self.is_dirty = is_dirty

class Project:
    """Manages a collection of snapshots within a specific project directory.

    A Project acts as a subdirectory within the Vault, responsible for versioning
    model weights and maintaining a chronological log of commits by scanning
    the filesystem dynamically.
    """

    def __init__(self, path: Path):
        """Initializes the Project instance.

        Args:
            path (Path): The filesystem path where snapshots are stored.
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        message: str,
        components: Dict[str, Component],
    ) -> Snapshot:
        """Create a reproducible snapshot of tracked components."""

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        snap_id = f"snap_{timestamp}"
        snap_path = self.get_snapshot_path(snap_id)

        # --- version ---
        try:
            version = importlib.metadata.version("useml")
        except importlib.metadata.PackageNotFoundError:
            version = "dev-local"

        # --- extract inline sources ---
        inline_sources = _get_source_assets(components)
        print(inline_sources)

        # --- build manifest (STRICT: only structure) ---
        component_manifest = {}
        for name, comp in components.items():
            model_class = comp.model.__class__
            
            component_manifest[name] = {
                "source": f"source/{name}.py",
                "class_name": model_class.__name__,  # Ex: "MyModel", pas "Component"
                "module_path": model_class.__module__,  # Ex: "models.mymodel"
                "weights": f"weights/{name}.pth",
                "config": f"config/{name}.yaml" if comp.config else None,
                "code_hash": self._get_code_hash(model_class),
            }

        manifest = {
            "components": component_manifest,
            # keep manifest deterministic → no timestamps, no message, no env
        }

        # --- build meta (context only) ---
        meta = {
            "message": message,
            "timestamp": timestamp,
            "iso_date": now.isoformat(),
            "useml_version": version,
            "environment": "notebook" if _is_notebook() else "script",
        }

        # --- snapshot save ---
        snapshot = Snapshot(snap_path)

        snapshot.save(
            components=components,
            manifest=manifest,
            meta=meta,  # renamed (clean separation)
            archive_source=False,
            inline_sources=inline_sources,
        )

        return snapshot

    def log(self) -> List[Snapshot]:
        """Retrieves snapshots from disk, sorted by timestamp (newest first).

        A directory is considered a valid snapshot if it starts with 'snap_'
        and contains a 'manifest.yaml' file.

        Returns:
            List[Snapshot]: Chronological list of snapshots.
        """
        if not self.path.exists():
            return []

        snapshots = []
        for d in self.path.iterdir():
            # Filter: must be a dir, start with snap_ AND have a manifest
            if d.is_dir() and d.name.startswith("snap_") and (d / "manifest.yaml").exists():
                snapshots.append(Snapshot(d))

        # Sort by folder name (alphabetic sort on timestamp = chronological sort)
        snapshots.sort(key=lambda x: x.path.name, reverse=True)
        return snapshots

    def get_snapshot_path(self, snap_id) -> Path:
        return self.path / snap_id
    
    def _get_code_hash(self, cls: type) -> str:
        """Computes the MD5 hash of a class's source code."""
        import hashlib
        import inspect
        
        try:
            source = inspect.getsource(cls)
            return hashlib.md5(source.encode()).hexdigest()
        except OSError:
            return "unknown"

    def __getitem__(self, index: int) -> Snapshot:
        """Access a snapshot by index (0 is the latest)."""
        return self.log()[index]

    def __len__(self) -> int:
        """Returns the count of valid snapshots on disk."""
        return len(self.log())

    def __repr__(self) -> str:
        return f"<useml.Project name='{self.path.name}' snapshots={len(self)}>"
    
    def __eq__(self, value):
        if isinstance(value, str):
            return self.path.name == value
        
        if isinstance(value, Project):
            return self.path.name == value.path.name
        
        raise RuntimeError(f"A useml.Project cannot be compare to {type(value)}")
    