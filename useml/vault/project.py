import datetime
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, List

from .snapshot import Snapshot

class ProjectError(Exception):
    """Base exception for all project-related errors."""
    pass

class ProjectAlreadyExistsError(Exception):
    """A project with the same name (identifier) is already defined in this vault."""
    pass

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
        components: Dict[str, Any], 
        **user_metadata: Any
    ) -> Snapshot:
        """Records the current state of all tracked components and source code.

        This method performs four main actions:
        1. Generates a unique snapshot ID based on high-precision timestamp.
        2. Archives the current project source code for full reproducibility.
        3. Serializes component weights and captures their class/module identity.
        4. Stores metadata and metrics in a manifest.yaml file.

        Args:
            message: A descriptive note for this snapshot.
            components: Dictionary of named components (e.g., {'model': MyNet()}).
            **user_metadata: Arbitrary metrics or hyperparameters (e.g., lr=0.01).

        Returns:
            Snapshot: The newly created snapshot instance.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        snap_id = f"snap_{timestamp}"
        snap_path = self.get_snapshot_path(snap_id)

        # get useml version dynamicly
        try:
            version = importlib.metadata.version("useml")
        except importlib.metadata.PackageNotFoundError:
            version = "dev-local"

        component_map = {}
        for name, obj in components.items():
            component_map[name] = {
                "file": f"{name}.pth",
                "class_name": obj.__class__.__name__,
                "module_path": obj.__class__.__module__,
            }

        manifest = {
            "info": {
                "message": message,
                "timestamp": timestamp,
                "iso_date": now.isoformat(),
                "useml_version": version,
            },
            "components": component_map,
        }

        snapshot = Snapshot(snap_path)
        
        snapshot.save(
            components=components,
            manifest=manifest,
            metadata=user_metadata,
            archive_source=True
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
    