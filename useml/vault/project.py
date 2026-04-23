import datetime
import importlib.metadata

from typing import Any, List, Optional
from pathlib import Path

from .snapshot import Snapshot

class Project:
    """
    Manages a collection of snapshots within a specific experiment or model category.
    
    A Project acts as a subdirectory within the Vault, responsible for versioning
    model weights and maintaining a chronological log of commits.
    """

    def __init__(self, path: Path):
        """
        Initializes the Project instance.

        Args:
            path (Path): The filesystem path where project snapshots are stored.
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def commit(
        self, 
        model: Any, 
        message: str, 
        optimizer: Optional[Any] = None, 
        **user_metadata: Any
    ) -> Snapshot:
        """
        Records the current state of a model as a new snapshot.

        This method generates a unique snapshot ID based on the current timestamp,
        collects system manifest data (versioning, timing), and delegates the
        saving process to the Snapshot class.

        Args:
            model: The PyTorch model (state_dict) to save.
            message (str): A descriptive message for this commit.
            optimizer (Optional[Any]): Optional optimizer state to preserve.
            **user_metadata: Arbitrary keyword arguments to store as experiment metrics.

        Returns:
            Snapshot: The newly created snapshot instance.
        """
        # Microseconds (%f) are included to prevent ID collisions during rapid commits
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_id = f"snap_{timestamp}"
        snap_path = self.path / snap_id
        
        # Dynamically fetch the framework version
        try:
            version = importlib.metadata.version("useml")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"

        # System-level information
        manifest = {
            "message": message,
            "timestamp": timestamp,
            "framework_version": version,
        }

        snapshot = Snapshot.create_new(snap_path)
        snapshot.save(model, optimizer, manifest, user_metadata)
        return snapshot

    def log(self) -> List[Snapshot]:
        """
        Retrieves the history of snapshots in the project.

        The list is sorted chronologically, with the most recent snapshot at index 0.

        Returns:
            List[Snapshot]: A list of Snapshot instances found in the project directory.
        """
        snapshots = [
            Snapshot(d) for d in self.path.iterdir() 
            if d.is_dir() and (d / "weights.pth").exists()
        ]
        
        # Sort by creation time, descending
        snapshots.sort(key=lambda x: x.path.stat().st_ctime, reverse=True)
        return snapshots

    def __getitem__(self, index: int) -> Snapshot:
        """
        Provides direct access to snapshots using list-like indexing.

        Args:
            index (int): The index of the snapshot (0 for the latest).

        Returns:
            Snapshot: The requested snapshot.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.log()[index]
        
    def __len__(self) -> int:
        """Returns the total number of valid snapshots in the project."""
        return len(self.log())

    def __repr__(self) -> str:
        return f"<useml.Project name='{self.path.name}' snapshots={len(self)}>"