import torch
import yaml

from pathlib import Path
from typing import Any, Dict, Optional, Union

class Snapshot:
    """
    Represents a point-in-time record of a model's state and associated metadata.
    
    A Snapshot is a directory containing the serialized model weights (pth) and 
    configuration files (yaml). It supports lazy loading of metadata to optimize 
    memory usage when browsing project history.
    """

    def __init__(self, path: Union[str, Path]):
        """
        Initializes a Snapshot instance from an existing directory.

        Args:
            path (Union[str, Path]): The directory containing snapshot files.
        """
        self.path = Path(path)
        self._manifest: Optional[Dict[str, Any]] = None
        self._metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create_new(cls, path: Path) -> "Snapshot":
        """
        Creates a new snapshot directory on disk and returns the instance.

        Args:
            path (Path): The destination directory for the new snapshot.

        Returns:
            Snapshot: An initialized Snapshot instance.
        """
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def save(
        self, 
        model: Any, 
        optimizer: Optional[Any], 
        manifest: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> None:
        """
        Serializes model weights and configuration to the snapshot directory.

        Args:
            model: The PyTorch model to save.
            optimizer: Optional PyTorch optimizer to save.
            manifest (Dict): System-level data (e.g., version, git hash).
            metadata (Dict): User-defined metrics and hyperparameters.
        """
        # 1. Save binary weights
        torch.save(model.state_dict(), self.path / "weights.pth")
        
        # 2. Save optimizer state if provided
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.path / "optimizer.pth")
        
        # 3. Save configuration files using YAML for human readability
        save_params = {"default_flow_style": False, "sort_keys": False}
        
        with open(self.path / "manifest.yaml", "w") as f:
            yaml.safe_dump(manifest, f, **save_params)
            
        with open(self.path / "metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f, **save_params)

    def _load_data(self) -> None:
        """
        Internal method to lazy-load YAML files into memory.
        
        This prevents unnecessary I/O when snapshots are listed but their 
        metadata is not yet accessed.
        """
        if self._manifest is None:
            # Load manifest
            manifest_path = self.path / "manifest.yaml"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    self._manifest = yaml.safe_load(f) or {}
            else:
                self._manifest = {}

            # Load user metadata
            metadata_path = self.path / "metadata.yaml"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self._metadata = yaml.safe_load(f) or {}
            else:
                self._metadata = {}

    def __getitem__(self, key: str) -> Any:
        """
        Accesses manifest or metadata values using key indexing.

        Priority is given to the system manifest, then user metadata.

        Args:
            key (str): The metadata or manifest key to retrieve.

        Returns:
            Any: The value associated with the key, or None if not found.
        """
        self._load_data()
        if self._manifest and key in self._manifest:
            return self._manifest[key]
        return self._metadata.get(key) if self._metadata else None

    def load_weights(self, model: Any) -> None:
        """
        Loads the saved weights into the provided model instance.

        Args:
            model: The PyTorch model to update with saved weights.
        """
        weights_path = self.path / "weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights.pth found in {self.path}")
            
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )

    def __repr__(self) -> str:
        return f"<useml.Snapshot id='{self.path.name}'>"
