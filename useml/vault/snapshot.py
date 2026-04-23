import torch
import yaml
import shutil

from pathlib import Path
from typing import Any, Dict, Optional, Union

ARCHIVE_SOURCE_IGNORE_PATTERNS = ["vault*", ".git*", "__pycache__*", "*.pyc", ".DS_Store", "outputs*"]

class SnapshotError(Exception):
    """Base exception for all snapshot-related errors."""
    pass


class SnapshotOverwriteError(SnapshotError):
    """Raised when an operation would overwrite an existing, non-empty snapshot."""
    pass

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
        self.id = self.path.name

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
        components: Dict[str, Any],
        manifest: Dict[str, Any],
        metadata: Dict[str, Any],
        archive_source: bool = True
    ) -> None:
        """Serializes components, system manifest, and user metadata to the vault.

        This method creates a self-contained capsule of the experiment. It archives:
        1. Model weights and optimizer states (Binary).
        2. Component configurations (YAML).
        3. The entire source code tree (for seamless 'useml.mount').
        4. Metadata and System Manifest (YAML).

        Args:
            components: Dictionary of named components to serialize.
            manifest: System-level data (framework version, high-level map).
            metadata: User-defined metrics and hyperparameters.
            archive_source: If True, copies the current working directory into 
                the snapshot's source folder.

        Raises:
            SnapshotOverwriteError: If the snapshot directory already contains data.
        """
        # 1. Anti-Overwrite Guard
        if self.path.exists() and any(self.path.iterdir()):
            raise SnapshotOverwriteError(
                f"Conflict: Directory '{self.path}' is not empty. "
                "Snapshot overwriting is disabled for data integrity."
            )

        # 2. Directory structure initialization
        # 'source' contiendra l'arborescence complète pour le mount
        dirs = {
            "weights": self.path / "weights",
            "configs": self.path / "configs",
            "optimizers": self.path / "optimizers",
            "source": self.path / "source",
        }

        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        # 3. Serialization configuration (Google Style YAML)
        yaml_params = {"default_flow_style": False, "sort_keys": False}

        # 4. Component serialization loop
        for name, comp in components.items():
            # A. Model Weights
            # On utilise le dictionnaire 'components' du manifest pour le mapping
            torch.save(comp.model.state_dict(), dirs["weights"] / f"{name}.pth")

            # B. Optimizer State
            if hasattr(comp, 'optimizer') and comp.optimizer is not None:
                torch.save(
                    comp.optimizer.state_dict(), 
                    dirs["optimizers"] / f"{name}.pth"
                )

            # C. Component Configuration
            if hasattr(comp, 'config') and comp.config is not None:
                config_file = dirs["configs"] / f"{name}.yaml"
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(comp.config, f, **yaml_params)

        # 5. Full Source Code Backup (The "Time Machine" part)
        # Contrairement à une simple copie de fichier, on capture tout le projet
        if archive_source:
            ignore_func = shutil.ignore_patterns(*ARCHIVE_SOURCE_IGNORE_PATTERNS)
            shutil.copytree(
                Path.cwd(), 
                dirs["source"], 
                ignore=ignore_func, 
                dirs_exist_ok=True
            )

        # 6. System Manifest & User Metadata (Strictly Separated)
        # Le manifest aide UseML à reconstruire la session
        with open(self.path / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, **yaml_params)

        # Le metadata contient les scores de l'utilisateur (Accuracy, Loss...)
        with open(self.path / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, **yaml_params)

    def _load_yaml(self, filename: str) -> dict:
        file_path = self.path / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def manifest(self) -> dict:
        """System-level data (message, timestamp, version)."""
        if self._manifest is None:
            self._manifest = self._load_yaml("manifest.yaml")
        return self._manifest.get("info", {})

    @property
    def components(self) -> dict:
        """Registry of tracked components and their file mappings."""
        if self._manifest is None:
            self._manifest = self._load_yaml("manifest.yaml")
        return self._manifest.get("components", {})

    @property
    def user_metadata(self) -> dict:
        """User-defined metrics (accuracy, loss, etc.)."""
        if self._metadata is None:
            self._metadata = self._load_yaml("metadata.yaml")
        return self._metadata

    def load_component(self, component: Any) -> None:
        """Restores weights and optimizer state for a specific component.

        Args:
            component (Component): The component instance to populate.

        Raises:
            FileNotFoundError: If the component's weights are missing.
        """
        weights_path = self.path / "weights" / f"{component.name}.pth"
        optim_path = self.path / "optimizers" / f"{component.name}.pth"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"No weights found for '{component.name}' in snapshot {self.path.name}"
            )

        # Load weights
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        component.model.load_state_dict(state)

        # Load optimizer if applicable
        if component.optimizer is not None and optim_path.exists():
            opt_state = torch.load(optim_path, map_location="cpu", weights_only=True)
            component.optimizer.load_state_dict(opt_state)

    def __repr__(self) -> str:
        return f"<useml.Snapshot id='{self.path.name}'>"
