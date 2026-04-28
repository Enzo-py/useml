import torch
import yaml
import shutil
import sys

from pathlib import Path
from typing import Any, Dict, Optional, Union

ARCHIVE_SOURCE_IGNORE_PATTERNS = [".git*", "__pycache__*", "*.pyc", ".DS_Store"]

import sys
import inspect
from pathlib import Path

def collect_source_from_components(components: dict, project_root: Path) -> set:
    """Collect source files from component instances."""
    paths = set()
    for obj in components.values():
        try:
            module = sys.modules.get(obj.model.__class__.__module__)
            p = Path(module.__file__).resolve()
            
            # Only save if within project
            if p.is_relative_to(project_root.resolve()):
                if p.exists():
                    paths.add(p)
        except (TypeError, ValueError):
            continue
    
    return paths

class SnapshotError(Exception):
    """Base exception for all snapshot-related errors."""
    pass


class SnapshotOverwriteError(SnapshotError):
    """Raised when an operation would overwrite an existing, non-empty snapshot."""
    pass

class SnapshotLimitExceededError(SnapshotError):
    """Raised when the number of files to archive exceeds the safety threshold."""
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
        meta: Dict[str, Any],
        archive_source: bool = True,
        inline_sources: Optional[Dict[str, str]] = None
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
                f"Conflict: Directory '{self.path}' is not empty."
            )

        # 2. Directory structure initialization
        dirs = {
            "weights": self.path / "weights",
            "configs": self.path / "configs",
            "optimizers": self.path / "optimizers",
            "source": self.path / "source",
        }

        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        # 3. Serialization configuration
        yaml_params = {"default_flow_style": False, "sort_keys": False}

        # 4. Component serialization loop
        for name, comp in components.items():
            # A. Model Weights
            # Note: On accède à .model car 'comp' est l'objet tracké (Component)
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


        # 5. Full Source Code Backup
        if archive_source:
            cwd = Path(sys.path[0]).resolve()
            # 1. On cible les fichiers VITAUX (ceux des objets trackés)
            core_dependencies = collect_source_from_components(components, cwd)
            
            # 2. On applique ton filtre de sécurité (Kill Path Vault)
            valid_paths = []
            for p in core_dependencies:
                # Si c'est dans un vault, on refuse
                if any((parent / ".useml_vault").exists() for parent in p.parents):
                    continue
                valid_paths.append(p)

            # 3. Copie physique
            for src in valid_paths:
                rel = src.relative_to(cwd)
                dest = dirs["source"] / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        
        
        # 6. Inline Sources Injection (Notebook & Object Inspection)
        # On écrit les sources extraites APRÈS la copie pour s'assurer qu'elles 
        # sont présentes même si le fichier source n'existe pas ou est différent.
        if inline_sources:
            for rel_path, content in inline_sources.items():
                target_file = dirs["source"] / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")

        # 7. System Manifest & User Metadata
        with open(self.path / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, **yaml_params)

        with open(self.path / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, **yaml_params)

        with open(dirs["source"] / "__init__.py", "w") as f:
            f.write("# snap_<id>/source/__init__.py")

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
    def metadata(self) -> dict:
        """User-defined metrics (accuracy, loss, etc.)."""
        if self._metadata is None:
            self._metadata = self._load_yaml("metadata.yaml")
        return self._metadata

    def _load_component(self, component: Any) -> None:
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
