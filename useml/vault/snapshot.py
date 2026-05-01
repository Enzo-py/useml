import sys

import torch
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from ..errors import SnapshotOverwriteError, WeightsNotFoundError

# Backward-compatible alias
SnapshotError = SnapshotOverwriteError


class Snapshot:
    """Represents a point-in-time record of model state and associated metadata.

    A Snapshot is a directory containing serialised model weights (.pth) and
    configuration files (.yaml). Metadata is loaded lazily.
    """

    def __init__(self, path) -> None:
        """Initializes a Snapshot instance.

        Args:
            path: Directory containing snapshot files.
        """
        self.path = Path(path)
        self._manifest: Optional[Dict[str, Any]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self.id = self.path.name

    def save(
        self,
        components: Dict[str, Any],
        manifest: Dict[str, Any],
        meta: Dict[str, Any],
        archive_source: bool = True,
        inline_sources: Optional[Dict[str, str]] = None,
    ) -> None:
        """Serialises components, manifest, and metadata to disk.

        Args:
            components: Named Component objects to serialise.
            manifest: System-level data (component registry, framework version).
            meta: User-defined metrics and commit message.
            archive_source: When True, copies the working directory into source/.
            inline_sources: Mapping of relative paths to file contents to write
                into source/ (used for notebook / in-memory classes).

        Raises:
            SnapshotOverwriteError: If the snapshot directory already contains data.
        """
        if self.path.exists() and any(self.path.iterdir()):
            raise SnapshotOverwriteError(
                f"Directory '{self.path}' is not empty — refusing to overwrite."
            )

        dirs = {
            "weights": self.path / "weights",
            "configs": self.path / "configs",
            "optimizers": self.path / "optimizers",
            "source": self.path / "source",
        }
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        yaml_params = {"default_flow_style": False, "sort_keys": False}

        for name, comp in components.items():
            torch.save(comp.model.state_dict(), dirs["weights"] / f"{name}.pth")

            if comp.optimizer is not None:
                torch.save(
                    comp.optimizer.state_dict(),
                    dirs["optimizers"] / f"{name}.pth",
                )

            if comp.config is not None:
                with open(dirs["configs"] / f"{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(comp.config, f, **yaml_params)

        if archive_source:
            self._copy_workdir(dirs["source"])

        if inline_sources:
            for rel_path, content in inline_sources.items():
                target = dirs["source"] / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        with open(dirs["source"] / "__init__.py", "w") as f:
            f.write("# snap source package\n")

        with open(self.path / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, **yaml_params)

        with open(self.path / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, **yaml_params)

    def _copy_workdir(self, source_dir: Path) -> None:
        """Copies the current working directory into source_dir.

        Args:
            source_dir: Destination directory inside the snapshot.
        """
        import shutil

        cwd = Path(sys.path[0]).resolve()
        ignore = shutil.ignore_patterns(".git*", "__pycache__*", "*.pyc", ".DS_Store")
        for item in cwd.iterdir():
            dest = source_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=ignore, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

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
        """User-defined metrics and commit message."""
        if self._metadata is None:
            self._metadata = self._load_yaml("metadata.yaml")
        return self._metadata

    def _load_component(self, component: Any) -> None:
        """Restores weights and optimizer state for a specific component.

        Args:
            component: The Component instance to populate.

        Raises:
            WeightsNotFoundError: If the component's weights file is missing.
        """
        weights_path = self.path / "weights" / f"{component.name}.pth"
        optim_path = self.path / "optimizers" / f"{component.name}.pth"

        if not weights_path.exists():
            raise WeightsNotFoundError(
                f"No weights found for '{component.name}' in snapshot '{self.path.name}'."
            )

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        component.model.load_state_dict(state)

        if component.optimizer is not None and optim_path.exists():
            opt_state = torch.load(optim_path, map_location="cpu", weights_only=True)
            component.optimizer.load_state_dict(opt_state)

    def __repr__(self) -> str:
        return f"<useml.Snapshot id='{self.path.name}'>"
