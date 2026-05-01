import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


class SnapshotError(Exception):
    """Base exception for snapshot-related errors."""


class SnapshotOverwriteError(SnapshotError):
    """Raised when a save would overwrite a non-empty snapshot directory."""


class Snapshot:
    """Point-in-time record of a model state and its associated metadata.

    A Snapshot is a directory on disk with the following layout::

        snap_<timestamp>/
            weights/          # model state dicts (.pth)
            optimizers/       # optimizer state dicts (.pth)
            configs/          # per-component hyperparameters (.yaml)
            source/           # archived source code
            manifest.yaml     # component registry (deterministic)
            metadata.yaml     # message, timestamp, metrics
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Binds a Snapshot instance to an existing or future directory.

        Args:
            path: Directory that contains (or will contain) snapshot files.
        """
        self.path = Path(path)
        self.id = self.path.name
        self._manifest: Optional[Dict[str, Any]] = None
        self._metadata: Optional[Dict[str, Any]] = None

    def save(
        self,
        components: Dict[str, Any],
        manifest: Dict[str, Any],
        meta: Dict[str, Any],
        archive_source: bool = False,
        inline_sources: Optional[Dict[str, str]] = None,
    ) -> None:
        """Serialises all experiment data to disk.

        Args:
            components: Named Component objects to serialise.
            manifest: Deterministic component registry written to
                ``manifest.yaml``.
            meta: Contextual data (message, timestamp, metrics) written to
                ``metadata.yaml``.
            archive_source: When True, copies the current working directory
                into ``source/``. Defaults to False (inline_sources preferred).
            inline_sources: Mapping of relative paths to file contents,
                written verbatim into ``source/``. Used to archive model and
                loss source code extracted by the code extractor.

        Raises:
            SnapshotOverwriteError: If the snapshot directory already contains
                data.
        """
        if self.path.exists() and any(self.path.iterdir()):
            raise SnapshotOverwriteError(
                f"Directory '{self.path}' is not empty."
            )

        weights_dir = self.path / "weights"
        optimizers_dir = self.path / "optimizers"
        configs_dir = self.path / "configs"
        source_dir = self.path / "source"
        for d in (weights_dir, optimizers_dir, configs_dir, source_dir):
            d.mkdir(parents=True, exist_ok=True)

        yaml_opts = {"default_flow_style": False, "sort_keys": False}

        for name, comp in components.items():
            torch.save(
                comp.model.state_dict(), weights_dir / f"{name}.pth"
            )
            if comp.optimizer is not None:
                torch.save(
                    comp.optimizer.state_dict(),
                    optimizers_dir / f"{name}.pth",
                )
            if comp.config is not None:
                with open(
                    configs_dir / f"{name}.yaml", "w", encoding="utf-8"
                ) as f:
                    yaml.safe_dump(comp.config, f, **yaml_opts)

        if archive_source:
            self._copy_workdir(source_dir)

        if inline_sources:
            for rel_path, content in inline_sources.items():
                target = source_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        (source_dir / "__init__.py").write_text(
            "# useml snapshot source root\n"
        )

        with open(self.path / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, **yaml_opts)
        with open(self.path / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, **yaml_opts)

    # ------------------------------------------------------------------
    # Lazy-loaded properties
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> Dict:
        """System-level component registry, loaded lazily from disk."""
        if self._manifest is None:
            self._manifest = self._load_yaml("manifest.yaml")
        return self._manifest.get("info", {})

    @property
    def components(self) -> Dict:
        """Mapping of component names to their manifest entries."""
        if self._manifest is None:
            self._manifest = self._load_yaml("manifest.yaml")
        return self._manifest.get("components", {})

    @property
    def metadata(self) -> Dict:
        """User-defined context: message, timestamp, metrics."""
        if self._metadata is None:
            self._metadata = self._load_yaml("metadata.yaml")
        return self._metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, filename: str) -> Dict:
        file_path = self.path / filename
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _copy_workdir(self, dest: Path) -> None:
        """Copies source files from cwd into dest, skipping vault directories.

        Args:
            dest: Target directory inside the snapshot.
        """
        import shutil

        cwd = Path(sys.path[0]).resolve()
        for src in cwd.rglob("*.py"):
            if any(
                (p / ".useml_vault").exists() for p in src.parents
            ):
                continue
            rel = src.relative_to(cwd)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    def _load_component(self, component: Any) -> None:
        """Restores weights and optimizer state for a component in-place.

        Args:
            component: Component instance to populate.

        Raises:
            FileNotFoundError: If the weights file is missing.
        """
        weights_path = self.path / "weights" / f"{component.name}.pth"
        optim_path = self.path / "optimizers" / f"{component.name}.pth"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"No weights for '{component.name}' in {self.path.name}"
            )

        state = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        component.model.load_state_dict(state)

        if component.optimizer is not None and optim_path.exists():
            opt_state = torch.load(
                optim_path, map_location="cpu", weights_only=True
            )
            component.optimizer.load_state_dict(opt_state)

    def __repr__(self) -> str:
        return f"<useml.Snapshot id='{self.path.name}'>"
