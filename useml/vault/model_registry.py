"""Project-as-workspace view: model versions across snapshots.

Provides ``project.models["encoder"].latest.load()`` etc. — a component-centric
view over snapshots. Storage stays the same; this is a read-only layer.
"""

from pathlib import Path
from typing import Dict, Iterator, Optional

import yaml

from ..errors import ModelInstantiationError


class ModelVersion:
    """One tagged version of one component inside a project.

    Lazily reads metadata, config, and weights from the underlying snapshot.
    """

    def __init__(
        self,
        name: str,
        label: str,
        snapshot_id: str,
        project_path: Path,
    ) -> None:
        self.name = name                      # component name: "encoder"
        self.label = label                    # version label: "v3"
        self.snapshot_tag = snapshot_id       # "snap_20260502_…"
        self._project_path = Path(project_path)

    # ── Lazy properties ────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict:
        """Metrics dict from the snapshot metadata (e.g. ``val_loss``)."""
        return self._meta().get("metrics", {})

    @property
    def config(self) -> dict:
        """Config dict saved alongside this component."""
        cfg_path = self._snap_path() / "configs" / f"{self.name}.yaml"
        if not cfg_path.exists():
            return {}
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def message(self) -> str:
        """Commit message of the underlying snapshot."""
        return self._meta().get("message", "")

    @property
    def date(self) -> str:
        """ISO timestamp of the snapshot."""
        return self._meta().get("iso_date", "")

    # ── Actions ────────────────────────────────────────────────────────────

    def load(self, model=None):
        """Load weights into a model instance.

        Args:
            model: Pre-instantiated model. If ``None``, useml tries to import
                the class from its archived ``module_path`` and instantiate it.

        Raises:
            ModelInstantiationError: If ``model`` is ``None`` and the class
                cannot be auto-instantiated.
            FileNotFoundError: If the underlying weights file is missing.
        """
        import torch

        weights_path = self._snap_path() / "weights" / f"{self.name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file missing for {self}: {weights_path}"
            )
        if model is None:
            model = self._try_instantiate()
        state = torch.load(weights_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        return model

    # ── Private helpers ────────────────────────────────────────────────────

    def _snap_path(self) -> Path:
        return self._project_path / self.snapshot_tag

    def _meta(self) -> dict:
        meta_path = self._snap_path() / "metadata.yaml"
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _manifest(self) -> dict:
        manifest_path = self._snap_path() / "manifest.yaml"
        if not manifest_path.exists():
            return {}
        with open(manifest_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _try_instantiate(self):
        """Attempt to import the model class and instantiate it.

        Mirrors the behaviour of ``Session._import_from_workdir`` /
        ``Session._instantiate_model`` so this works without a session.
        """
        import importlib
        import inspect

        manifest = self._manifest()
        comp = manifest.get("components", {}).get(self.name, {})
        module_path = comp.get("module_path", "")
        class_name = comp.get("class_name", "")
        try:
            if module_path == "__main__":
                import __main__ as module
            else:
                module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except Exception as exc:
            raise ModelInstantiationError(
                f"Cannot import {class_name} from '{module_path}'. "
                f"Pass an instance explicitly: version.load(MyModel()). [{exc}]"
            ) from exc

        # Try to forward config keys as kwargs (matches Session._instantiate_model)
        cfg = self.config
        try:
            sig = inspect.signature(cls.__init__)
            params = set(list(sig.parameters.keys())[1:])  # skip self
            kwargs = {k: v for k, v in cfg.items() if k in params}
            return cls(**kwargs) if kwargs else cls()
        except TypeError as exc:
            raise ModelInstantiationError(
                f"Cannot instantiate '{class_name}': {exc}. "
                f"Pass an instance: version.load({class_name}(<args>))."
            ) from exc

    def __repr__(self) -> str:
        return (
            f"<ModelVersion {self.name}@{self.label} "
            f"snap={self.snapshot_tag[:24]}>"
        )


class ModelRegistry:
    """All versions of one named component in a project.

    Dict-like access (``registry["v2"]``) plus ``.latest`` and ``.best()``.
    Versions are ordered oldest→newest.
    """

    def __init__(self, name: str, versions: Dict[str, ModelVersion]) -> None:
        self.name = name
        # ordered insertion = oldest→newest
        self._versions: Dict[str, ModelVersion] = dict(versions)

    @property
    def latest(self) -> ModelVersion:
        """Most recently committed version."""
        if not self._versions:
            raise KeyError(f"No versions registered for '{self.name}'.")
        return list(self._versions.values())[-1]

    def best(self, metric: str, mode: str = "min") -> ModelVersion:
        """Return the version with the best value for ``metric``.

        Args:
            metric: Metric key (e.g. ``"val_loss"`` or ``"loss"`` — both
                tried).
            mode: ``"min"`` (lower is better) or ``"max"``.
        """
        scored = []
        for v in self._versions.values():
            m = v.metrics
            val = m.get(metric)
            if val is None and not metric.startswith("val_"):
                val = m.get(f"val_{metric}")
            if val is None and metric.startswith("val_"):
                val = m.get(metric[4:])
            if val is not None:
                scored.append((float(val), v))
        if not scored:
            raise KeyError(
                f"No versions of '{self.name}' have metric '{metric}'. "
                f"Available metrics: "
                f"{sorted({k for v in self._versions.values() for k in v.metrics})}"
            )
        scored.sort(key=lambda x: x[0], reverse=(mode == "max"))
        return scored[0][1]

    def __getitem__(self, label: str) -> ModelVersion:
        if label not in self._versions:
            raise KeyError(
                f"Version '{label}' not found for '{self.name}'. "
                f"Available: {list(self._versions)}"
            )
        return self._versions[label]

    def __iter__(self) -> Iterator[ModelVersion]:
        return iter(self._versions.values())

    def __len__(self) -> int:
        return len(self._versions)

    def __contains__(self, label: object) -> bool:
        return label in self._versions

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry '{self.name}' "
            f"versions={list(self._versions)}>"
        )


class ModelsView:
    """All ModelRegistries in a project — dict-like view."""

    def __init__(self, project_path: Path, versions_index: dict) -> None:
        self._project_path = Path(project_path)
        self._registries: Dict[str, ModelRegistry] = {}
        for comp_name, versions in versions_index.items():
            mv_dict = {
                label: ModelVersion(
                    comp_name, label, snap_id, self._project_path
                )
                for label, snap_id in versions.items()
            }
            self._registries[comp_name] = ModelRegistry(comp_name, mv_dict)

    def __getitem__(self, name: str) -> ModelRegistry:
        if name not in self._registries:
            raise KeyError(
                f"No model named '{name}' in this project. "
                f"Available: {list(self._registries)}"
            )
        return self._registries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._registries)

    def __len__(self) -> int:
        return len(self._registries)

    def __contains__(self, name: object) -> bool:
        return name in self._registries

    def keys(self):
        return self._registries.keys()

    def values(self):
        return self._registries.values()

    def items(self):
        return self._registries.items()

    def __repr__(self) -> str:
        if not self._registries:
            return "<ModelsView empty>"
        summary = ", ".join(
            f"{n}({len(r)})" for n, r in self._registries.items()
        )
        return f"<ModelsView {summary}>"
