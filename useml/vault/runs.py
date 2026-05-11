"""Project-as-workspace view: runs (training experiments) over snapshots.

A ``Run`` is the user-facing object for both *creating* a training experiment
and *reading* a finished one:

- ``project.runs.new(Encoder, "mnist", config=cfg)`` creates a runnable Run
- ``project.runs.best("val_loss")`` returns a ``RunRecord`` (read-only view)
- ``project.runs`` is the iterable ``RunsView``

Snapshots remain the underlying storage. Runs/records are a thin layer.
"""

from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import torch
import torch.nn as nn

from ..errors import ModelInstantiationError
from ..session.component import Component
from ..template.config import Config
from ..template.trainer import Trainer, _build_model

# ─────────────────────────────────────────────────────────────────────────────
# RunRecord — read-only view of a finished run
# ─────────────────────────────────────────────────────────────────────────────


class RunRecord:
    """Read-only view of a saved run (one snapshot from the user's POV).

    Most users obtain ``RunRecord``s via ``project.runs.best(...)`` or by
    iterating ``project.runs``. They are never constructed directly.
    """

    def __init__(self, snapshot, project_path: Path) -> None:
        self._snapshot = snapshot                      # Snapshot instance
        self._project_path = Path(project_path)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        """Snapshot id (folder name)."""
        return self._snapshot.path.name

    @property
    def metrics(self) -> dict:
        """Metrics dict, e.g. ``{"val_loss": 0.23, "epoch": 10}``."""
        return self._snapshot.metadata.get("metrics", {})

    @property
    def message(self) -> str:
        """Commit message / note for this run."""
        return self._snapshot.metadata.get("message", "")

    @property
    def date(self) -> str:
        """ISO timestamp of the run."""
        return self._snapshot.metadata.get("iso_date", "")

    @property
    def models(self) -> List[str]:
        """Names of the model components saved in this run."""
        return list(self._snapshot.components.keys())

    # ── Actions ────────────────────────────────────────────────────────────

    def load(self, name: Optional[str] = None, model: Optional[nn.Module] = None):
        """Load a model saved in this run.

        Args:
            name: Component name. If ``None`` and only one model exists in
                this run, that one is loaded.
            model: Pre-instantiated model. If ``None``, useml tries to import
                the class from its archived ``module_path``.

        Raises:
            ModelInstantiationError: If ``model`` is ``None`` and the class
                cannot be auto-instantiated.
        """
        names = self.models
        if name is None:
            if len(names) == 0:
                raise KeyError(f"No models saved in run '{self.id}'.")
            if len(names) > 1:
                raise ValueError(
                    f"Run '{self.id}' has multiple models ({names}); "
                    f"specify which one with .load(name=…)."
                )
            name = names[0]
        if name not in names:
            raise KeyError(
                f"Model '{name}' not in run '{self.id}'. Available: {names}"
            )

        weights_path = self._snapshot.path / "weights" / f"{name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        if model is None:
            model = self._try_instantiate(name)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    # ── Private ────────────────────────────────────────────────────────────

    def _try_instantiate(self, name: str):
        """Try to import + instantiate the model class for this component."""
        import importlib
        import inspect

        comp = self._snapshot.components.get(name, {})
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
                f"Pass an instance: run.load(model=MyModel()). [{exc}]"
            ) from exc

        try:
            sig = inspect.signature(cls.__init__)
            params = set(list(sig.parameters.keys())[1:])
            cfg_path = self._snapshot.path / "configs" / f"{name}.yaml"
            cfg_dict: dict = {}
            if cfg_path.exists():
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
            kwargs = {k: v for k, v in cfg_dict.items() if k in params}
            return cls(**kwargs) if kwargs else cls()
        except TypeError as exc:
            raise ModelInstantiationError(
                f"Cannot instantiate '{class_name}': {exc}. "
                f"Pass an instance: run.load(model={class_name}(<args>))."
            ) from exc

    def __repr__(self) -> str:
        m = self.metrics
        metric_str = " ".join(f"{k}={v}" for k, v in m.items())
        return f"<RunRecord {self.id[:24]} {metric_str or self.message}>"


# ─────────────────────────────────────────────────────────────────────────────
# RunsView — collection of runs in a project
# ─────────────────────────────────────────────────────────────────────────────


class RunsView:
    """All runs in a project — iterable, queryable, factory for new runs.

    .. code-block:: python

        run = project.runs.new(Encoder, "mnist", config=cfg)
        history = run.train()

        project.runs.best("val_loss").load()
        project.runs.latest
        project.runs.leaderboard()
    """

    def __init__(self, project) -> None:
        self._project = project    # Project instance (avoids circular import)

    # ── Read side ──────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[RunRecord]:
        # newest first (matches project.log())
        for snap in self._project.log():
            yield RunRecord(snap, self._project.path)

    def __len__(self) -> int:
        return len(self._project)

    def __getitem__(self, idx) -> RunRecord:
        return RunRecord(self._project[idx], self._project.path)

    @property
    def latest(self) -> RunRecord:
        """Most recent run."""
        if len(self) == 0:
            raise KeyError("No runs in this project.")
        return self[0]

    def best(self, metric: str, mode: str = "min") -> RunRecord:
        """Return the run with the best value for ``metric``."""
        scored = []
        for r in self:
            m = r.metrics
            val = m.get(metric)
            if val is None and not metric.startswith("val_"):
                val = m.get(f"val_{metric}")
            if val is None and metric.startswith("val_"):
                val = m.get(metric[4:])
            if val is not None:
                scored.append((float(val), r))
        if not scored:
            raise KeyError(
                f"No runs have metric '{metric}'. "
                f"Available metrics across runs: "
                f"{sorted({k for r in self for k in r.metrics})}"
            )
        scored.sort(key=lambda x: x[0], reverse=(mode == "max"))
        return scored[0][1]

    def filter(self, model: Optional[str] = None) -> "RunsView":
        """Return a filtered RunsView.

        Args:
            model: Keep only runs that contain a model component with this name.
        """
        # MVP: produce a lightweight filtered view (in-memory list)
        records = list(self)
        if model is not None:
            records = [r for r in records if model in r.models]
        return _ListBackedRunsView(self._project, records)

    def leaderboard(
        self,
        sort_by: str = "val_loss",
        mode: str = "min",
        top: int = 10,
    ) -> str:
        """Return a string-formatted leaderboard table sorted by ``sort_by``."""
        rows: list = []
        for r in self:
            m = r.metrics
            val = m.get(sort_by)
            if val is None and not sort_by.startswith("val_"):
                val = m.get(f"val_{sort_by}")
            rows.append((val, r))
        rows = [r for r in rows if r[0] is not None]
        rows.sort(key=lambda x: x[0], reverse=(mode == "max"))
        rows = rows[:top]

        if not rows:
            return f"(no runs with metric '{sort_by}')"

        lines = [f"Leaderboard — sorted by {sort_by} ({mode}):"]
        lines.append(f"  {'rank':<5}{sort_by:<14}{'models':<24}{'message':<30}")
        for i, (val, r) in enumerate(rows, 1):
            models_str = ",".join(r.models)
            lines.append(
                f"  #{i:<4}{val:<14.4f}{models_str[:22]:<24}{r.message[:28]:<30}"
            )
        return "\n".join(lines)

    # ── Write side ─────────────────────────────────────────────────────────

    def new(
        self,
        model_or_cls: Any = None,
        dataset: Any = None,
        *,
        config: Optional[Config] = None,
        name: Optional[str] = None,
    ) -> "Run":
        """Create a new training run inside this project.

        Args:
            model_or_cls: Model class (e.g. ``Encoder``) or instance. Optional —
                omit for Pattern B (register models manually).
            dataset: Built-in name, ``"hf:<name>"``, ``Dataset``, or
                ``DataBundle``. Optional — omit to pass loaders to ``train()``.
            config: Training configuration. Defaults to :class:`Config`.
            name: Override the auto-derived component name (defaults to the
                lowercased class name when ``model_or_cls`` is provided).

        Examples::

            run = project.runs.new(Encoder, "mnist", config=cfg)
            run.train()

            run = project.runs.new(Encoder, config=cfg)
            run.train(train_loader, val_loader)

            run = project.runs.new(config=cfg)            # Pattern B
            run.register("G", Generator(), optimizer=opt_G)
            run.register("D", Discriminator(), optimizer=opt_D)
            for epoch in run.epochs(50):
                ...
        """
        return Run(
            project=self._project,
            model_or_cls=model_or_cls,
            dataset=dataset,
            config=config,
            name=name,
        )

    def __repr__(self) -> str:
        n = len(self)
        return f"<RunsView project='{self._project.path.name}' runs={n}>"


class _ListBackedRunsView(RunsView):
    """Internal: a RunsView populated from an explicit list (for filter())."""

    def __init__(self, project, records: List[RunRecord]) -> None:
        super().__init__(project)
        self._records = records

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx) -> RunRecord:
        return self._records[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Run — the new write-side training object (replaces public Trainer)
# ─────────────────────────────────────────────────────────────────────────────


class Run(Trainer):
    """A training experiment scoped to a specific project.

    Inherits all of :class:`Trainer`'s primitives (``register``, ``epochs``,
    ``update``, ``epoch_end``, ``resume``) and adds:

    - explicit project context (no global session lookup)
    - automatic name inference from class name
    - automatic data loading when a dataset is given to ``runs.new()``
    """

    def __init__(
        self,
        project,
        model_or_cls: Any = None,
        dataset: Any = None,
        config: Optional[Config] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        self._project_obj = project        # avoid clashing with Trainer attrs
        self._dataset = dataset
        self._auto_name: Optional[str] = None

        # Bundle metadata threading (mirrors run_training)
        self._maybe_bind_bundle(dataset)

        if model_or_cls is not None:
            if isinstance(model_or_cls, type) and issubclass(model_or_cls, nn.Module):
                model = _build_model(model_or_cls, self.config)
                self._auto_name = name or model_or_cls.__name__.lower()
                self.register(self._auto_name, model)
            elif isinstance(model_or_cls, nn.Module):
                self._auto_name = name or type(model_or_cls).__name__.lower()
                self.register(self._auto_name, model_or_cls)
            else:
                raise TypeError(
                    f"model_or_cls must be a Module class or instance, "
                    f"got {type(model_or_cls).__name__}."
                )

    # ── Bundle support ─────────────────────────────────────────────────────

    def _maybe_bind_bundle(self, dataset: Any) -> None:
        from ..dataset import DataBundle
        if isinstance(dataset, DataBundle):
            self.bundle_meta = dataset.to_meta_dict()
            src = dataset.transform_source()
            self._bundle_transform_source = src
            self._bundle_transform_key = dataset.inline_source_key() if src else None

    # ── Train (overrides Trainer.run for Level-0 ergonomics) ──────────────

    def train(
        self,
        train_loader: Any = None,
        val_loader: Any = None,
        step_fn: Optional[Callable] = None,
        metrics: Optional[dict] = None,
        scheduler: Optional[Any] = None,
        resume_from: Optional[str] = None,
    ) -> dict:
        """Run the training loop.

        - With a ``dataset`` already configured (via ``runs.new(..., "mnist")``),
          loaders are built automatically and you can call ``run.train()``.
        - Otherwise pass ``train_loader, val_loader`` explicitly.
        - For custom training (Pattern B), don't call ``train()`` — use
          ``epochs()/update()/epoch_end()`` directly.
        """
        if train_loader is None and self._dataset is not None:
            from ..dataset import load_dataset
            train_loader, val_loader = load_dataset(self._dataset, self.config)

        if train_loader is None:
            raise ValueError(
                "Run.train() needs either a dataset (passed to runs.new(...)) "
                "or explicit (train_loader, val_loader). For custom loops use "
                "run.epochs() / run.update() / run.epoch_end() instead."
            )

        if step_fn is not None:
            _device = self.config.device
            _fn = step_fn
            self.step = lambda batch: _fn(self.model, batch, _device)

        if metrics:
            self.metrics = metrics

        if scheduler is not None:
            self.scheduler = (
                scheduler(self.optimizer)
                if callable(scheduler) and not hasattr(scheduler, "step")
                else scheduler
            )

        return self.run(train_loader, val_loader, resume_from=resume_from)

    # ── Override Trainer's session-based checkpoint with project-based ────

    def _do_checkpoint(self, epoch: int, monitor: dict) -> None:
        """Commit all registered models atomically — directly to the project,
        bypassing the global session.
        """
        components = {}
        for n, model in self._models.items():
            components[n] = Component(
                name=n,
                model=model,
                config=self.config,
                optimizer=self._optimizers.get(n),
            )

        metric_kwargs = {
            (k if k.startswith("val_") else f"val_{k}"): round(v, 6)
            for k, v in monitor.items()
        }

        inline = (
            {self._bundle_transform_key: self._bundle_transform_source}
            if self._bundle_transform_source and self._bundle_transform_key
            else None
        )

        self._project_obj.commit(
            message=f"epoch {epoch}",
            components=components,
            bundle_meta=self.bundle_meta,
            bundle_inline_source=inline,
            epoch=epoch,
            **metric_kwargs,
        )

    # ── resume() override (Trainer's version uses session._resolve_…) ─────

    def resume(self, snapshot_tag: str = "\\latest") -> int:
        """Load weights + optimizer states from a snapshot in this project."""
        from ..vault.snapshot import Snapshot as _Snapshot

        snap_path = self._project_obj.resolve_snapshot(snapshot_tag)

        for name, model in self._models.items():
            weights_path = snap_path / "weights" / f"{name}.pth"
            if weights_path.exists():
                state = torch.load(
                    weights_path,
                    map_location=self.config.device,
                    weights_only=True,
                )
                model.load_state_dict(state)
            opt = self._optimizers.get(name)
            opt_path = snap_path / "optimizers" / f"{name}.pth"
            if opt is not None and opt_path.exists():
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
                opt.load_state_dict(opt_state)

        snap = _Snapshot(snap_path)
        meta = snap.metadata
        saved_metrics = meta.get("metrics", {})
        saved_epoch = saved_metrics.get("epoch", 0)

        if self.config.checkpoint_strategy == "best" and saved_metrics:
            key = self.config.checkpoint_metric
            val = saved_metrics.get(key) or saved_metrics.get(key.replace("val_", ""))
            if val is not None:
                self._best_ckpt_value = float(val)

        self._epoch = saved_epoch
        remaining = self.config.epochs - saved_epoch
        print(
            f"  ↩  Resumed '{snap_path.name}' — "
            f"epoch {saved_epoch}/{self.config.epochs} done, "
            f"{remaining} remaining"
        )
        return saved_epoch
