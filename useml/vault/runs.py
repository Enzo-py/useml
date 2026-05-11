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

    @property
    def history(self) -> dict:
        """Per-epoch metric history: ``{"train_loss": [...], "val_loss": [...]}``.

        Returns an empty dict if no history was recorded (old snapshots).
        """
        h_path = self._snapshot.path / "history.yaml"
        if not h_path.exists():
            return {}
        import yaml
        with open(h_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def history_df(self):
        """Return :attr:`history` as a pandas ``DataFrame`` (requires pandas).

        Each column is a metric; the index is 1-based epoch numbers.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for history_df(). Run: pip install pandas"
            )
        h = self.history
        if not h:
            return pd.DataFrame()
        df = pd.DataFrame(h)
        df.index = range(1, len(df) + 1)
        df.index.name = "epoch"
        return df

    def plot(self, metrics=None, *, ax=None, label_prefix: str = "", title: str = None):
        """Plot training curves for this run.

        Train and val variants of the same base metric share a color;
        train uses a solid line and val uses a dashed line.

        Args:
            metrics: Metric name(s) to plot. ``None`` plots all available.
            ax: Existing :class:`matplotlib.axes.Axes` to draw on.
                When ``None`` (default), a new figure is created.
            label_prefix: String prepended to each legend label (useful when
                combining several runs on the same axes).
            title: Figure title. Defaults to the run id.

        Returns:
            ``(fig, ax)`` where *fig* is ``None`` when *ax* was supplied.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If no matching metrics are found.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot(). Run: pip install matplotlib"
            )

        h = self.history
        if not h:
            raise ValueError(f"No history data found for run '{self.id}'.")

        if metrics is None:
            keys = list(h.keys())
        elif isinstance(metrics, str):
            keys = [m for m in [metrics] if m in h]
        else:
            keys = [m for m in metrics if m in h]

        if not keys:
            raise ValueError(
                f"No matching metrics in history. Available: {list(h.keys())}"
            )

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = None

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        base_to_color: dict = {}
        color_idx = 0

        for k in keys:
            if k.startswith("train_"):
                base = k[6:]
            elif k.startswith("val_"):
                base = k[4:]
            else:
                base = k
            if base not in base_to_color:
                base_to_color[base] = color_cycle[color_idx % len(color_cycle)]
                color_idx += 1

            vals = h[k]
            epochs = list(range(1, len(vals) + 1))
            linestyle = "--" if k.startswith("val_") else "-"
            label = f"{label_prefix}{k}" if label_prefix else k
            ax.plot(epochs, vals, linestyle=linestyle,
                    color=base_to_color[base], label=label)

        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if created_fig:
            ax.set_title(title or self.id[:30])
            fig.tight_layout()

        return fig, ax

    # ── Actions ────────────────────────────────────────────────────────────

    def _resolve_name(self, name: Optional[str]) -> str:
        """Resolve component name, defaulting to the only one when there is one."""
        names = self.models
        if name is None:
            if not names:
                raise KeyError(f"No models saved in run '{self.id}'.")
            if len(names) > 1:
                raise ValueError(
                    f"Run '{self.id}' has multiple models ({names}); "
                    f"specify which one."
                )
            return names[0]
        if name not in names:
            raise KeyError(
                f"Model '{name}' not in run '{self.id}'. Available: {names}"
            )
        return name

    def state_dict(self, name: Optional[str] = None) -> dict:
        """Return the raw weights dict for a component — no model needed.

        Useful when the model requires complex constructor arguments that
        cannot be auto-inferred from the snapshot config.

        Args:
            name: Component name. Omit when the run has only one model.

        Returns:
            ``OrderedDict`` as returned by ``torch.load``.
        """
        name = self._resolve_name(name)
        weights_path = self._snapshot.path / "weights" / f"{name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")
        return torch.load(weights_path, map_location="cpu", weights_only=True)

    def instantiate(self, name: Optional[str] = None) -> nn.Module:
        """Instantiate the model class from the archived config — without loading weights.

        Useful to inspect the architecture or create a fresh copy at the
        same hyperparameters without restoring the trained weights.

        Args:
            name: Component name. Omit when the run has only one model.

        Returns:
            A newly constructed (untrained) model instance.

        Raises:
            ModelInstantiationError: If the class cannot be imported or its
                constructor requires arguments not stored in the config YAML.
                In that case pass a pre-built instance to :meth:`load` instead.
        """
        name = self._resolve_name(name)
        return self._try_instantiate(name)

    def load(self, name: Optional[str] = None, model: Optional[nn.Module] = None) -> nn.Module:
        """Load trained weights into a model.

        Three usage patterns:

        .. code-block:: python

            # 1. Auto-instantiate + load weights (works when all constructor
            #    args are stored as Config custom fields)
            model = record.load("encoder")

            # 2. Provide your own instance — always works
            model = record.load("encoder", model=Encoder(hidden=256))

            # 3. Just the weights dict (no model class needed)
            sd = record.state_dict("encoder")
            model.load_state_dict(sd)

        Args:
            name: Component name. Omit when the run has only one model.
            model: Pre-instantiated model. When ``None`` useml tries to
                import and construct the class from the archived config YAML.

        Raises:
            ModelInstantiationError: When ``model`` is ``None`` and the class
                cannot be auto-instantiated (e.g. requires object-type args).
                Solution: pass ``model=YourClass(<args>)`` explicitly.
        """
        name = self._resolve_name(name)
        weights_path = self._snapshot.path / "weights" / f"{name}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        if model is None:
            model = self._try_instantiate(name)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    # ── Private ────────────────────────────────────────────────────────────

    def _try_instantiate(self, name: str) -> nn.Module:
        """Import the model class and construct it from the archived config.

        Raises ``ModelInstantiationError`` with an actionable message when
        the class cannot be imported or has required constructor arguments
        that were not stored in the config YAML.
        """
        import importlib
        import inspect

        comp = self._snapshot.components.get(name, {})
        module_path = comp.get("module_path", "")
        class_name = comp.get("class_name", "")

        # ── 1. Import the class ───────────────────────────────────────────
        try:
            if module_path == "__main__":
                import __main__ as module
            else:
                module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except Exception as exc:
            raise ModelInstantiationError(
                f"Cannot import {class_name!r} from module '{module_path}'. "
                f"Pass a pre-built instance: record.load(model={class_name}()). "
                f"[{exc}]"
            ) from exc

        # ── 2. Read config YAML ───────────────────────────────────────────
        cfg_path = self._snapshot.path / "configs" / f"{name}.yaml"
        cfg_dict: dict = {}
        if cfg_path.exists():
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}

        # ── 3. Identify which config keys match constructor params ─────────
        sig = inspect.signature(cls.__init__)
        all_params = list(sig.parameters.values())[1:]   # skip 'self'
        required = [
            p.name for p in all_params
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        matched_kwargs = {
            p.name: cfg_dict[p.name]
            for p in all_params
            if p.name in cfg_dict
        }
        missing = [r for r in required if r not in matched_kwargs]

        if missing:
            raise ModelInstantiationError(
                f"Cannot auto-instantiate '{class_name}': required argument(s) "
                f"{missing} are not stored in the snapshot config.\n"
                f"  → Store them before training:  Config(..., {missing[0]}=<value>)\n"
                f"  → Or provide an instance:       record.load('{name}', model={class_name}(<args>))\n"
                f"  → Or load weights only:         record.state_dict('{name}')"
            )

        try:
            return cls(**matched_kwargs)
        except TypeError as exc:
            raise ModelInstantiationError(
                f"Cannot instantiate '{class_name}' even with config kwargs "
                f"{list(matched_kwargs)}: {exc}.\n"
                f"  → Pass an instance: record.load('{name}', model={class_name}(<args>))"
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

    def plot(
        self,
        metric: str,
        *,
        mode: str = "min",
        top: Optional[int] = None,
        title: Optional[str] = None,
    ):
        """Plot one metric across all (or top N) runs on a single figure.

        Args:
            metric: Metric to compare, e.g. ``"val_loss"`` or ``"train_accuracy"``.
            mode: ``"min"`` or ``"max"`` — controls ranking when *top* is set.
            top: If set, show only the top N runs (ranked by final metric value).
            title: Figure title. Defaults to ``"<metric> across runs"``.

        Returns:
            ``(fig, ax)`` tuple.

        Raises:
            ImportError: If matplotlib is not installed.
            KeyError: If no run has history for *metric*.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot(). Run: pip install matplotlib"
            )

        candidates = []
        for r in self:
            h = r.history
            if metric in h and h[metric]:
                candidates.append((h[metric][-1], r, h[metric]))

        if not candidates:
            available = sorted({k for r in self for k in r.history})
            raise KeyError(
                f"No runs have history for metric '{metric}'. "
                f"Available: {available}"
            )

        candidates.sort(key=lambda x: x[0], reverse=(mode == "max"))
        if top is not None:
            candidates = candidates[:top]

        fig, ax = plt.subplots(figsize=(9, 4))
        for final_val, r, vals in candidates:
            epochs = list(range(1, len(vals) + 1))
            label = f"{r.id[5:21]} (final {final_val:.4f})"
            ax.plot(epochs, vals, label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} across runs")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

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

        result = self.run(train_loader, val_loader, resume_from=resume_from)

        if getattr(self, "_was_interrupted", False) and self._epoch > 0:
            monitor = self._last_val_metrics or self._last_train_metrics
            if monitor:
                print("  💾  Saving current model state...")
                try:
                    self._do_checkpoint(self._epoch, monitor)
                except Exception as exc:
                    print(f"  ⚠  Emergency save failed: {exc}")

        self._write_history_to_latest()
        return result

    def epochs(self, n=None):
        """Epoch iterator — writes complete history when the loop ends or is interrupted."""
        try:
            yield from super().epochs(n)
        except KeyboardInterrupt:
            print(f"\n  ⏸  Training interrupted at epoch {self._epoch}.")
            monitor = self._last_val_metrics or self._last_train_metrics
            if monitor and self._epoch > 0:
                print("  💾  Saving current model state...")
                try:
                    self._do_checkpoint(self._epoch, monitor)
                except Exception as exc:
                    print(f"  ⚠  Emergency save failed: {exc}")
            raise   # let Jupyter show "KeyboardInterrupt" (cell stops, kernel lives)
        finally:
            self._write_history_to_latest()

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

    def _write_history_to_latest(self) -> None:
        """Write the full per-epoch history to the most recent snapshot.

        Called once at the end of training — not at each checkpoint — so the
        latest snapshot always holds the complete epoch-by-epoch record.
        """
        import yaml as _yaml

        snaps = self._project_obj.log()
        if not snaps or not self._history:
            return
        history_path = snaps[0].path / "history.yaml"
        with open(history_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(dict(self._history), f, default_flow_style=False)

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
