import inspect
import time
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Type

import torch
import torch.nn as nn

from .config import Config
from ..dataset import load_dataset
from ..errors import UnknownLossError, InvalidLossTypeError, UnknownOptimizerError


_BUILTIN_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse":           nn.MSELoss,
    "bce":           nn.BCEWithLogitsLoss,
    "l1":            nn.L1Loss,
}

_BUILTIN_OPTIMIZERS = {
    "adam":  lambda p, lr: torch.optim.Adam(p, lr=lr),
    "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr),
    "sgd":   lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
}


def _build_loss(config: Config) -> nn.Module:
    loss = config.loss
    if isinstance(loss, str):
        key = loss.lower()
        if key not in _BUILTIN_LOSSES:
            raise UnknownLossError(
                f"Unknown loss '{loss}'. Built-in choices: {list(_BUILTIN_LOSSES)}."
            )
        return _BUILTIN_LOSSES[key]()
    if isinstance(loss, nn.Module):
        return loss
    if isinstance(loss, type) and issubclass(loss, nn.Module):
        return loss()
    if callable(loss):
        _fn = loss
        class _WrappedLoss(nn.Module):
            def forward(self, pred, target):
                return _fn(pred, target)
        return _WrappedLoss()
    raise InvalidLossTypeError(
        f"config.loss must be a str, nn.Module subclass, nn.Module instance, "
        f"or callable. Got {type(loss).__name__}."
    )


def _build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    key = config.optimizer.lower()
    if key not in _BUILTIN_OPTIMIZERS:
        raise UnknownOptimizerError(
            f"Unknown optimizer '{config.optimizer}'. Choices: {list(_BUILTIN_OPTIMIZERS)}."
        )
    return _BUILTIN_OPTIMIZERS[key](model.parameters(), config.lr)


def _build_model(model_cls: Type[nn.Module], config: Config) -> nn.Module:
    sig = inspect.signature(model_cls.__init__)
    for param in list(sig.parameters.keys())[1:]:
        if param in ("config", "cfg"):
            return model_cls(**{param: config})
    return model_cls()


def _batch_size(batch: Any) -> int:
    first = batch[0] if isinstance(batch, (list, tuple)) else batch
    return first.size(0)


def _improved(current: float, best: float, mode: str) -> bool:
    return current < best if mode == "min" else current > best


class Trainer:
    """Training bookkeeper — works in two complementary patterns.

    **Pattern A — convenience (standard supervised loop):**

    .. code-block:: python

        trainer = Trainer(model, config)
        trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=20)
        trainer.metrics   = {"accuracy": accuracy_fn}
        history = trainer.run(train_loader, val_loader)

    **Pattern B — custom loop (GAN, diffusion, multi-optimizer, …):**

    .. code-block:: python

        trainer = Trainer(config)
        trainer.register("G", G)
        trainer.register("D", D)

        for epoch in trainer.epochs(EPOCHS):
            for batch in train_loader:
                g_loss, d_loss = adversarial_step(G, D, opt_G, opt_D, batch)
                trainer.update(g_loss=g_loss, d_loss=d_loss)

            for batch in val_loader:           # optional
                trainer.update_val(g_loss=eval_step(G, batch))

            trainer.epoch_end(epoch)

        print(trainer.history)

    In Pattern B the Trainer never touches your optimizers or backward passes.
    It only accumulates metrics, aggregates per epoch, logs, steps the
    scheduler (if set), decides checkpoints, and checks early stopping.
    All registered models are committed **atomically** on each checkpoint.

    Config fields honoured in both patterns:
    ``checkpoint_strategy``, ``checkpoint_metric``, ``checkpoint_mode``,
    ``checkpoint_every``, ``early_stop_patience``, ``early_stop_metric``,
    ``early_stop_mode``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_or_config: Any = None,
        config: Optional[Config] = None,
    ) -> None:
        # Dual signature:
        #   Trainer(config)        → Pattern B (new-style)
        #   Trainer(model, config) → Pattern A (backward-compat)
        if isinstance(model_or_config, nn.Module):
            _primary = model_or_config
            self.config = config if config is not None else Config()
        elif isinstance(model_or_config, Config):
            _primary = None
            self.config = model_or_config
        elif model_or_config is None:
            _primary = None
            self.config = config if config is not None else Config()
        else:
            raise TypeError(
                f"First argument must be nn.Module or Config, "
                f"got {type(model_or_config).__name__}."
            )

        # Model / optimizer registry
        self._models: Dict[str, nn.Module] = {}
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

        # Level-2 user-assignable
        self.scheduler: Optional[Any] = None
        self.scheduler_per_batch: bool = False
        self.metrics: Dict[str, Callable] = {}    # used by run()

        # Level-1 DataBundle metadata threading
        self.bundle_meta: Optional[dict] = None
        self._bundle_transform_source: Optional[str] = None
        self._bundle_transform_key: Optional[str] = None

        # Lazy criterion (built on first access inside step())
        self._criterion: Optional[nn.Module] = None

        # Per-epoch metric accumulation
        self._train_totals: Dict[str, float] = {}
        self._train_count: int = 0
        self._val_totals: Dict[str, float] = {}
        self._val_count: int = 0

        # History and checkpoint state
        self._history: Dict[str, list] = {}
        self._epoch: int = 0
        self._stopped: bool = False
        self._best_ckpt_value: Optional[float] = None
        self._es_best: Optional[float] = None
        self._patience_counter: int = 0
        self._last_val_metrics: Dict[str, float] = {}
        self._last_train_metrics: Dict[str, float] = {}

        self._session = self._get_active_session()

        # Auto-register if a model was passed (Pattern A backward compat)
        if _primary is not None:
            self.register("model", _primary)

    # ------------------------------------------------------------------
    # Backward-compat properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> Optional[nn.Module]:
        """Primary model — 'model' key, or first registered."""
        return self._models.get("model") or (
            next(iter(self._models.values())) if self._models else None
        )

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Primary optimizer — mirrors :attr:`model`."""
        return self._optimizers.get("model") or (
            next(iter(self._optimizers.values())) if self._optimizers else None
        )

    @property
    def criterion(self) -> nn.Module:
        """Loss module — built lazily so step_fn users never trigger _build_loss."""
        if self._criterion is None:
            self._criterion = _build_loss(self.config)
        return self._criterion

    @criterion.setter
    def criterion(self, value: nn.Module) -> None:
        self._criterion = value

    # ------------------------------------------------------------------
    # Model registration  (Pattern B)
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> "Trainer":
        """Register a model for atomic checkpointing.

        An optimizer is auto-created from ``config`` when not supplied.
        Returns ``self`` for chaining::

            trainer.register("G", G).register("D", D)
        """
        self._models[name] = model.to(self.config.device)
        self._optimizers[name] = (
            optimizer if optimizer is not None
            else _build_optimizer(model, self.config)
        )
        return self

    # ------------------------------------------------------------------
    # Metric accumulation  (Pattern B)
    # ------------------------------------------------------------------

    def update(self, n: int = 1, **metrics) -> None:
        """Accumulate **training-phase** metrics for the current batch.

        Args:
            n: Batch size (for weighted averaging). Default 1 = simple mean.
            **metrics: Scalar values, e.g. ``g_loss=0.4, d_loss=0.7``.
        """
        for k, v in metrics.items():
            self._train_totals[k] = self._train_totals.get(k, 0.0) + float(v) * n
        self._train_count += n

    def update_val(self, n: int = 1, **metrics) -> None:
        """Accumulate **validation-phase** metrics for the current batch.

        Args:
            n: Batch size (for weighted averaging). Default 1 = simple mean.
            **metrics: Scalar values, e.g. ``val_loss=0.5``.
        """
        for k, v in metrics.items():
            self._val_totals[k] = self._val_totals.get(k, 0.0) + float(v) * n
        self._val_count += n

    # ------------------------------------------------------------------
    # Epoch boundary  (Pattern B)
    # ------------------------------------------------------------------

    def epoch_end(
        self,
        epoch: Optional[int] = None,
        elapsed: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Close the current epoch: aggregate, log, step scheduler,
        checkpoint (if strategy matches), and check early stopping.

        Call once at the end of every epoch in your custom loop.

        Args:
            epoch: Current epoch number. Auto-increments when omitted.
            elapsed: Wall-clock seconds for this epoch (printed if provided).

        Returns:
            ``(train_metrics, val_metrics)`` — epoch-level averages.
        """
        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1

        # Aggregate
        train_m: Dict[str, float] = (
            {k: v / self._train_count for k, v in self._train_totals.items()}
            if self._train_count > 0 else {}
        )
        val_m: Dict[str, float] = (
            {k: v / self._val_count for k, v in self._val_totals.items()}
            if self._val_count > 0 else {}
        )

        # Reset accumulators
        self._train_totals.clear(); self._train_count = 0
        self._val_totals.clear();   self._val_count = 0

        # Append to history
        for k, v in train_m.items():
            self._history.setdefault(f"train_{k}", []).append(v)
        for k, v in val_m.items():
            self._history.setdefault(f"val_{k}", []).append(v)

        # The metric set used for checkpoint decisions and early stopping.
        # Fall back to training metrics when there is no validation loop.
        monitor = val_m if val_m else train_m

        self._last_train_metrics = train_m
        self._last_val_metrics   = val_m

        self._print_epoch(self._epoch, train_m, val_m, elapsed)
        self._step_scheduler_epoch(monitor)
        self._maybe_checkpoint(self._epoch, monitor)
        self.on_epoch_end(self._epoch, train_m, val_m)

        if self._should_early_stop(monitor):
            print(
                f"  ⏹  Early stop at epoch {self._epoch} "
                f"(no improvement on {self.config.early_stop_metric} "
                f"for {self.config.early_stop_patience} epochs)"
            )
            self._stopped = True

        return train_m, val_m

    # ------------------------------------------------------------------
    # Epoch iterator  (Pattern B)
    # ------------------------------------------------------------------

    def epochs(self, n: Optional[int] = None) -> Generator[int, None, None]:
        """Iterate over epoch numbers with early-stopping support.

        Starts from ``self._epoch + 1`` so that calling :meth:`resume` before
        the loop automatically skips already-completed epochs::

            trainer.resume("\\latest")          # sets _epoch = 12
            for epoch in trainer.epochs(50):    # yields 13 … 50
                ...
                trainer.epoch_end(epoch)
        """
        n     = n if n is not None else self.config.epochs
        start = self._epoch           # 0 normally; saved epoch after resume()
        last_epoch = start
        for epoch in range(start + 1, n + 1):
            last_epoch = epoch
            yield epoch
            if self._stopped:
                break

        # "last" strategy: single commit at very end.
        # Guard: only if at least one epoch ran this session.
        if (
            self.config.checkpoint_strategy == "last"
            and self._session
            and last_epoch > start
        ):
            final = self._last_val_metrics or self._last_train_metrics
            self._do_checkpoint(last_epoch, final)

    def resume(self, snapshot_tag: str = "\\latest") -> int:
        """Load weights and optimizer states from a snapshot into registered models.

        Sets the internal epoch counter so :meth:`epochs` continues from where
        training left off. Returns the saved epoch number.

        **Pattern A** — pass ``resume_from`` to :meth:`run` instead::

            trainer.run(train_loader, val_loader, resume_from="\\latest")

        **Pattern B** — call before your loop::

            trainer = Trainer(config)
            trainer.register("G", G, optimizer=opt_G)
            trainer.register("D", D, optimizer=opt_D)
            trainer.resume("\\latest")            # loads weights + opt states

            for epoch in trainer.epochs(EPOCHS):  # starts from saved epoch + 1
                ...

        Args:
            snapshot_tag: Snapshot to resume from. Defaults to ``"\\latest"``.

        Returns:
            Epoch number stored in the snapshot (i.e. where training stopped).

        Raises:
            NoFocusError: If no project is focused.
            SnapshotNotFoundError: If the snapshot tag cannot be resolved.
        """
        if not self._session:
            return 0

        import torch as _torch
        from ..vault.snapshot import Snapshot as _Snapshot

        snap_path = self._session._resolve_snapshot_path(snapshot_tag)

        for name, model in self._models.items():
            weights_path = snap_path / "weights" / f"{name}.pth"
            if weights_path.exists():
                state = _torch.load(
                    weights_path,
                    map_location=self.config.device,
                    weights_only=True,
                )
                model.load_state_dict(state)

            opt      = self._optimizers.get(name)
            opt_path = snap_path / "optimizers" / f"{name}.pth"
            if opt is not None and opt_path.exists():
                opt_state = _torch.load(opt_path, map_location="cpu", weights_only=True)
                opt.load_state_dict(opt_state)

        # Restore checkpoint-tracking state from saved metadata
        snap          = _Snapshot(snap_path)
        meta          = snap.metadata
        saved_metrics = meta.get("metrics", {})
        # "epoch" is stored inside metrics by _do_checkpoint
        saved_epoch   = saved_metrics.get("epoch", 0)

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

    @property
    def should_stop(self) -> bool:
        """``True`` once early stopping has triggered."""
        return self._stopped

    @property
    def history(self) -> Dict[str, list]:
        """Accumulated per-epoch metric history (``train_*`` / ``val_*`` keys)."""
        return dict(self._history)

    # ------------------------------------------------------------------
    # Override points  (Pattern A)
    # ------------------------------------------------------------------

    def step(self, batch: Any) -> torch.Tensor:
        """Compute the loss for one batch. Override for custom forward logic.

        Only called by :meth:`run`. In Pattern B the user controls forward
        passes entirely — this method is never invoked.
        """
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)
        return self.criterion(self.model(x), y)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """No-op hook — override in a subclass for arbitrary per-epoch logic.

        Called at the end of every epoch in both Pattern A and Pattern B,
        after checkpointing and early-stop evaluation.
        """
        pass

    # ------------------------------------------------------------------
    # Convenience: standard supervised loop  (Pattern A)
    # ------------------------------------------------------------------

    def run(self, train_loader, val_loader, resume_from: Optional[str] = None) -> dict:
        """Standard supervised training loop (Pattern A).

        Internally uses :meth:`epochs`, :meth:`update`, :meth:`update_val`,
        and :meth:`epoch_end`, so all checkpoint strategies, early stopping,
        schedulers, and custom metrics work automatically.

        For non-standard training (GAN, diffusion, multi-optimizer …) use
        :meth:`epochs` / :meth:`update` / :meth:`epoch_end` directly.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            resume_from: Snapshot tag to resume from (e.g. ``"\\latest"``).
                Restores weights and optimizer states before training starts.

        Returns:
            History dict with ``train_<metric>`` / ``val_<metric>`` lists.
        """
        if resume_from is not None:
            self.resume(resume_from)
        print(f"\n{'=' * 55}")
        print(
            f"  useml training — {self.config.epochs} epochs "
            f"on {self.config.device}  |  loss: {self.config.loss_name()}"
        )
        print(f"{'=' * 55}")

        primary = self.model
        opt     = self.optimizer
        self._was_interrupted = False

        try:
            for epoch in self.epochs(self.config.epochs):
                t0 = time.time()

                # ── train phase ────────────────────────────────────────────
                primary.train()
                for batch in train_loader:
                    opt.zero_grad()
                    loss = self.step(batch)
                    loss.backward()
                    opt.step()
                    if self.scheduler is not None and self.scheduler_per_batch:
                        self.scheduler.step()
                    bs = _batch_size(batch)
                    batch_m: Dict[str, float] = {"loss": loss.item()}
                    if self.metrics:
                        with torch.no_grad():
                            for name, fn in self.metrics.items():
                                batch_m[name] = float(fn(primary, batch, self.config.device))
                    self.update(n=bs, **batch_m)

                # ── val phase ──────────────────────────────────────────────
                primary.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        loss = self.step(batch)
                        bs = _batch_size(batch)
                        batch_v: Dict[str, float] = {"loss": loss.item()}
                        if self.metrics:
                            for name, fn in self.metrics.items():
                                batch_v[name] = float(fn(primary, batch, self.config.device))
                        self.update_val(n=bs, **batch_v)

                self.epoch_end(epoch, elapsed=time.time() - t0)

        except KeyboardInterrupt:
            self._was_interrupted = True
            print(f"\n  ⏸  Training interrupted at epoch {self._epoch}.")

        best_val = min(self._history.get("val_loss", [float("inf")]))
        if not self._was_interrupted:
            print(f"{'=' * 55}")
            print(f"  Best val_loss: {best_val:.4f}")
            print(f"{'=' * 55}\n")

        return self.history

    # ------------------------------------------------------------------
    # Internal — scheduler
    # ------------------------------------------------------------------

    def _step_scheduler_epoch(self, monitor: Dict[str, float]) -> None:
        if self.scheduler is None or self.scheduler_per_batch:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            key   = self.config.checkpoint_metric.replace("val_", "")
            value = monitor.get(key, monitor.get("loss", 0.0))
            self.scheduler.step(value)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # Internal — printing
    # ------------------------------------------------------------------

    def _print_epoch(
        self,
        epoch: int,
        train: Dict[str, float],
        val: Dict[str, float],
        elapsed: Optional[float],
    ) -> None:
        w = len(str(self.config.epochs))
        parts = [f"  epoch {epoch:{w}d}/{self.config.epochs}"]
        for k, v in train.items():
            parts.append(f"train_{k} {v:.4f}")
        for k, v in val.items():
            parts.append(f"val_{k} {v:.4f}")
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s")
        print(" │ ".join(parts))

    # ------------------------------------------------------------------
    # Internal — checkpointing
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self, epoch: int, monitor: Dict[str, float]) -> None:
        if not self._session:
            return
        strategy = self.config.checkpoint_strategy

        if strategy == "every_n":
            if epoch % self.config.checkpoint_every == 0:
                self._do_checkpoint(epoch, monitor)

        elif strategy == "best":
            current = self._lookup_metric(monitor, self.config.checkpoint_metric)
            if current is None:
                return
            if self._best_ckpt_value is None or _improved(
                current, self._best_ckpt_value, self.config.checkpoint_mode
            ):
                self._best_ckpt_value = current
                self._do_checkpoint(epoch, monitor)

        # "last" is handled at the end of epochs()

    def _do_checkpoint(self, epoch: int, monitor: Dict[str, float]) -> None:
        """Commit **all registered models** (with optimizer states) atomically."""
        inline = (
            {self._bundle_transform_key: self._bundle_transform_source}
            if self._bundle_transform_source and self._bundle_transform_key
            else None
        )
        for name, model in self._models.items():
            self._session.track(
                name, model,
                config=self.config,
                optimizer=self._optimizers.get(name),   # ← persisted for resume
            )

        # Normalise metric keys: ensure val_ prefix for snapshot metadata
        metric_kwargs = {
            (k if k.startswith("val_") else f"val_{k}"): round(v, 6)
            for k, v in monitor.items()
        }
        self._session.commit(
            message=f"epoch {epoch}",
            bundle_meta=self.bundle_meta,
            bundle_inline_source=inline,
            epoch=epoch,
            **metric_kwargs,
        )

    # ------------------------------------------------------------------
    # Internal — early stopping
    # ------------------------------------------------------------------

    def _should_early_stop(self, monitor: Dict[str, float]) -> bool:
        patience = self.config.early_stop_patience
        if patience is None:
            return False
        current = self._lookup_metric(monitor, self.config.early_stop_metric)
        if current is None:
            return False
        if self._es_best is None or _improved(current, self._es_best, self.config.early_stop_mode):
            self._es_best = current
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        return self._patience_counter >= patience

    @staticmethod
    def _lookup_metric(metrics: Dict[str, float], name: str) -> Optional[float]:
        """Resolve a metric name against a dict, stripping optional ``val_`` prefix."""
        if name in metrics:
            return metrics[name]
        if name.startswith("val_") and name[4:] in metrics:
            return metrics[name[4:]]
        return None

    # ------------------------------------------------------------------
    # Internal — session
    # ------------------------------------------------------------------

    def _get_active_session(self):
        try:
            import useml as _useml
            _ = _useml._session.project
            return _useml._session
        except Exception:
            return None


# ---------------------------------------------------------------------------
# run_training — entry point for useml.train()
# ---------------------------------------------------------------------------

def run_training(
    model_cls: Type[nn.Module],
    dataset: Any,
    config: Optional[Config] = None,
    step_fn: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    scheduler: Optional[Any] = None,
    resume_from: Optional[str] = None,
) -> dict:
    """Level-0 training entry point used by :func:`useml.train`.

    Args:
        model_cls: Model class (subclass of ``nn.Module``).
        dataset: Built-in name, ``"hf:<name>"``, ``Dataset``, or ``DataBundle``.
        config: Training configuration. Defaults to :class:`Config`.
        step_fn: Optional ``(model, batch, device) -> loss`` replacing the
            default ``criterion(model(x), y)`` step.
        metrics: Optional ``{name: fn}`` where ``fn(model, batch, device) -> scalar``.
        scheduler: Optional scheduler instance or ``(optimizer) -> scheduler`` factory.
        resume_from: Snapshot tag to resume from (e.g. ``"\\latest"``).
            Restores weights and optimizer states before training.

    Returns:
        History dict with ``train_<metric>`` / ``val_<metric>`` lists.
    """
    if config is None:
        config = Config()

    from ..dataset import DataBundle
    if isinstance(dataset, DataBundle):
        bundle_meta             = dataset.to_meta_dict()
        bundle_transform_source = dataset.transform_source()
        bundle_transform_key    = dataset.inline_source_key() if bundle_transform_source else None
    else:
        bundle_meta             = None
        bundle_transform_source = None
        bundle_transform_key    = None

    train_loader, val_loader = load_dataset(dataset, config)
    model   = _build_model(model_cls, config)
    trainer = Trainer(config)
    trainer.register("model", model)

    trainer.bundle_meta              = bundle_meta
    trainer._bundle_transform_source = bundle_transform_source
    trainer._bundle_transform_key    = bundle_transform_key

    if step_fn is not None:
        _device = config.device
        _fn     = step_fn
        trainer.step = lambda batch: _fn(trainer.model, batch, _device)

    if metrics:
        trainer.metrics = metrics

    if scheduler is not None:
        trainer.scheduler = (
            scheduler(trainer.optimizer)
            if callable(scheduler) and not hasattr(scheduler, "step")
            else scheduler
        )

    return trainer.run(train_loader, val_loader, resume_from=resume_from)
