import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..template.config import Config


class Component:
    """Named element of the machine learning pipeline tracked by a session.

    Attributes:
        name: Unique identifier for this component.
        model: PyTorch model instance.
        config: YAML-safe hyperparameter dict, derived from the Config object
            when one is supplied.
        useml_config: Original Config instance, preserved so the vault layer
            can extract loss metadata and source code.
        optimizer: Optional associated optimiser.
        source_path: Resolved path to the file defining the model class,
            or None for site-packages / built-in classes.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        config: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Initialises a Component and normalises the config argument.

        Args:
            name: Unique identifier for this component.
            model: PyTorch model or compatible object.
            config: Hyperparameters as a plain dict, or a ``useml.Config``
                instance. When a Config is passed, a YAML-safe dict is derived
                automatically via ``Config.to_dict()``.
            optimizer: Optional associated optimiser.
        """
        from ..template.config import Config as _Config

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.source_path: Optional[Path] = self._resolve_source(model)

        if isinstance(config, _Config):
            self.useml_config: Optional[_Config] = config
            self.config: Optional[Dict] = config.to_dict()
        else:
            self.useml_config = None
            self.config = config

    def _resolve_source(self, model: Any) -> Optional[Path]:
        try:
            source_file = inspect.getfile(model.__class__)
            path = Path(source_file).resolve()
            if "site-packages" in str(path):
                return None
            return path
        except (TypeError, OSError):
            return None

    def __repr__(self) -> str:
        return (
            f"<useml.Component name='{self.name}' "
            f"has_config={self.config is not None} "
            f"has_optimizer={self.optimizer is not None}>"
        )
