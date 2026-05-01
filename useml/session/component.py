import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..template.config import Config


class Component:
    """Represents a named element of the machine learning pipeline.

    Attributes:
        name (str): Unique identifier for the component.
        model (Any): The PyTorch model instance.
        config (Optional[Dict]): Serializable hyperparameters (YAML-safe dict).
        useml_config (Optional[Config]): Original Config object, preserved so
            the vault can extract loss-source code and other metadata.
        optimizer (Optional[Any]): The PyTorch optimizer instance.
        source_path (Optional[Path]): Path to the file defining the model class.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        config: Optional[Any] = None,    # dict OR Config instance
        optimizer: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.source_path: Optional[Path] = self._inspect_source(model)

        # Accept either a raw dict or a Config instance.
        # Always expose a YAML-safe dict via self.config.
        from ..template.config import Config as _Config
        if isinstance(config, _Config):
            self.useml_config: Optional[_Config] = config
            self.config: Optional[Dict] = config.to_dict()
        else:
            self.useml_config = None
            self.config = config          # dict or None — stored as-is

    def _inspect_source(self, model: Any) -> Optional[Path]:
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
