import inspect
from pathlib import Path
from typing import Any, Dict, Optional


class Component:
    """Represents a named element of the machine learning pipeline.

    A Component bundles a PyTorch model with its associated configuration
    and optimizer. It also automatically attempts to locate the source code
    defining the model's class for auditability.

    Attributes:
        name (str): Unique identifier for the component.
        model (Any): The PyTorch model instance.
        config (Optional[Dict]): Hyperparameters or architectural metadata.
        optimizer (Optional[Any]): The PyTorch optimizer instance.
        source_path (Optional[Path]): Path to the file defining the model class.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Initializes the Component and inspects the model source.

        Args:
            name (str): Unique name (e.g., 'encoder', 'classifier').
            model (Any): PyTorch model object.
            config (Optional[Dict]): Dictionary of parameters.
            optimizer (Optional[Any]): Optimizer object associated with the model.
        """
        self.name = name
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.source_path: Optional[Path] = self._inspect_source(model)

    def _inspect_source(self, model: Any) -> Optional[Path]:
        """Identifies the file path where the model's class is defined.

        Args:
            model (Any): The model instance to inspect.

        Returns:
            Optional[Path]: The absolute path to the source file, or None if
                the inspection fails or points to a built-in/compiled module.
        """
        try:
            source_file = inspect.getfile(model.__class__)
            path = Path(source_file).resolve()

            # Optional: Ignore source code if it's from site-packages (library code)
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
