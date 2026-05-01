import torch.nn as nn


class Model(nn.Module):
    """Base class for useml Level-0 models.

    Subclass this and implement :meth:`forward`. If ``__init__`` accepts a
    ``config`` or ``cfg`` parameter, :func:`useml.train` will pass the active
    :class:`~useml.Config` automatically.

    Example:
        >>> class MyModel(useml.Model):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.Linear(784, 10)
        ...
        ...     def forward(self, x):
        ...         return self.fc(x.view(x.size(0), -1))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward(self, x).")
