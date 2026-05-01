import torch.nn as nn


class Loss(nn.Module):
    """Base class for useml-managed custom loss functions.

    Subclass this and implement :meth:`forward`. The vault automatically
    archives the source code of any ``Loss`` subclass alongside model weights,
    mirroring the behaviour of :class:`useml.Model`.

    Example:
        >>> class FocalLoss(useml.Loss):
        ...     def __init__(self, gamma: float = 2.0):
        ...         super().__init__()
        ...         self.gamma = gamma
        ...
        ...     def forward(self, logits, targets):
        ...         import torch.nn.functional as F
        ...         import torch
        ...         ce = F.cross_entropy(logits, targets, reduction="none")
        ...         pt = torch.exp(-ce)
        ...         return ((1 - pt) ** self.gamma * ce).mean()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        raise NotImplementedError(
            "Subclasses must implement forward(self, predictions, targets)."
        )
