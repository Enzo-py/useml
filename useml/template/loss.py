import torch.nn as nn


class Loss(nn.Module):
    """Base class for useml-managed custom loss functions.

    Subclass this and implement forward(self, predictions, targets).
    Source code is automatically archived with each vault snapshot,
    just like useml.Model.

    Example
    -------
    class FocalLoss(useml.Loss):
        def __init__(self, gamma: float = 2.0):
            super().__init__()
            self.gamma = gamma

        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        raise NotImplementedError("Implement forward(self, predictions, targets)")
