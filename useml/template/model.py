import torch.nn as nn


class Model(nn.Module):
    """Base class for useml Level-0 models.

    Subclass this and implement forward(self, x).
    The Trainer handles everything else (loop, optimizer, checkpointing).

    If your __init__ accepts a 'config' or 'cfg' parameter, useml.train()
    will pass the active Config automatically.

    Example
    -------
    class MyModel(useml.Model):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Implement forward(self, x)")
