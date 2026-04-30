from dataclasses import dataclass, asdict
import torch


@dataclass
class Config:
    # --- Training ---
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"       # adam | adamw | sgd
    loss: str = "cross_entropy"   # cross_entropy | mse | bce | l1

    # --- Hardware ---
    device: str = "auto"          # auto | cpu | cuda | mps

    # --- Data ---
    val_split: float = 0.1
    num_workers: int = 0
    data_dir: str = ".useml_data"
    seed: int = 42

    # --- Vault checkpointing ---
    checkpoint_every: int = 5     # commit to vault every N epochs
    checkpoint_metric: str = "val_loss"

    def __post_init__(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)
