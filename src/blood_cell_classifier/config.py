from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path = Path("outputs")
    image_size: int = 64
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.4
    train_fraction: float = 0.8
    seed: int = 42
    workers: int = 0
    interpretability: bool = True
