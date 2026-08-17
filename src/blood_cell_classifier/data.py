from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .config import ExperimentConfig

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class DataBundle:
    train: Dataset
    test: Dataset
    train_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]


def build_data(config: ExperimentConfig) -> DataBundle:
    transform = Compose([Resize((config.image_size, config.image_size)), ToTensor(), Normalize(MEAN, STD)])
    dataset = ImageFolder(config.data_dir, transform=transform)
    train_size = int(config.train_fraction * len(dataset))
    if train_size == 0 or train_size == len(dataset):
        raise ValueError("The dataset must contain enough samples for a train/test split.")
    train, test = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(config.seed))
    loader_args = {"batch_size": config.batch_size, "num_workers": config.workers, "pin_memory": torch.cuda.is_available()}
    return DataBundle(train, test, DataLoader(train, shuffle=True, **loader_args), DataLoader(test, shuffle=False, **loader_args), dataset.classes)
