import torch
from torch import nn


class BloodCellMLP(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * image_size * image_size, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class BloodCellCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for input_channels, output_channels in ((3, 32), (32, 64), (64, 128)):
            blocks.extend([nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.25)])
        self.features = nn.Sequential(*blocks)
        spatial = image_size // 8
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * spatial * spatial, 384), nn.BatchNorm1d(384), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
