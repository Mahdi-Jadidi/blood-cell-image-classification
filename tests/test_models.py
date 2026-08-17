import torch

from blood_cell_classifier.models import BloodCellCNN, BloodCellMLP


def test_models_produce_class_logits() -> None:
    inputs = torch.randn(2, 3, 64, 64)
    assert BloodCellMLP(5)(inputs).shape == (2, 5)
    assert BloodCellCNN(5)(inputs).shape == (2, 5)
