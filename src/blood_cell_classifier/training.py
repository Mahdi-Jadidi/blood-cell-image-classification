from collections.abc import Iterable

import torch
from torch import nn


def run_epoch(model: nn.Module, batches: Iterable, criterion: nn.Module, device: torch.device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = criterion(output, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def fit(model: nn.Module, train_loader, test_loader, device: torch.device, epochs: int, learning_rate: float, weight_decay: float) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {key: [] for key in ("train_loss", "test_loss", "train_accuracy", "test_accuracy")}
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        scheduler.step()
        for key, value in (("train_loss", train_loss), ("test_loss", test_loss), ("train_accuracy", train_accuracy), ("test_accuracy", test_accuracy)):
            history[key].append(value)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    return history
