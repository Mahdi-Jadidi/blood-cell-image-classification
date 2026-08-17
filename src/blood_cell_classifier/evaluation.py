import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions, probabilities = [], [], []
    for images, batch_labels in loader:
        output = model(images.to(device))
        probs = output.softmax(1).cpu().numpy()
        labels.extend(batch_labels.numpy())
        predictions.extend(probs.argmax(1))
        probabilities.extend(probs)
    return np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def evaluate_model(model: nn.Module, loader, classes: list[str], device: torch.device) -> dict:
    labels, predictions, probabilities = predict(model, loader, device)
    return {"accuracy": float((labels == predictions).mean()), "classification_report": classification_report(labels, predictions, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(labels, predictions, labels=range(len(classes))).tolist(), "labels": labels, "predictions": predictions, "probabilities": probabilities}


def misclassified_samples(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, limit_per_class: int = 5) -> list[dict]:
    records, counts = [], {}
    for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
        label = int(label)
        if label != prediction and counts.get(label, 0) < limit_per_class:
            records.append({"index": index, "true": label, "predicted": int(prediction), "confidence": float(probabilities[index, prediction])})
            counts[label] = counts.get(label, 0) + 1
    return records
