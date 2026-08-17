import json
import random

import numpy as np
import torch

from .config import ExperimentConfig
from .data import build_data
from .evaluation import evaluate_model, misclassified_samples
from .interpretability import first_layer_feature_maps, geometric_robustness, occlusion_heatmap
from .models import BloodCellCNN, BloodCellMLP, parameter_count
from .training import fit


def run_pipeline(config: ExperimentConfig) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data(config)
    models = {"mlp": BloodCellMLP(len(data.classes), config.image_size, config.dropout).to(device), "cnn": BloodCellCNN(len(data.classes), config.image_size, config.dropout).to(device)}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(device), "classes": data.classes, "models": {}}
    for name, model in models.items():
        history = fit(model, data.train_loader, data.test_loader, device, config.epochs, config.learning_rate, config.weight_decay)
        evaluation = evaluate_model(model, data.test_loader, data.classes, device)
        serial_config = {key: str(value) if hasattr(value, "parts") else value for key, value in config.__dict__.items()}
        torch.save({"state_dict": model.state_dict(), "classes": data.classes, "config": serial_config}, config.output_dir / f"{name}.pt")
        (config.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        np.save(config.output_dir / f"{name}_confusion_matrix.npy", evaluation["confusion_matrix"])
        summary["models"][name] = {"parameters": parameter_count(model), "accuracy": evaluation["accuracy"], "classification_report": evaluation["classification_report"], "misclassified": misclassified_samples(evaluation["labels"], evaluation["predictions"], evaluation["probabilities"])}
    if config.interpretability and len(data.test):
        image, true_class = data.test[0]
        np.savez_compressed(config.output_dir / "interpretability.npz", feature_maps=first_layer_feature_maps(models["cnn"], image, device), mlp_occlusion=occlusion_heatmap(models["mlp"], image, true_class, device), cnn_occlusion=occlusion_heatmap(models["cnn"], image, true_class, device))
        summary["geometric_robustness"] = geometric_robustness(models, image, true_class, device)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
