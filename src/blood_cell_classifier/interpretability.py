import numpy as np
import torch
from torch import nn
from torchvision.transforms import Normalize, RandomAffine, RandomRotation, ToPILImage, ToTensor

from .data import MEAN, STD


@torch.no_grad()
def probabilities(model: nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    return model(image.unsqueeze(0).to(device)).softmax(1)[0].cpu().numpy()


def first_layer_feature_maps(model: nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
    captured = {}
    layer = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    handle = layer.register_forward_hook(lambda _module, _inputs, output: captured.update(output=output.detach().cpu()))
    probabilities(model, image, device)
    handle.remove()
    return captured["output"][0].numpy()


def geometric_robustness(models: dict[str, nn.Module], image: torch.Tensor, true_class: int, device: torch.device, angles=(15, 30, 45, 90), shifts=(5, 10, 20)) -> list[dict]:
    pil = ToPILImage()(image * 0.5 + 0.5)
    variants = [("original", image)]
    variants += [(f"rotation_{angle}", Normalize(MEAN, STD)(ToTensor()(RandomRotation((angle, angle))(pil)))) for angle in angles]
    size = image.shape[-1]
    variants += [(f"translation_{shift}", Normalize(MEAN, STD)(ToTensor()(RandomAffine(0, translate=(shift / size, shift / size))(pil)))) for shift in shifts]
    records = []
    for transform_name, transformed in variants:
        for model_name, model in models.items():
            probs = probabilities(model, transformed, device)
            prediction = int(probs.argmax())
            records.append({"transform": transform_name, "model": model_name, "true": true_class, "predicted": prediction, "confidence": float(probs[prediction]), "correct": prediction == true_class})
    return records


def occlusion_heatmap(model: nn.Module, image: torch.Tensor, true_class: int, device: torch.device, patch_size: int = 8, stride: int = 4) -> np.ndarray:
    _, height, width = image.shape
    heatmap = np.zeros((height, width))
    counts = np.zeros((height, width))
    baseline = probabilities(model, image, device)[true_class]
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, y:y + patch_size, x:x + patch_size] = image.mean()
            drop = baseline - probabilities(model, occluded, device)[true_class]
            heatmap[y:y + patch_size, x:x + patch_size] += drop
            counts[y:y + patch_size, x:x + patch_size] += 1
    return heatmap / np.maximum(counts, 1)
