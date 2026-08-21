# Blood Cell Image Classification

An applied computer-vision study for recognizing blood-cell classes from microscopy imagery. The project compares a dense neural baseline against a convolutional network, then asks the useful follow-up question: what image evidence is each model relying on?

## Why it matters

Blood-cell morphology is defined by local structure: nucleus shape, membrane boundaries, and granular texture. A classifier that only predicts a label is not enough for this domain, so this repository pairs model evaluation with error analysis and image-level interpretability.

## What was built

- Deterministic `ImageFolder` data split and training pipeline for repeatable experiments.
- Matched MLP and CNN baselines for a fair inductive-bias comparison.
- Per-class metrics, confusion matrices, misclassification review, and training histories.
- First-layer feature maps, geometric-robustness checks, and occlusion-sensitivity heatmaps.

## Main takeaways

The CNN provides the stronger visual baseline because it can learn local morphological patterns while the MLP must infer them from flattened pixels. The interpretability stage makes this claim inspectable: it shows whether model confidence changes around diagnostically meaningful cell regions rather than only reporting an aggregate score.

## Architecture

```text
data -> deterministic split -> MLP/CNN training -> evaluation
                                      -> error analysis -> feature maps/occlusion maps
```

The implementation lives in `src/blood_cell_classifier`, with dedicated data, training, evaluation, and interpretability modules.

## Reproduce

```bash
pip install -e .
blood-cell-classifier train --data-dir /path/to/dataset --output-dir outputs
```

The dataset uses the standard ImageFolder layout: one directory per cell class. A run writes checkpoints, classification reports, confusion matrices, and compressed interpretability artifacts.

## Stack

PyTorch, torchvision, scikit-learn, NumPy, and Pillow. GitHub Actions validates the package and model contract on every change.
