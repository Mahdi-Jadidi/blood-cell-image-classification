# Blood Cell Image Classification

Production PyTorch package for comparing a fully connected network and a convolutional network on multi-class blood-cell microscopy classification. The pipeline covers deterministic data splitting, both model families, metric comparison, error analysis, first-layer feature maps, geometric robustness, and occlusion sensitivity.

## Architecture

```text
src/blood_cell_classifier/
├── config.py
├── data.py
├── models.py
├── training.py
├── evaluation.py
├── interpretability.py
├── pipeline.py
└── cli.py
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
blood-cell-classifier train --data-dir /path/to/dataset --output-dir outputs
```

The dataset follows `torchvision.datasets.ImageFolder`, with one directory per class. Use `--epochs 80` for the default full training run and `--skip-interpretability` for a training-only run. Outputs include model checkpoints, histories, classification reports, confusion matrices, and compressed interpretability artifacts.

## Topics

`computer-vision` `pytorch` `medical-imaging` `classification` `explainable-ai`
