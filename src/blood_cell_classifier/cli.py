import argparse
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="blood-cell-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run the complete MLP/CNN experiment")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=Path("outputs"))
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--skip-interpretability", action="store_true")
    args = parser.parse_args()
    run_pipeline(ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, interpretability=not args.skip_interpretability))


if __name__ == "__main__":
    main()
