"""Evaluation helpers for MMER sentiment classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions.")
    parser.add_argument("--labels", type=Path, required=True, help="Joblib/NumPy file containing integer labels.")
    parser.add_argument("--predictions", type=Path, required=True, help="Joblib/NumPy file containing integer predictions.")
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    return np.asarray(joblib.load(path))


def main() -> None:
    args = parse_args()
    labels = _load_array(args.labels)
    predictions = _load_array(args.predictions)
    metrics = compute_metrics(labels, predictions)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
