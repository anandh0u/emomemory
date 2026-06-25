"""Train a lightweight FER2013 face-emotion classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.fer_loader import BinaryMode, extract_fer_features, find_default_fer_csv, load_fer2013_csv


def parse_args() -> argparse.Namespace:
    default_csv = find_default_fer_csv()
    parser = argparse.ArgumentParser(description="Train a FER2013 image emotion model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_csv,
        help="Path to fer2013.csv. Defaults to the saved local FER CSV if found.",
    )
    parser.add_argument("--task", choices=["fer7", "binary"], default="fer7", help="7-class FER or binary positive/negative.")
    parser.add_argument("--classifier", choices=["sgd", "mlp"], default="sgd", help="CPU-friendly classifier.")
    parser.add_argument("--feature-size", type=int, default=24, help="Downsampled feature image size.")
    parser.add_argument("--max-iter", type=int, default=80, help="Classifier iteration limit.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/fer_classifier.joblib"), help="Output artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_path is None:
        raise FileNotFoundError(
            "Could not find fer2013.csv automatically. Pass --data-path E:\\path\\to\\fer2013.csv"
        )

    dataset = load_fer2013_csv(args.data_path, task=args.task)
    print("Loaded FER2013:", json.dumps(split_sizes(dataset), indent=2))
    print("Building FER feature arrays...")
    features = {
        "X_train": extract_fer_features(dataset.train.images, args.feature_size),
        "y_train": dataset.train.labels,
        "X_val": extract_fer_features(dataset.val.images, args.feature_size),
        "y_val": dataset.val.labels,
        "X_test": extract_fer_features(dataset.test.images, args.feature_size),
        "y_test": dataset.test.labels,
    }

    model = build_model(args)
    model.fit(features["X_train"], features["y_train"])
    metrics = {
        "train": evaluate(model, features["X_train"], features["y_train"]),
        "validation": evaluate(model, features["X_val"], features["y_val"]),
        "test": evaluate(model, features["X_test"], features["y_test"]),
        "task": args.task,
        "classifier": args.classifier,
        "class_names": dataset.class_names,
        "feature_size": args.feature_size,
        "source": str(dataset.source),
    }

    payload: dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "class_names": dataset.class_names,
        "task": args.task,
        "feature_size": args.feature_size,
        "source": str(dataset.source),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print(json.dumps(metrics, indent=2))
    print(f"Saved FER model to {args.model_out}")


def split_sizes(dataset: object) -> dict[str, Any]:
    return {
        "source": str(dataset.source),
        "task": dataset.task,
        "class_names": dataset.class_names,
        "train": int(dataset.train.labels.shape[0]),
        "val": int(dataset.val.labels.shape[0]),
        "test": int(dataset.test.labels.shape[0]),
    }


def build_model(args: argparse.Namespace) -> object:
    if args.classifier == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate_init=1e-3,
                        max_iter=args.max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=8,
                        random_state=args.random_state,
                        verbose=True,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    penalty="elasticnet",
                    l1_ratio=0.15,
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=-1,
                    verbose=0,
                ),
            ),
        ]
    )


def evaluate(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


if __name__ == "__main__":
    main()
