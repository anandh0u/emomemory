"""Train binary CMU-MOSEI sentiment models from the saved feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary CMU-MOSEI sentiment from cached features.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/mosei_aligned_native_concat_features.joblib"),
        help="Feature cache produced by train.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Keep samples with abs(sentiment_score) > threshold. Use 0.3 for non-neutral, 1.0 for strong sentiment.",
    )
    parser.add_argument("--c", type=float, default=1.0, help="Logistic regression C value.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/mosei_binary_nonneutral_logreg.joblib"),
        help="Output artifact path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.cache_path}")

    cache = joblib.load(args.cache_path)
    features = build_binary_splits(cache, args.threshold)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(features["X_train"], features["y_train"])

    metrics = {
        "train": evaluate(model, features["X_train"], features["y_train"]),
        "validation": evaluate(model, features["X_val"], features["y_val"]),
        "test": evaluate(model, features["X_test"], features["y_test"]),
        "task": "mosei_binary_sentiment",
        "threshold": args.threshold,
        "classifier": "logreg",
        "class_names": ["negative", "positive"],
        "split_sizes": {
            "train": int(features["y_train"].shape[0]),
            "validation": int(features["y_val"].shape[0]),
            "test": int(features["y_test"].shape[0]),
        },
        "source_cache": str(args.cache_path),
    }
    payload: dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "class_names": ["negative", "positive"],
        "task": "mosei_binary_sentiment",
        "threshold": args.threshold,
        "source_cache": str(args.cache_path),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print(json.dumps(metrics, indent=2))
    print(f"Saved binary MOSEI model to {args.model_out}")


def build_binary_splits(cache: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        scores = np.asarray(cache[f"s_{split}"], dtype=np.float32)
        mask = np.abs(scores) > threshold
        labels = (scores[mask] > 0).astype(np.int64)
        if labels.size == 0 or np.unique(labels).size < 2:
            raise ValueError(f"Threshold {threshold} leaves split '{split}' without both classes.")
        output[f"X_{split}"] = np.asarray(cache[f"X_{split}"], dtype=np.float32)[mask]
        output[f"y_{split}"] = labels
    return output


def evaluate(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


if __name__ == "__main__":
    main()
