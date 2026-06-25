"""Train and tune a CatBoost classifier on SAVEE audio features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost on SAVEE audio features.")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("artifacts/savee_binary_features.joblib"),
        help="Feature cache path.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/savee_binary_catboost.joblib"),
        help="Where to save the trained model.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")

    features = joblib.load(args.feature_cache)
    print("Loaded SAVEE features:")
    print(f"  Train: {features['X_train'].shape}")
    print(f"  Val: {features['X_val'].shape}")
    print(f"  Test: {features['X_test'].shape}")

    from catboost import CatBoostClassifier

    # We will search over a grid of hyperparameters for CatBoost
    depths = [4, 6, 8]
    learning_rates = [0.01, 0.03, 0.05, 0.1]
    l2_leaf_regs = [1.0, 3.0, 5.0]

    best_model = None
    best_val_acc = -1.0
    best_results = None

    for depth in depths:
        for lr in learning_rates:
            for l2 in l2_leaf_regs:
                model = CatBoostClassifier(
                    iterations=800,
                    depth=depth,
                    learning_rate=lr,
                    l2_leaf_reg=l2,
                    random_seed=args.random_state,
                    verbose=False,
                    early_stopping_rounds=80,
                )
                model.fit(
                    features["X_train"],
                    features["y_train"],
                    eval_set=(features["X_val"], features["y_val"]),
                    use_best_model=True,
                )

                train_metrics = evaluate(model, features["X_train"], features["y_train"])
                val_metrics = evaluate(model, features["X_val"], features["y_val"])
                test_metrics = evaluate(model, features["X_test"], features["y_test"])

                val_acc = val_metrics["accuracy"]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_results = {
                        "depth": depth,
                        "learning_rate": lr,
                        "l2_leaf_reg": l2,
                        "train": train_metrics,
                        "validation": val_metrics,
                        "test": test_metrics,
                    }

    print("\nBest SAVEE CatBoost Model:")
    print(json.dumps(best_results, indent=2))

    # Save payload
    payload = {
        "model": best_model,
        "metrics": {
            **best_results,
            "task": "binary",
            "classifier": "catboost",
            "class_names": ["negative", "positive"],
            "sample_rate": 16000,
        },
        "class_names": ["negative", "positive"],
        "task": "binary",
        "sample_rate": 16000,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print(f"Saved best model to {args.model_out}")


def evaluate(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


if __name__ == "__main__":
    main()
