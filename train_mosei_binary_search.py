"""Search for the best-performing classifiers on CMU-MOSEI binary sentiment tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for best CMU-MOSEI binary classifiers.")
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
        help="Keep samples with abs(sentiment_score) > threshold. Use 0.3, 1.0, 2.0, or 2.5.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Where to save the best model. Defaults to artifacts/mosei_binary_<threshold>_best.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.cache_path}")

    print(f"Loading feature cache from {args.cache_path}...")
    cache = joblib.load(args.cache_path)
    features = build_binary_splits(cache, args.threshold)

    print(f"Threshold: {args.threshold}")
    print(f"Split sizes - Train: {features['X_train'].shape[0]}, Val: {features['X_val'].shape[0]}, Test: {features['X_test'].shape[0]}")

    candidates = build_candidates(args.random_state, features["X_train"].shape[0])
    results = []
    best_model = None
    best_score = -1.0
    best_name = ""

    for name, model in candidates:
        print(f"\nTraining {name}...")
        try:
            model.fit(features["X_train"], features["y_train"])
            train_metrics = evaluate(model, features["X_train"], features["y_train"])
            val_metrics = evaluate(model, features["X_val"], features["y_val"])
            test_metrics = evaluate(model, features["X_test"], features["y_test"])

            result = {
                "model": name,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
            }
            results.append(result)
            print(f"  Train Acc: {train_metrics['accuracy'] * 100:.2f}% | Val Acc: {val_metrics['accuracy'] * 100:.2f}% | Test Acc: {test_metrics['accuracy'] * 100:.2f}%")

            # Prioritize validation accuracy, then test accuracy
            score_key = val_metrics["accuracy"]
            if score_key > best_score:
                best_score = score_key
                best_model = model
                best_name = name
        except Exception as e:
            print(f"  Error training {name}: {e}")

    # Train CatBoostClassifier if available
    try:
        from catboost import CatBoostClassifier
        print("\nTraining CatBoostClassifier...")
        cat_model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            random_seed=args.random_state,
            verbose=False,
            early_stopping_rounds=80,
        )
        cat_model.fit(
            features["X_train"],
            features["y_train"],
            eval_set=(features["X_val"], features["y_val"]),
            use_best_model=True,
        )
        train_metrics = evaluate(cat_model, features["X_train"], features["y_train"])
        val_metrics = evaluate(cat_model, features["X_val"], features["y_val"])
        test_metrics = evaluate(cat_model, features["X_test"], features["y_test"])

        result = {
            "model": "CatBoost",
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }
        results.append(result)
        print(f"  Train Acc: {train_metrics['accuracy'] * 100:.2f}% | Val Acc: {val_metrics['accuracy'] * 100:.2f}% | Test Acc: {test_metrics['accuracy'] * 100:.2f}%")

        if val_metrics["accuracy"] > best_score:
            best_score = val_metrics["accuracy"]
            best_model = cat_model
            best_name = "CatBoost"
    except Exception as e:
        print(f"  Error training CatBoost: {e}")

    if best_model is None:
        raise RuntimeError("No candidates were successfully trained.")

    best_result = [r for r in results if r["model"] == best_name][0]
    print(f"\nBest Model: {best_name}")
    print(json.dumps(best_result, indent=2))

    model_out = args.model_out
    if model_out is None:
        model_out = Path("artifacts") / f"mosei_binary_{args.threshold}_best.joblib"

    payload = {
        "model": best_model,
        "metrics": {
            **best_result,
            "task": "mosei_binary_sentiment",
            "threshold": args.threshold,
            "class_names": ["negative", "positive"],
            "split_sizes": {
                "train": int(features["y_train"].shape[0]),
                "validation": int(features["y_val"].shape[0]),
                "test": int(features["y_test"].shape[0]),
            },
            "source_cache": str(args.cache_path),
        },
        "class_names": ["negative", "positive"],
        "task": "mosei_binary_sentiment",
        "threshold": args.threshold,
        "source_cache": str(args.cache_path),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_out, compress=3)
    print(f"Saved best model to {model_out}")


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


def build_candidates(random_state: int, train_size: int = 0) -> list[tuple[str, object]]:
    candidates = []
    
    # Logistic Regression
    for c in [0.01, 0.1, 1.0, 5.0, 10.0]:
        candidates.append((
            f"LogisticRegression_C{c}",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=random_state))
            ])
        ))

    # Support Vector Machines - Only run for small datasets to prevent slow training on CPU
    if train_size <= 1000:
        for c in [0.1, 1.0, 10.0]:
            candidates.append((
                f"SVM_RBF_C{c}",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(C=c, kernel="rbf", class_weight="balanced", random_state=random_state, probability=True))
                ])
            ))
            candidates.append((
                f"SVM_Linear_C{c}",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(C=c, kernel="linear", class_weight="balanced", random_state=random_state, probability=True))
                ])
            ))

    # Random Forest
    for estimators in [200, 500]:
        for max_depth in [10, None]:
            candidates.append((
                f"RandomForest_N{estimators}_D{max_depth}",
                RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, class_weight="balanced", random_state=random_state, n_jobs=-1)
            ))

    # Multi-Layer Perceptron (MLP)
    candidates.append((
        "MLP_512_256",
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(512, 256),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state
            ))
        ])
    ))

    return candidates


def evaluate(model: object, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


if __name__ == "__main__":
    main()
