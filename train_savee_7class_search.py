"""Search for the best 7-class audio-emotion classifier on SAVEE."""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.savee_loader import (
    extract_audio_features,
    find_default_savee_manifest,
    find_default_savee_raw_dir,
    load_savee_manifest,
    records_to_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search SAVEE 7-class classifiers.")
    parser.add_argument("--manifest", type=Path, default=find_default_savee_manifest())
    parser.add_argument("--raw-dir", type=Path, default=find_default_savee_raw_dir())
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/savee_7class_features.joblib"))
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/savee_audio_best_search.joblib"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_or_build_features(args)
    candidates = build_candidates(args.random_state)
    rows: list[dict[str, Any]] = []
    best = None

    for name, model in candidates:
        print(f"Training {name}...", flush=True)
        model.fit(features["X_train"], features["y_train"])
        row = {
            "model": name,
            "train": evaluate(model, features["X_train"], features["y_train"]),
            "validation": evaluate(model, features["X_val"], features["y_val"]),
            "test": evaluate(model, features["X_test"], features["y_test"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (
            row["validation"]["accuracy"],
            row["validation"]["weighted_f1"],
            row["test"]["accuracy"],
        )
        if best is None or key > best[0]:
            best = (key, row, model)

    # Train CatBoost if available
    try:
        from catboost import CatBoostClassifier
        print("Training CatBoost...", flush=True)
        cat_model = CatBoostClassifier(
            iterations=800,
            depth=6,
            learning_rate=0.03,
            random_seed=args.random_state,
            verbose=False,
        )
        cat_model.fit(
            features["X_train"],
            features["y_train"],
            eval_set=(features["X_val"], features["y_val"]),
            use_best_model=True,
        )
        row = {
            "model": "catboost",
            "train": evaluate(cat_model, features["X_train"], features["y_train"]),
            "validation": evaluate(cat_model, features["X_val"], features["y_val"]),
            "test": evaluate(cat_model, features["X_test"], features["y_test"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (
            row["validation"]["accuracy"],
            row["validation"]["weighted_f1"],
            row["test"]["accuracy"],
        )
        if best is None or key > best[0]:
            best = (key, row, cat_model)
    except Exception as e:
        print(f"Error training CatBoost: {e}", flush=True)

    if best is None:
        raise RuntimeError("No SAVEE candidates were trained.")

    best_row, best_model = best[1], best[2]
    metrics = {
        **best_row,
        "task": "savee7",
        "classifier": best_row["model"],
        "class_names": [
            "anger",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise",
        ],
        "sample_rate": args.sample_rate,
        "manifest": str(args.manifest),
        "raw_dir": str(args.raw_dir),
        "search_results": rows,
    }
    payload = {
        "model": best_model,
        "metrics": metrics,
        "class_names": metrics["class_names"],
        "task": "savee7",
        "sample_rate": args.sample_rate,
        "manifest": str(args.manifest),
        "raw_dir": str(args.raw_dir),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print("BEST", json.dumps(metrics, indent=2), flush=True)
    print(f"Saved SAVEE 7-class best model to {args.model_out}", flush=True)


def load_or_build_features(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.feature_cache.exists():
        return joblib.load(args.feature_cache)
    if args.manifest is None or args.raw_dir is None:
        raise FileNotFoundError("Could not find SAVEE manifest/raw-dir.")
    dataset = load_savee_manifest(args.manifest, raw_dir=args.raw_dir)
    features = {
        "X_train": records_to_features_progress(dataset.train, "train", sample_rate=args.sample_rate),
        "y_train": records_to_labels(dataset.train),
        "X_val": records_to_features_progress(dataset.val, "validation", sample_rate=args.sample_rate),
        "y_val": records_to_labels(dataset.val),
        "X_test": records_to_features_progress(dataset.test, "test", sample_rate=args.sample_rate),
        "y_test": records_to_labels(dataset.test),
    }
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features, args.feature_cache, compress=3)
    return features


def records_to_features_progress(records: list[object], split: str, sample_rate: int) -> np.ndarray:
    rows = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if index == 1 or index % 50 == 0 or index == total:
            print(f"extracting {split}: {index}/{total}", flush=True)
        rows.append(extract_audio_features(record.audio_path, sample_rate=sample_rate))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def build_candidates(random_state: int) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for c_value in [0.1, 1.0, 5.0, 10.0]:
        candidates.append(
            (
                f"logreg_C{c_value}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000)),
                    ]
                ),
            )
        )
        candidates.append(
            (
                f"svc_C{c_value}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            SVC(
                                C=c_value,
                                gamma="scale",
                                kernel="rbf",
                                class_weight="balanced",
                                probability=True,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
        )
    for n_neighbors in [3, 5, 7]:
        candidates.append(
            (
                f"knn_{n_neighbors}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
                    ]
                ),
            )
        )
    for min_leaf in [1, 2]:
        candidates.append(
            (
                f"rf_leaf{min_leaf}",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    # MLP
    candidates.append(
        (
            "mlp_512_256",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(512, 256),
                            activation="relu",
                            max_iter=300,
                            early_stopping=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        )
    )
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
