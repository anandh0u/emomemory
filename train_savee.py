"""Train a SAVEE audio-emotion classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.savee_loader import (
    find_default_savee_manifest,
    find_default_savee_raw_dir,
    load_savee_manifest,
    records_to_features,
    records_to_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAVEE audio emotion model.")
    parser.add_argument("--manifest", type=Path, default=find_default_savee_manifest(), help="SAVEE labels CSV.")
    parser.add_argument("--raw-dir", type=Path, default=find_default_savee_raw_dir(), help="Root containing SAVEE audio files.")
    parser.add_argument("--task", choices=["savee7", "binary"], default="savee7", help="7-class SAVEE or binary positive/negative.")
    parser.add_argument("--classifier", choices=["svc", "rf"], default="svc", help="Classifier type.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sampling rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/savee_audio_best_search.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest is None:
        raise FileNotFoundError("Could not find SAVEE manifest. Pass --manifest explicitly.")
    if args.raw_dir is None:
        raise FileNotFoundError("Could not find SAVEE raw directory. Pass --raw-dir explicitly.")

    dataset = load_savee_manifest(args.manifest, raw_dir=args.raw_dir)
    train_records, val_records, test_records, class_names = prepare_task_records(dataset, args.task)
    print("Loaded SAVEE:", json.dumps(split_sizes(dataset), indent=2))
    print("Task:", args.task, class_names)
    
    features = load_features_cached(args, train_records, val_records, test_records)

    model = build_model(args)
    model.fit(features["X_train"], features["y_train"])
    metrics = {
        "train": evaluate(model, features["X_train"], features["y_train"]),
        "validation": evaluate(model, features["X_val"], features["y_val"]),
        "test": evaluate(model, features["X_test"], features["y_test"]),
        "task": args.task,
        "classifier": args.classifier,
        "class_names": class_names,
        "sample_rate": args.sample_rate,
        "manifest": str(dataset.manifest_path),
        "raw_dir": str(dataset.raw_dir),
    }
    payload: dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "class_names": class_names,
        "task": args.task,
        "sample_rate": args.sample_rate,
        "manifest": str(dataset.manifest_path),
        "raw_dir": str(dataset.raw_dir),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print(json.dumps(metrics, indent=2))
    print(f"Saved SAVEE model to {args.model_out}")


def split_sizes(dataset: object) -> dict[str, Any]:
    return {
        "manifest": str(dataset.manifest_path),
        "raw_dir": str(dataset.raw_dir),
        "class_names": dataset.class_names,
        "train": len(dataset.train),
        "val": len(dataset.val),
        "test": len(dataset.test),
    }


def load_features_cached(args: argparse.Namespace, train_records: list[object], val_records: list[object], test_records: list[object]) -> dict[str, np.ndarray]:
    cache_path = Path("artifacts/savee_7class_features.joblib")
    if cache_path.exists():
        print(f"Loading SAVEE features from cache: {cache_path}", flush=True)
        cached = joblib.load(cache_path)
        if args.task == "savee7":
            return cached
        # binary task: filter out neutral (label 4) and map
        # anger(0), disgust(1), fear(2), sadness(5) -> 0
        # happiness(3), surprise(6) -> 1
        label_map = {0: 0, 1: 0, 2: 0, 5: 0, 3: 1, 6: 1}
        features = {}
        for split in ["train", "val", "test"]:
            X = cached[f"X_{split}"]
            y = cached[f"y_{split}"]
            mask = np.isin(y, [0, 1, 2, 3, 5, 6])
            X_filtered = X[mask]
            y_filtered = np.array([label_map[val] for val in y[mask]], dtype=np.int64)
            features[f"X_{split}"] = X_filtered
            features[f"y_{split}"] = y_filtered
        return features

    # Fallback to extracting from scratch
    print("Building SAVEE audio features from raw files...", flush=True)
    return {
        "X_train": records_to_features(train_records, sample_rate=args.sample_rate),
        "y_train": records_to_labels(train_records),
        "X_val": records_to_features(val_records, sample_rate=args.sample_rate),
        "y_val": records_to_labels(val_records),
        "X_test": records_to_features(test_records, sample_rate=args.sample_rate),
        "y_test": records_to_labels(test_records),
    }


def prepare_task_records(dataset: object, task: str) -> tuple[list[object], list[object], list[object], list[str]]:
    if task == "savee7":
        return dataset.train, dataset.val, dataset.test, dataset.class_names

    class_names = ["negative", "positive"]

    def convert(records: list[object]) -> list[object]:
        converted = []
        for record in records:
            if record.label in {"happiness", "surprise"}:
                converted.append(replace(record, label="positive", label_id=1))
            elif record.label in {"anger", "disgust", "fear", "sadness"}:
                converted.append(replace(record, label="negative", label_id=0))
        return converted

    return convert(dataset.train), convert(dataset.val), convert(dataset.test), class_names


def build_model(args: argparse.Namespace) -> object:
    from sklearn.decomposition import PCA
    if args.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=25, random_state=args.random_state)),
            (
                "classifier",
                SVC(
                    C=0.5,
                    gamma="scale",
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=args.random_state,
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
