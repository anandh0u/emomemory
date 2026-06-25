"""Train an animated storyboard optimization classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURES = Path(r"E:\emotion_recognition_internship\features\animated_embeddings.pt")
DEFAULT_LABELS = Path(r"E:\emotion_recognition_internship\data\labels_animated.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train animated storyboard classifier.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="animated_embeddings.pt path.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="labels_animated.csv path.")
    parser.add_argument("--classifier", choices=["logreg", "rf", "extra"], default="logreg")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/animated_classifier.joblib"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_animated_features(args.features, args.labels)
    model = build_model(args)
    model.fit(features["X_train"], features["y_train"])
    metrics = {
        "train": evaluate(model, features["X_train"], features["y_train"]),
        "validation": evaluate(model, features["X_val"], features["y_val"]),
        "test": evaluate(model, features["X_test"], features["y_test"]),
        "classifier": args.classifier,
        "class_names": features["class_names"],
        "features": str(args.features),
        "labels": str(args.labels),
    }
    payload: dict[str, Any] = {
        "model": model,
        "metrics": metrics,
        "task": "animated_binary",
        "classifier": args.classifier,
        "class_names": features["class_names"],
        "features": str(args.features),
        "labels": str(args.labels),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print(json.dumps(metrics, indent=2))
    print(f"Saved animated model to {args.model_out}")


def load_animated_features(feature_path: Path, label_path: Path) -> dict[str, Any]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Animated feature file not found: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Animated labels file not found: {label_path}")
    payload = torch.load(feature_path, map_location="cpu")
    samples = payload["samples"]
    class_names = list(payload.get("class_names", ["not_optimized", "optimized"]))

    rows = []
    embeddings = []
    labels = []
    for sample in samples:
        rows.append({"sample_id": sample["sample_id"], "split": sample["split"]})
        audio = torch.as_tensor(sample["audio_embedding"], dtype=torch.float32).reshape(-1)
        visual = torch.as_tensor(sample["visual_embedding"], dtype=torch.float32).reshape(-1)
        embeddings.append(torch.cat([audio, visual]).numpy())
        labels.append(int(sample["label_id"]))

    frame = pd.DataFrame(rows)
    labels_frame = pd.read_csv(label_path)[["sample_id", "script_text"]]
    frame = frame.merge(labels_frame, on="sample_id", how="left")
    frame["script_text"] = frame["script_text"].fillna("")
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000), "script_text"),
        ]
    )
    metadata_sparse = preprocessor.fit_transform(frame)
    metadata_dense = metadata_sparse.toarray() if hasattr(metadata_sparse, "toarray") else np.asarray(metadata_sparse)
    matrix = np.hstack([metadata_dense, embedding_matrix])

    splits = {
        "train": frame["split"].eq("train").to_numpy(),
        "val": frame["split"].eq("val").to_numpy(),
        "test": frame["split"].eq("test").to_numpy(),
    }
    return {
        "X_train": matrix[splits["train"]],
        "y_train": labels_array[splits["train"]],
        "X_val": matrix[splits["val"]],
        "y_val": labels_array[splits["val"]],
        "X_test": matrix[splits["test"]],
        "y_test": labels_array[splits["test"]],
        "class_names": class_names,
        "preprocessor": preprocessor,
    }


def build_model(args: argparse.Namespace) -> object:
    if args.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=5,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
    if args.classifier == "extra":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=5,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
    if args.classifier == "svc":
        from sklearn.svm import SVC
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=args.random_state)),
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
