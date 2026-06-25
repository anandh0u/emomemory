"""Search for the best Animated Storyboard classifier."""

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DEFAULT_FEATURES = Path(r"E:\emotion_recognition_internship\features\animated_embeddings.pt")
DEFAULT_LABELS = Path(r"E:\emotion_recognition_internship\data\labels_animated.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search animated storyboard classifiers.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/animated_best_search.joblib"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_animated_features(args.features, args.labels)
    candidates = build_candidates(args.random_state)
    rows: list[dict[str, Any]] = []
    best = None

    # Convert sparse matrices to dense arrays for classifiers like CatBoost or MLP if needed
    X_train_dense = features["X_train"].toarray() if sparse.issparse(features["X_train"]) else features["X_train"]
    X_val_dense = features["X_val"].toarray() if sparse.issparse(features["X_val"]) else features["X_val"]
    X_test_dense = features["X_test"].toarray() if sparse.issparse(features["X_test"]) else features["X_test"]

    for name, model in candidates:
        print(f"Training {name}...", flush=True)
        try:
            # Use dense representations for MLP and other models to keep it uniform
            model.fit(X_train_dense, features["y_train"])
            row = {
                "model": name,
                "train": evaluate(model, X_train_dense, features["y_train"]),
                "validation": evaluate(model, X_val_dense, features["y_val"]),
                "test": evaluate(model, X_test_dense, features["y_test"]),
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
        except Exception as e:
            print(f"Error training {name}: {e}", flush=True)

    # Train CatBoost Classifier
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
            X_train_dense,
            features["y_train"],
            eval_set=(X_val_dense, features["y_val"]),
            use_best_model=True,
        )
        row = {
            "model": "catboost",
            "train": evaluate(cat_model, X_train_dense, features["y_train"]),
            "validation": evaluate(cat_model, X_val_dense, features["y_val"]),
            "test": evaluate(cat_model, X_test_dense, features["y_test"]),
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
        raise RuntimeError("No animated candidates were trained.")

    best_row, best_model = best[1], best[2]
    metrics = {
        **best_row,
        "task": "animated_binary",
        "classifier": best_row["model"],
        "class_names": features["class_names"],
        "features": str(args.features),
        "labels": str(args.labels),
        "search_results": rows,
    }
    payload = {
        "model": best_model,
        "metrics": metrics,
        "class_names": features["class_names"],
        "task": "animated_binary",
        "features": str(args.features),
        "labels": str(args.labels),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out, compress=3)
    print("BEST", json.dumps(metrics, indent=2), flush=True)
    print(f"Saved animated best model to {args.model_out}", flush=True)


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
    labels_frame = pd.read_csv(label_path)[["sample_id", "script_text", "scene_id"]]
    frame = frame.merge(labels_frame, on="sample_id", how="left")
    frame["script_text"] = frame["script_text"].fillna("")
    frame["scene_id"] = frame["scene_id"].astype(str)
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2)), "script_text"),
            ("scene", OneHotEncoder(handle_unknown="ignore"), ["scene_id"]),
        ]
    )
    metadata = preprocessor.fit_transform(frame)
    matrix = sparse.hstack([metadata, sparse.csr_matrix(embedding_matrix)]).tocsr()

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


def build_candidates(random_state: int) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    # Logistic Regression
    for c_value in [0.01, 0.1, 1.0, 5.0, 10.0]:
        candidates.append(
            (
                f"logreg_C{c_value}",
                Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=False)),
                        ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000)),
                    ]
                ),
            )
        )
    # SVM
    for c_value in [0.1, 1.0, 10.0]:
        candidates.append(
            (
                f"svc_C{c_value}",
                Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=False)),
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
    # Random Forest / Extra Trees
    for estimators in [200, 500]:
        candidates.append(
            (
                f"rf_N{estimators}",
                RandomForestClassifier(
                    n_estimators=estimators,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
        candidates.append(
            (
                f"extra_N{estimators}",
                ExtraTreesClassifier(
                    n_estimators=estimators,
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
                    ("scaler", StandardScaler(with_mean=False)),
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
