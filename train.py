"""Training entry point for the MMER multi-agent pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import MoseiLoader, MoseiSample, sentiment_to_5_class
from fusion import FeatureAggregator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal emotion fusion pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CMU-MOSEI .csd files.")
    parser.add_argument("--split-dir", type=Path, default=None, help="Optional directory with train/val/test split files.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "acoustic", "text"],
        default=["visual", "acoustic", "text"],
        help="Modalities to load. Use '--modalities text' for a small smoke test before downloading big files.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["aligned", "pooled", "eager"],
        default="aligned",
        help="Use aligned for segment-level interval pooling, pooled for one mean vector per video, or eager for raw arrays.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "mean_std"],
        default="mean",
        help="Temporal summary for each segment. mean_std can improve accuracy by preserving variation.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any labeled segment is missing a modality.")
    parser.add_argument("--summary-only", action="store_true", help="Only load and summarize MOSEI; do not train.")
    parser.add_argument("--build-cache-only", action="store_true", help="Build pooled feature cache and exit.")
    parser.add_argument("--train-model", action="store_true", help="Train and save a fusion classifier.")
    parser.add_argument(
        "--feature-layout",
        choices=["padded", "native"],
        default="padded",
        help="padded keeps fixed 1024-dim slots per modality; native concatenates the real modality dimensions.",
    )
    parser.add_argument(
        "--classifier",
        choices=["auto", "catboost", "catboost_regressor", "mlp", "logreg"],
        default="auto",
        help="Model to train. auto tries a small validation search and keeps the best model.",
    )
    parser.add_argument("--target-dim", type=int, default=1024, help="Per-modality padded feature size.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/mosei_pooled_features.joblib"),
        help="Feature cache path. Reused on later runs to avoid rereading large CSD files.",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild feature cache from CSD files.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/fusion_classifier.joblib"),
        help="Where to save the trained classifier and metrics.",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost boosting iterations.")
    parser.add_argument("--depth", type=int, default=6, help="CatBoost tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoost learning rate.")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="CatBoost early stopping rounds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.build_cache_only:
        features = load_or_build_features(args)
        print(
            "Built feature cache:",
            features["X_train"].shape,
            features["X_val"].shape,
            features["X_test"].shape,
        )
        return

    if args.train_model:
        features = load_or_build_features(args)
        metrics, model = train_classifier(args, features)
        save_training_artifacts(args.model_out, model, metrics, args)
        print(json.dumps(metrics, indent=2))
        print(f"Saved model to {args.model_out}")
        return

    dataset = load_dataset(args)
    print(dataset.summary().to_string(index=False))
    if not args.summary_only:
        print("\nPass --train-model to train and save a classifier.")


def load_dataset(args: argparse.Namespace):
    return MoseiLoader(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        strict=args.strict,
        modalities=args.modalities,
        feature_mode=args.feature_mode,
        pooling=args.pooling,
    ).load()


def load_or_build_features(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.cache_path.exists() and not args.rebuild_cache:
        logging.info("Loading cached features from %s", args.cache_path)
        cached = joblib.load(args.cache_path)
        if cache_matches_args(cached, args):
            return cached
        logging.info("Cached feature layout/config does not match this run; rebuilding %s", args.cache_path)

    dataset = load_dataset(args)
    print(dataset.summary().to_string(index=False))

    logging.info("Building fusion feature arrays")
    modality_dims = infer_modality_dims(dataset.as_dict(), args.modalities) if args.feature_layout == "native" else None
    features = {
        "X_train": samples_to_features(dataset.train, args.modalities, args.target_dim, args.feature_layout, modality_dims),
        "y_train": samples_to_labels(dataset.train),
        "s_train": samples_to_scores(dataset.train),
        "X_val": samples_to_features(dataset.val, args.modalities, args.target_dim, args.feature_layout, modality_dims),
        "y_val": samples_to_labels(dataset.val),
        "s_val": samples_to_scores(dataset.val),
        "X_test": samples_to_features(dataset.test, args.modalities, args.target_dim, args.feature_layout, modality_dims),
        "y_test": samples_to_labels(dataset.test),
        "s_test": samples_to_scores(dataset.test),
        "feature_layout": args.feature_layout,
        "modalities": list(args.modalities),
        "target_dim": args.target_dim,
        "modality_dims": modality_dims,
    }

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features, args.cache_path, compress=3)
    logging.info("Saved feature cache to %s", args.cache_path)
    return features


def cache_matches_args(features: dict[str, Any], args: argparse.Namespace) -> bool:
    if features.get("feature_layout") is None:
        return args.feature_layout == "padded"
    return (
        features.get("feature_layout") == args.feature_layout
        and features.get("modalities") == list(args.modalities)
        and int(features.get("target_dim", args.target_dim)) == args.target_dim
    )


def infer_modality_dims(splits: dict[str, list[MoseiSample]], modalities: list[str]) -> dict[str, int]:
    dims: dict[str, int] = {}
    for modality in modalities:
        dims[modality] = 0
        for samples in splits.values():
            for sample in samples:
                value = getattr(sample, modality)
                if value is not None:
                    dims[modality] = int(np.asarray(value, dtype=np.float32).reshape(-1).shape[0])
                    break
            if dims[modality]:
                break
        if not dims[modality]:
            raise ValueError(f"Could not infer a feature dimension for modality '{modality}'.")
    logging.info("Native modality dimensions: %s", dims)
    return dims


def samples_to_features(
    samples: list[MoseiSample],
    modalities: list[str],
    target_dim: int,
    feature_layout: str = "padded",
    modality_dims: dict[str, int] | None = None,
) -> np.ndarray:
    if feature_layout == "native":
        if modality_dims is None:
            raise ValueError("Native feature layout requires modality_dims.")
        rows = [native_concat(sample, modalities, modality_dims) for sample in samples]
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    aggregator = FeatureAggregator(target_dim=target_dim)
    use_visual = "visual" in modalities
    use_acoustic = "acoustic" in modalities
    use_text = "text" in modalities
    rows = [
        aggregator.transform(
            sample.visual if use_visual else None,
            sample.acoustic if use_acoustic else None,
            sample.text if use_text else None,
        )
        for sample in samples
    ]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def native_concat(sample: MoseiSample, modalities: list[str], modality_dims: dict[str, int]) -> np.ndarray:
    vectors = []
    for modality in modalities:
        value = getattr(sample, modality)
        dim = modality_dims[modality]
        if value is None:
            vector = np.zeros(dim, dtype=np.float32)
        else:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            vector = np.nan_to_num(vector, copy=False)
            if vector.shape[0] < dim:
                padded = np.zeros(dim, dtype=np.float32)
                padded[: vector.shape[0]] = vector
                vector = padded
            elif vector.shape[0] > dim:
                vector = vector[:dim]
        vectors.append(vector)
    return np.concatenate(vectors).astype(np.float32, copy=False)


def samples_to_labels(samples: list[MoseiSample]) -> np.ndarray:
    return np.array([sample.sentiment_label for sample in samples], dtype=np.int64)


def samples_to_scores(samples: list[MoseiSample]) -> np.ndarray:
    return np.array([sample.sentiment_score for sample in samples], dtype=np.float32)


def train_classifier(args: argparse.Namespace, features: dict[str, np.ndarray]) -> tuple[dict[str, Any], object]:
    selected_model = args.classifier
    search_results: list[dict[str, Any]] | None = None

    if args.classifier == "auto":
        model, selected_model, search_results = train_best_classifier(args, features)
    elif args.classifier == "catboost":
        model = train_catboost(args, features)
    elif args.classifier == "catboost_regressor":
        model = train_catboost_regressor(args, features)
    elif args.classifier == "mlp":
        model = train_mlp(args, features)
    else:
        model = train_logreg(features)

    metrics = {
        "validation": evaluate_model(model, features["X_val"], features["y_val"], args.classifier if args.classifier != "auto" else selected_model),
        "test": evaluate_model(model, features["X_test"], features["y_test"], args.classifier if args.classifier != "auto" else selected_model),
        "classifier": args.classifier,
        "selected_model": selected_model,
        "feature_shape": {
            "train": list(features["X_train"].shape),
            "val": list(features["X_val"].shape),
            "test": list(features["X_test"].shape),
        },
    }
    if search_results is not None:
        metrics["search_results"] = search_results
    return metrics, model


def train_best_classifier(args: argparse.Namespace, features: dict[str, np.ndarray]) -> tuple[object, str, list[dict[str, Any]]]:
    candidates: list[tuple[str, object, dict[str, float]]] = []
    search_results: list[dict[str, Any]] = []

    candidate_builders = [
        ("catboost_default", lambda: train_catboost(args, features)),
        (
            "catboost_balanced",
            lambda: train_catboost(
                args,
                features,
                overrides={
                    "auto_class_weights": "Balanced",
                    "depth": args.depth,
                    "learning_rate": args.learning_rate,
                    "iterations": args.iterations,
                },
            ),
        ),
        (
            "catboost_deeper",
            lambda: train_catboost(
                args,
                features,
                overrides={
                    "depth": min(args.depth + 2, 10),
                    "learning_rate": max(args.learning_rate * 0.7, 0.02),
                    "iterations": int(args.iterations * 1.2),
                    "l2_leaf_reg": 5.0,
                    "random_strength": 0.25,
                },
            ),
        ),
        (
            "catboost_regularized",
            lambda: train_catboost(
                args,
                features,
                overrides={
                    "depth": max(args.depth - 1, 4),
                    "learning_rate": min(args.learning_rate * 1.15, 0.15),
                    "iterations": args.iterations,
                    "l2_leaf_reg": 9.0,
                    "random_strength": 0.75,
                },
            ),
        ),
        ("mlp", lambda: train_mlp(args, features)),
        ("logreg_balanced", lambda: train_logreg(features, class_weight="balanced")),
    ]

    for name, builder in candidate_builders:
        model = builder()
        model_kind = "catboost" if name.startswith("catboost") else name
        validation_metrics = evaluate_model(model, features["X_val"], features["y_val"], model_kind)
        candidates.append((name, model, validation_metrics))
        search_results.append({"model": name, "validation": validation_metrics})

    best_name, best_model, _ = max(
        candidates,
        key=lambda item: (item[2]["accuracy"], item[2]["weighted_f1"], -item[2]["mae"]),
    )
    logging.info("Selected %s from auto search.", best_name)
    return best_model, best_name, search_results


def train_catboost(
    args: argparse.Namespace,
    features: dict[str, np.ndarray],
    overrides: dict[str, Any] | None = None,
) -> object:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost is required. Install requirements.txt first.") from exc

    overrides = overrides or {}
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": int(overrides.get("iterations", args.iterations)),
        "depth": int(overrides.get("depth", args.depth)),
        "learning_rate": float(overrides.get("learning_rate", args.learning_rate)),
        "l2_leaf_reg": float(overrides.get("l2_leaf_reg", 3.0)),
        "random_seed": args.random_state,
        "verbose": 100,
        "allow_writing_files": False,
    }
    if overrides.get("random_strength") is not None:
        params["random_strength"] = float(overrides["random_strength"])
    if overrides.get("auto_class_weights") is not None:
        params["auto_class_weights"] = overrides["auto_class_weights"]
    model = CatBoostClassifier(**params)
    model.fit(
        features["X_train"],
        features["y_train"],
        eval_set=(features["X_val"], features["y_val"]),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    return model


def train_catboost_regressor(args: argparse.Namespace, features: dict[str, np.ndarray]) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is required. Install requirements.txt first.") from exc

    if "s_train" not in features or "s_val" not in features:
        raise ValueError("Feature cache does not contain sentiment scores. Rebuild it with --rebuild-cache.")

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.random_state,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        features["X_train"],
        features["s_train"],
        eval_set=(features["X_val"], features["s_val"]),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    return model


def train_logreg(features: dict[str, np.ndarray], class_weight: str | None = None) -> object:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, C=1.0, class_weight=class_weight, n_jobs=-1)),
        ]
    )
    model.fit(features["X_train"], features["y_train"])
    return model


def train_mlp(args: argparse.Namespace, features: dict[str, np.ndarray]) -> object:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(512, 512),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=args.random_state,
                    verbose=True,
                ),
            ),
        ]
    )
    model.fit(features["X_train"], features["y_train"])
    return model


def evaluate_model(model: object, features: np.ndarray, labels: np.ndarray, model_kind: str) -> dict[str, float]:
    raw_predictions = np.asarray(model.predict(features)).reshape(-1)
    if model_kind == "catboost_regressor":
        predictions = np.array([sentiment_to_5_class(score) for score in raw_predictions], dtype=int)
    else:
        predictions = raw_predictions.astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def save_training_artifacts(
    model_out: Path,
    model: object,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metrics": metrics,
        "config": {
            "classifier": args.classifier,
            "selected_model": metrics.get("selected_model", args.classifier),
            "modalities": args.modalities,
            "feature_mode": args.feature_mode,
            "pooling": args.pooling,
            "feature_layout": args.feature_layout,
            "target_dim": args.target_dim,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "random_state": args.random_state,
        },
    }
    joblib.dump(payload, model_out, compress=3)


if __name__ == "__main__":
    main()
