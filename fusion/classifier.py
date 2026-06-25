"""Fusion classifiers for 5-class sentiment prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ClassifierKind = Literal["catboost", "logreg"]


class FusionClassifier:
    """Train and run a fusion classifier."""

    def __init__(self, kind: ClassifierKind = "catboost", random_state: int = 42) -> None:
        self.kind = kind
        self.random_state = random_state
        self.model = self._build_model()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FusionClassifier":
        self.model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features)).reshape(-1).astype(int)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        predictions = self.predict(features)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
            "mae": float(mean_absolute_error(labels, predictions)),
        }

    def save(self, path: str | Path) -> None:
        joblib.dump({"kind": self.kind, "random_state": self.random_state, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FusionClassifier":
        payload = joblib.load(path)
        classifier = cls(kind=payload["kind"], random_state=payload["random_state"])
        classifier.model = payload["model"]
        return classifier

    def _build_model(self) -> object:
        if self.kind == "catboost":
            try:
                from catboost import CatBoostClassifier
            except Exception as exc:
                raise ImportError("catboost is required for kind='catboost'.") from exc
            return CatBoostClassifier(
                depth=6,
                iterations=1000,
                loss_function="MultiClass",
                random_seed=self.random_state,
                verbose=False,
            )
        if self.kind == "logreg":
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, multi_class="auto")),
                ]
            )
            return GridSearchCV(
                estimator=pipeline,
                param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                n_jobs=-1,
            )
        raise ValueError(f"Unsupported classifier kind: {self.kind}")
