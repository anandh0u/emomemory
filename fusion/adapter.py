"""Feature-space adapter for mapping extracted embeddings to MOSEI space."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeFeatureAdapter:
    """StandardScaler plus Ridge regression adapter."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        self.is_fitted = False

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "RidgeFeatureAdapter":
        self.pipeline.fit(source_features, target_features)
        self.is_fitted = True
        return self

    def transform(self, source_features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RidgeFeatureAdapter must be fitted before transform.")
        return np.asarray(self.pipeline.predict(source_features), dtype=np.float32)

    def fit_transform(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        self.fit(source_features, target_features)
        return self.transform(source_features)

    def save(self, path: str | Path) -> None:
        joblib.dump({"alpha": self.alpha, "pipeline": self.pipeline, "is_fitted": self.is_fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeFeatureAdapter":
        payload = joblib.load(path)
        adapter = cls(alpha=payload["alpha"])
        adapter.pipeline = payload["pipeline"]
        adapter.is_fitted = payload["is_fitted"]
        return adapter
