"""Temporal pooling and feature dimension normalization."""

from __future__ import annotations

import numpy as np


def temporal_mean_pool(features: np.ndarray | None) -> np.ndarray:
    """Mean-pool a sequence of embeddings into a single vector."""
    if features is None:
        return np.zeros(0, dtype=np.float32)
    array = np.asarray(features, dtype=np.float32)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    if array.ndim == 1:
        return np.nan_to_num(array, copy=False)
    return np.nan_to_num(np.nanmean(array, axis=0).astype(np.float32, copy=False), copy=False)


def pad_or_truncate(vector: np.ndarray, target_dim: int = 1024) -> np.ndarray:
    """Pad with zeros or truncate a vector to ``target_dim``."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] == target_dim:
        return array
    if array.shape[0] > target_dim:
        return array[:target_dim]
    output = np.zeros(target_dim, dtype=np.float32)
    output[: array.shape[0]] = array
    return output


def aggregate_modalities(
    fed_embeddings: np.ndarray | None,
    ser_embeddings: np.ndarray | None,
    ted_embeddings: np.ndarray | None,
    target_dim: int = 1024,
) -> np.ndarray:
    """Pool FED/SER/TED embeddings and concatenate to a 3072-dim vector."""
    visual = pad_or_truncate(temporal_mean_pool(fed_embeddings), target_dim)
    speech = pad_or_truncate(temporal_mean_pool(ser_embeddings), target_dim)
    text = pad_or_truncate(temporal_mean_pool(ted_embeddings), target_dim)
    return np.concatenate([visual, speech, text]).astype(np.float32, copy=False)


class FeatureAggregator:
    """Object-oriented wrapper for the fusion aggregation step."""

    def __init__(self, target_dim: int = 1024) -> None:
        self.target_dim = target_dim

    @property
    def output_dim(self) -> int:
        return self.target_dim * 3

    def transform(
        self,
        fed_embeddings: np.ndarray | None,
        ser_embeddings: np.ndarray | None,
        ted_embeddings: np.ndarray | None,
    ) -> np.ndarray:
        return aggregate_modalities(
            fed_embeddings=fed_embeddings,
            ser_embeddings=ser_embeddings,
            ted_embeddings=ted_embeddings,
            target_dim=self.target_dim,
        )
