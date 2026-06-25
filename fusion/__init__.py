"""Feature fusion components."""

from .adapter import RidgeFeatureAdapter
from .aggregator import FeatureAggregator, aggregate_modalities, pad_or_truncate, temporal_mean_pool
from .classifier import FusionClassifier

__all__ = [
    "FeatureAggregator",
    "FusionClassifier",
    "RidgeFeatureAdapter",
    "aggregate_modalities",
    "pad_or_truncate",
    "temporal_mean_pool",
]
