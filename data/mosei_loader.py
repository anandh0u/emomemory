"""CMU-MOSEI CSD loader.

This module reads CMU Multimodal Data SDK ``.csd`` files directly with h5py,
discretizes sentiment labels into the requested 5 classes, and returns
train/validation/test splits.

The loader supports three split sources, in this order:
1. Explicit split files in ``split_dir``.
2. Official CMU-MOSEI folds from ``mmsdk`` if that optional package is installed.
3. A deterministic stratified fallback split for local smoke tests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
VALID_MODALITIES = ("visual", "acoustic", "text")
FeatureMode = Literal["aligned", "pooled", "eager"]
PoolingMode = Literal["mean", "mean_std"]

LABEL_TO_NAME: dict[int, str] = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive",
}
NAME_TO_LABEL: dict[str, int] = {name: label for label, name in LABEL_TO_NAME.items()}


@dataclass(frozen=True)
class MoseiSample:
    """One CMU-MOSEI segment with optional modality features."""

    segment_id: str
    video_id: str
    segment_index: int | None
    visual: np.ndarray | None
    acoustic: np.ndarray | None
    text: np.ndarray | None
    sentiment_score: float
    sentiment_label: int


@dataclass(frozen=True)
class MoseiDataset:
    """Loaded CMU-MOSEI splits."""

    train: list[MoseiSample]
    val: list[MoseiSample]
    test: list[MoseiSample]
    split_source: str

    def as_dict(self) -> dict[str, list[MoseiSample]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def summary(self) -> pd.DataFrame:
        """Return per-split label counts as a DataFrame."""
        pd = _import_pandas()
        rows: list[dict[str, int | str]] = []
        for split_name, samples in self.as_dict().items():
            counts = np.bincount([sample.sentiment_label for sample in samples], minlength=5)
            row: dict[str, int | str] = {"split": split_name, "count": len(samples)}
            row.update({LABEL_TO_NAME[index]: int(value) for index, value in enumerate(counts)})
            rows.append(row)
        return pd.DataFrame(rows)


def sentiment_to_5_class(score: float) -> int:
    """Map CMU-MOSEI sentiment score in [-3, 3] to a 5-class label."""
    value = float(score)
    if value < -1.0:
        return 0
    if value < -0.3:
        return 1
    if value <= 0.3:
        return 2
    if value <= 1.0:
        return 3
    return 4


class MoseiLoader:
    """Load CMU-MOSEI features and labels from CSD files."""

    _FILE_PATTERNS = {
        "visual": ("*visual*.csd", "*facet*.csd", "*openface*.csd"),
        "acoustic": ("*acoustic*.csd", "*covarep*.csd", "*audio*.csd"),
        "text": ("*text*.csd", "*glove*.csd", "*word*.csd"),
        "label": ("*label*.csd", "*sentiment*.csd"),
    }

    def __init__(
        self,
        data_dir: str | Path,
        visual_csd: str | Path | None = None,
        acoustic_csd: str | Path | None = None,
        text_csd: str | Path | None = None,
        label_csd: str | Path | None = None,
        split_dir: str | Path | None = None,
        random_state: int = 42,
        validation_size: float = 0.08,
        test_size: float = 0.20,
        strict: bool = False,
        modalities: Sequence[str] = VALID_MODALITIES,
        feature_mode: FeatureMode = "aligned",
        pooling: PoolingMode = "mean",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.visual_csd = Path(visual_csd) if visual_csd else None
        self.acoustic_csd = Path(acoustic_csd) if acoustic_csd else None
        self.text_csd = Path(text_csd) if text_csd else None
        self.label_csd = Path(label_csd) if label_csd else None
        self.split_dir = Path(split_dir) if split_dir else None
        self.random_state = random_state
        self.validation_size = validation_size
        self.test_size = test_size
        self.strict = strict
        self.modalities = _normalize_modalities(modalities)
        self.feature_mode = feature_mode
        if self.feature_mode not in {"aligned", "pooled", "eager"}:
            raise ValueError("feature_mode must be 'aligned', 'pooled', or 'eager'.")
        self.pooling = pooling
        if self.pooling not in {"mean", "mean_std"}:
            raise ValueError("pooling must be 'mean' or 'mean_std'.")

    def load(self) -> MoseiDataset:
        """Load samples and return train/validation/test splits."""
        paths = self._resolve_paths()
        if self.feature_mode == "aligned":
            samples = self._load_aligned_samples(paths)
            dataset = self._split_samples(samples)
            LOGGER.info("Loaded CMU-MOSEI splits from %s", dataset.split_source)
            return dataset

        LOGGER.info("Loading CMU-MOSEI labels from %s", paths["label"])
        labels = self.load_labels(paths["label"])

        visual: dict[str, np.ndarray] = {}
        acoustic: dict[str, np.ndarray] = {}
        text: dict[str, np.ndarray] = {}

        if "visual" in self.modalities:
            LOGGER.info("Loading visual features from %s", paths["visual"])
            visual = self.load_csd_features(
                paths["visual"],
                pooled=self.feature_mode == "pooled",
                pooling=self.pooling,
            )
        if "acoustic" in self.modalities:
            LOGGER.info("Loading acoustic features from %s", paths["acoustic"])
            acoustic = self.load_csd_features(
                paths["acoustic"],
                pooled=self.feature_mode == "pooled",
                pooling=self.pooling,
            )
        if "text" in self.modalities:
            LOGGER.info("Loading text features from %s", paths["text"])
            text = self.load_csd_features(
                paths["text"],
                pooled=self.feature_mode == "pooled",
                pooling=self.pooling,
            )

        samples = self._build_samples(labels, visual, acoustic, text)
        dataset = self._split_samples(samples)
        LOGGER.info("Loaded CMU-MOSEI splits from %s", dataset.split_source)
        return dataset

    def _load_aligned_samples(self, paths: Mapping[str, Path]) -> list[MoseiSample]:
        h5py = _import_h5py()
        LOGGER.info("Loading aligned CMU-MOSEI samples from segment intervals")
        samples: list[MoseiSample] = []
        missing_counts = {"visual": 0, "acoustic": 0, "text": 0}

        with h5py.File(paths["label"], "r") as label_file:
            label_group = _find_data_group(label_file)
            modality_files = {
                modality: h5py.File(paths[modality], "r")
                for modality in self.modalities
                if modality in paths and modality != "label"
            }
            try:
                modality_groups = {name: _find_data_group(file_handle) for name, file_handle in modality_files.items()}
                for video_count, (raw_video_id, label_segment_group) in enumerate(label_group.items(), start=1):
                    video_id = _decode_key(raw_video_id)
                    label_features = _to_numpy(_read_features(label_segment_group))
                    label_intervals = _to_numpy(_read_intervals(label_segment_group))
                    label_rows = _as_label_rows(label_features)

                    modality_sequences = {
                        modality: _read_video_sequence(group, video_id)
                        for modality, group in modality_groups.items()
                    }

                    for segment_index, label_row in enumerate(label_rows):
                        start, end = _label_interval_for_index(label_intervals, segment_index)
                        feature_by_modality: dict[str, np.ndarray | None] = {}
                        for modality, sequence in modality_sequences.items():
                            if sequence is None:
                                feature_by_modality[modality] = None
                            else:
                                features, intervals = sequence
                                feature_by_modality[modality] = _pool_interval(
                                    features,
                                    intervals,
                                    start,
                                    end,
                                    pooling=self.pooling,
                                )

                        missing = [
                            modality
                            for modality in self.modalities
                            if feature_by_modality.get(modality) is None
                        ]
                        for modality in missing:
                            missing_counts[modality] += 1

                        if missing and self.strict:
                            raise KeyError(
                                f"Segment {video_id}[{segment_index}] is missing modality features: {', '.join(missing)}"
                            )

                        samples.append(
                            MoseiSample(
                                segment_id=f"{video_id}[{segment_index}]",
                                video_id=video_id,
                                segment_index=segment_index,
                                visual=feature_by_modality.get("visual"),
                                acoustic=feature_by_modality.get("acoustic"),
                                text=feature_by_modality.get("text"),
                                sentiment_score=_extract_sentiment_score(label_row),
                                sentiment_label=sentiment_to_5_class(_extract_sentiment_score(label_row)),
                            )
                        )

                    if video_count % 500 == 0:
                        LOGGER.info("Aligned %d label videos", video_count)
            finally:
                for file_handle in modality_files.values():
                    file_handle.close()

        LOGGER.info(
            "Built %d aligned samples. Missing modality counts: visual=%d acoustic=%d text=%d",
            len(samples),
            missing_counts["visual"],
            missing_counts["acoustic"],
            missing_counts["text"],
        )
        if not samples:
            raise ValueError("No aligned MOSEI samples were loaded.")
        return samples

    @staticmethod
    def load_csd_features(
        path: str | Path,
        pooled: bool = False,
        pooling: PoolingMode = "mean",
    ) -> dict[str, np.ndarray]:
        """Load the ``features`` dataset for every segment in a CSD file."""
        h5py = _import_h5py()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSD file not found: {path}")

        features_by_id: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as h5_file:
            data_group = _find_data_group(h5_file)
            for index, (raw_key, segment_group) in enumerate(data_group.items(), start=1):
                segment_id = _decode_key(raw_key)
                raw_features = _read_features(segment_group)
                if raw_features is None:
                    LOGGER.debug("Skipping %s in %s because no features dataset was found", segment_id, path)
                    continue
                features = _to_numpy(raw_features)
                features_by_id[segment_id] = _temporal_pool(features, pooling=pooling) if pooled else features
                if index % 500 == 0:
                    LOGGER.info("Loaded %d sequences from %s", index, path.name)

        if not features_by_id:
            raise ValueError(f"No segment features found in {path}")
        return features_by_id

    @classmethod
    def load_labels(cls, path: str | Path) -> dict[str, float]:
        """Load sentiment scores from a labels CSD file."""
        raw_labels = cls.load_csd_features(path)
        labels: dict[str, float] = {}
        for video_id, features in raw_labels.items():
            rows = _as_label_rows(features)
            for index, row in enumerate(rows):
                labels[f"{video_id}[{index}]"] = _extract_sentiment_score(row)
        return labels

    def to_dataframe(self, samples: Sequence[MoseiSample]) -> pd.DataFrame:
        """Create a lightweight metadata DataFrame for inspection."""
        pd = _import_pandas()
        rows = [
            {
                "segment_id": sample.segment_id,
                "video_id": sample.video_id,
                "segment_index": sample.segment_index,
                "sentiment_score": sample.sentiment_score,
                "sentiment_label": sample.sentiment_label,
                "sentiment_name": LABEL_TO_NAME[sample.sentiment_label],
                "visual_shape": _shape_or_none(sample.visual),
                "acoustic_shape": _shape_or_none(sample.acoustic),
                "text_shape": _shape_or_none(sample.text),
            }
            for sample in samples
        ]
        return pd.DataFrame(rows)

    def _resolve_paths(self) -> dict[str, Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MOSEI data directory not found: {self.data_dir}")

        paths = {"label": self.label_csd or self._find_csd("label")}
        if "visual" in self.modalities:
            paths["visual"] = self.visual_csd or self._find_csd("visual")
        if "acoustic" in self.modalities:
            paths["acoustic"] = self.acoustic_csd or self._find_csd("acoustic")
        if "text" in self.modalities:
            paths["text"] = self.text_csd or self._find_csd("text")
        return paths

    def _find_csd(self, kind: str) -> Path:
        for pattern in self._FILE_PATTERNS[kind]:
            matches = sorted(self.data_dir.rglob(pattern))
            if matches:
                return matches[0]
        patterns = ", ".join(self._FILE_PATTERNS[kind])
        raise FileNotFoundError(
            f"Could not infer {kind} CSD file under {self.data_dir}. "
            f"Looked for: {patterns}. Pass {kind}_csd explicitly."
        )

    def _build_samples(
        self,
        labels: Mapping[str, float],
        visual: Mapping[str, np.ndarray],
        acoustic: Mapping[str, np.ndarray],
        text: Mapping[str, np.ndarray],
    ) -> list[MoseiSample]:
        samples: list[MoseiSample] = []
        missing_counts = {"visual": 0, "acoustic": 0, "text": 0}

        for segment_id in sorted(labels):
            visual_features = visual.get(segment_id)
            acoustic_features = acoustic.get(segment_id)
            text_features = text.get(segment_id)
            video_id, segment_index = _parse_segment_id(segment_id)

            # Unaligned MOSEI CSD files are keyed by video id. Use the
            # video-level sequence as a smoke-test fallback until alignment is
            # added for segment-specific features.
            visual_features = visual_features if visual_features is not None else visual.get(video_id)
            acoustic_features = acoustic_features if acoustic_features is not None else acoustic.get(video_id)
            text_features = text_features if text_features is not None else text.get(video_id)

            missing = [
                name
                for name, value in (
                    ("visual", visual_features),
                    ("acoustic", acoustic_features),
                    ("text", text_features),
                )
                if name in self.modalities and value is None
            ]
            for name in missing:
                missing_counts[name] += 1

            if missing and self.strict:
                raise KeyError(f"Segment {segment_id} is missing modality features: {', '.join(missing)}")

            score = float(labels[segment_id])
            samples.append(
                MoseiSample(
                    segment_id=segment_id,
                    video_id=video_id,
                    segment_index=segment_index,
                    visual=visual_features,
                    acoustic=acoustic_features,
                    text=text_features,
                    sentiment_score=score,
                    sentiment_label=sentiment_to_5_class(score),
                )
            )

        LOGGER.info(
            "Built %d samples. Missing modality counts: visual=%d acoustic=%d text=%d",
            len(samples),
            missing_counts["visual"],
            missing_counts["acoustic"],
            missing_counts["text"],
        )
        if not samples:
            raise ValueError("No labeled MOSEI samples were loaded.")
        return samples

    def _split_samples(self, samples: Sequence[MoseiSample]) -> MoseiDataset:
        explicit = self._load_explicit_splits()
        if explicit:
            return self._partition_by_ids(samples, explicit, "explicit split files")

        official = _load_mmsdk_standard_folds()
        if official:
            return self._partition_by_ids(samples, official, "mmsdk official standard folds")

        LOGGER.warning(
            "Official CMU-MOSEI folds were not available. Using deterministic "
            "stratified fallback split; use split_dir or install mmsdk for official folds."
        )
        train, val, test = _stratified_fallback_split(
            list(samples),
            validation_size=self.validation_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return MoseiDataset(train=train, val=val, test=test, split_source="deterministic stratified fallback")

    def _load_explicit_splits(self) -> dict[str, set[str]] | None:
        if not self.split_dir:
            return None
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        aliases = {
            "train": ("train.txt", "training.txt"),
            "val": ("val.txt", "valid.txt", "validation.txt", "dev.txt"),
            "test": ("test.txt", "testing.txt"),
        }
        split_ids: dict[str, set[str]] = {}
        for split_name, filenames in aliases.items():
            path = next((self.split_dir / name for name in filenames if (self.split_dir / name).exists()), None)
            if path is None:
                raise FileNotFoundError(f"Could not find {split_name} split file in {self.split_dir}")
            split_ids[split_name] = _read_id_file(path)
        return split_ids

    @staticmethod
    def _partition_by_ids(
        samples: Sequence[MoseiSample],
        split_ids: Mapping[str, set[str]],
        source: str,
    ) -> MoseiDataset:
        partitions: dict[str, list[MoseiSample]] = {"train": [], "val": [], "test": []}
        for sample in samples:
            for split_name in ("train", "val", "test"):
                ids = split_ids[split_name]
                if sample.segment_id in ids or sample.video_id in ids:
                    partitions[split_name].append(sample)
                    break

        if not all(partitions.values()):
            sizes = {name: len(values) for name, values in partitions.items()}
            raise ValueError(
                f"Split source '{source}' did not match all required partitions. "
                f"Matched sizes: {sizes}"
            )

        return MoseiDataset(
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
            split_source=source,
        )


def _import_h5py() -> object:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read CMU-MOSEI .csd files. Run: pip install -r requirements.txt") from exc
    return h5py


def _normalize_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(modality.lower() for modality in modalities))
    invalid = sorted(set(normalized) - set(VALID_MODALITIES))
    if invalid:
        valid = ", ".join(VALID_MODALITIES)
        raise ValueError(f"Invalid modality value(s): {invalid}. Valid values: {valid}")
    if not normalized:
        raise ValueError("At least one modality must be requested.")
    return normalized


def _import_pandas() -> object:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for MOSEI DataFrame summaries. Run: pip install -r requirements.txt") from exc
    return pd


def _find_data_group(h5_file: object) -> object:
    h5py = _import_h5py()
    if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
        return h5_file["data"]
    groups = [value for value in h5_file.values() if isinstance(value, h5py.Group)]
    rooted_data_groups = [group["data"] for group in groups if "data" in group and isinstance(group["data"], h5py.Group)]
    if len(rooted_data_groups) == 1:
        return rooted_data_groups[0]
    if len(groups) == 1:
        return groups[0]
    raise ValueError("Could not locate CSD data group. Expected a top-level 'data' group.")


def _read_features(segment_group: object) -> np.ndarray | bytes | str | None:
    h5py = _import_h5py()
    if isinstance(segment_group, h5py.Dataset):
        return segment_group[()]
    if "features" in segment_group:
        return segment_group["features"][()]
    datasets = [value for value in segment_group.values() if isinstance(value, h5py.Dataset)]
    if len(datasets) == 1:
        return datasets[0][()]
    return None


def _read_intervals(segment_group: object) -> np.ndarray:
    if hasattr(segment_group, "keys") and "intervals" in segment_group:
        return segment_group["intervals"][()]
    features = _read_features(segment_group)
    if features is None:
        return np.zeros((0, 2), dtype=np.float32)
    array = np.asarray(features)
    length = array.shape[0] if array.ndim > 1 else 1
    return np.column_stack([np.arange(length), np.arange(1, length + 1)]).astype(np.float32)


def _read_video_sequence(data_group: object, video_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    if video_id not in data_group:
        return None
    video_group = data_group[video_id]
    raw_features = _read_features(video_group)
    if raw_features is None:
        return None
    features = _to_numpy(raw_features)
    intervals = _to_numpy(_read_intervals(video_group))
    if features.size == 0 or intervals.size == 0:
        return None
    return features, intervals


def _label_interval_for_index(label_intervals: np.ndarray, index: int) -> tuple[float, float]:
    intervals = np.asarray(label_intervals, dtype=np.float32)
    if intervals.ndim == 1:
        intervals = intervals.reshape(1, -1)
    if index < intervals.shape[0] and intervals.shape[1] >= 2:
        start = float(intervals[index, 0])
        end = float(intervals[index, 1])
        if end > start:
            return start, end
    return float(index), float(index + 1)


def _to_numpy(value: np.ndarray | bytes | str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"S", "O"}:
        return np.vectorize(_decode_value)(array)
    return array.astype(np.float32, copy=False)


def _pool_interval(
    features: np.ndarray,
    intervals: np.ndarray,
    start: float,
    end: float,
    pooling: PoolingMode = "mean",
) -> np.ndarray | None:
    feature_array = np.asarray(features, dtype=np.float32)
    interval_array = np.asarray(intervals, dtype=np.float32)
    if feature_array.size == 0:
        return None
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
    if interval_array.ndim == 1:
        interval_array = interval_array.reshape(1, -1)
    if interval_array.shape[0] != feature_array.shape[0] or interval_array.shape[1] < 2:
        return _temporal_pool(feature_array, pooling=pooling)

    overlaps = (interval_array[:, 0] < end) & (interval_array[:, 1] > start)
    if not np.any(overlaps):
        midpoints = interval_array[:, :2].mean(axis=1)
        overlaps = (midpoints >= start) & (midpoints <= end)
    if not np.any(overlaps):
        return None
    return _temporal_pool(feature_array[overlaps], pooling=pooling)


def _temporal_pool(features: np.ndarray, pooling: PoolingMode = "mean") -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    if array.ndim == 1:
        return np.nan_to_num(array, copy=False)
    mean = np.nanmean(array, axis=0).astype(np.float32, copy=False)
    if pooling == "mean":
        return np.nan_to_num(mean, copy=False)
    std = np.nanstd(array, axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(np.concatenate([mean, std]).astype(np.float32, copy=False), copy=False)


def _extract_sentiment_score(features: np.ndarray) -> float:
    array = np.asarray(features, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Encountered an empty label feature array.")
    squeezed = np.squeeze(array)
    if squeezed.ndim == 0:
        return float(squeezed)
    if squeezed.ndim == 1:
        return float(squeezed[0])
    return float(squeezed[0, 0])


def _as_label_rows(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _parse_segment_id(segment_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<video>.+)\[(?P<segment>\d+)\]$", segment_id)
    if not match:
        return segment_id, None
    return match.group("video"), int(match.group("segment"))


def _read_id_file(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.add(value)
    if not ids:
        raise ValueError(f"Split file is empty: {path}")
    return ids


def _load_mmsdk_standard_folds() -> dict[str, set[str]] | None:
    try:
        from mmsdk import mmdatasdk
    except Exception:
        return None

    standard_folds = getattr(getattr(mmdatasdk, "cmu_mosei", None), "standard_folds", None)
    if standard_folds is None:
        return None

    train = getattr(standard_folds, "standard_train_fold", None)
    val = getattr(standard_folds, "standard_valid_fold", None) or getattr(
        standard_folds, "standard_validation_fold", None
    )
    test = getattr(standard_folds, "standard_test_fold", None)
    if train is None or val is None or test is None:
        return None
    return {"train": set(train), "val": set(val), "test": set(test)}


def _stratified_fallback_split(
    samples: list[MoseiSample],
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[MoseiSample], list[MoseiSample], list[MoseiSample]]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be in (0, 1).")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    indices = np.arange(len(samples))
    labels = np.array([sample.sentiment_label for sample in samples])
    stratify = labels if _can_stratify(labels) else None

    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        LOGGER.warning("scikit-learn unavailable; using non-stratified numpy fallback split.")
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(indices)
        test_count = max(1, int(round(len(samples) * test_size)))
        val_count = max(1, int(round(len(samples) * validation_size)))
        test_idx = shuffled[:test_count]
        val_idx = shuffled[test_count : test_count + val_count]
        train_idx = shuffled[test_count + val_count :]
        return _take(samples, train_idx), _take(samples, val_idx), _take(samples, test_idx)

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_val_labels = labels[train_val_idx]
    val_fraction = validation_size / (1.0 - test_size)
    train_stratify = train_val_labels if _can_stratify(train_val_labels) else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_stratify,
    )
    return _take(samples, train_idx), _take(samples, val_idx), _take(samples, test_idx)


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and bool(np.all(counts >= 2))


def _take(samples: Sequence[MoseiSample], indices: Iterable[int]) -> list[MoseiSample]:
    return [samples[int(index)] for index in indices]


def _shape_or_none(array: np.ndarray | None) -> tuple[int, ...] | None:
    return tuple(array.shape) if array is not None else None


def _decode_key(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
