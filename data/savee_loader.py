"""SAVEE audio emotion loading and feature extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SAVEE_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

LABEL_ALIASES = {
    "a": "anger",
    "ang": "anger",
    "anger": "anger",
    "angry": "anger",
    "d": "disgust",
    "dis": "disgust",
    "disgust": "disgust",
    "f": "fear",
    "fea": "fear",
    "fear": "fear",
    "h": "happiness",
    "hap": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "n": "neutral",
    "neu": "neutral",
    "neutral": "neutral",
    "sa": "sadness",
    "sad": "sadness",
    "sadness": "sadness",
    "su": "surprise",
    "sur": "surprise",
    "surprise": "surprise",
}


@dataclass(frozen=True)
class SaveeRecord:
    sample_id: str
    split: str
    label: str
    label_id: int
    audio_path: Path


@dataclass(frozen=True)
class SaveeDataset:
    train: list[SaveeRecord]
    val: list[SaveeRecord]
    test: list[SaveeRecord]
    class_names: list[str]
    manifest_path: Path
    raw_dir: Path


def find_default_savee_manifest() -> Path | None:
    candidates = [
        Path("datasets/savee/labels_savee_fer_paired.csv"),
        Path(r"E:\emotion_recognition_data\agents\multimodal\manifests\labels_savee_fer_paired.csv"),
        Path(r"E:\emotion_recognition_internship\data\labels.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_default_savee_raw_dir() -> Path | None:
    candidates = [
        Path("datasets/savee"),
        Path(r"E:\emotion_recognition_internship\data\raw"),
        Path(r"E:\emotion_recognition_data\raw"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_savee_manifest(
    manifest_path: str | Path,
    raw_dir: str | Path | None = None,
) -> SaveeDataset:
    manifest_path = Path(manifest_path)
    if raw_dir is None:
        default_raw_dir = find_default_savee_raw_dir()
        if default_raw_dir is None:
            raise FileNotFoundError("Could not infer SAVEE raw directory. Pass --raw-dir explicitly.")
        raw_root = default_raw_dir
    else:
        raw_root = Path(raw_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"SAVEE manifest not found: {manifest_path}")
    if not raw_root.exists():
        raise FileNotFoundError(f"SAVEE raw directory not found: {raw_root}")

    records_by_split: dict[str, list[SaveeRecord]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            audio_value = str(row.get("audio_path") or "").strip()
            if not audio_value:
                continue
            split = normalize_split(row.get("split") or "train")
            label = normalize_label(row.get("label"))
            audio_path = resolve_audio_path(audio_value, raw_root, manifest_path.parent)
            record = SaveeRecord(
                sample_id=str(row.get("sample_id") or f"savee_{index:04d}"),
                split=split,
                label=label,
                label_id=SAVEE_LABELS.index(label),
                audio_path=audio_path,
            )
            records_by_split.setdefault(split, []).append(record)

    if not records_by_split["train"]:
        raise ValueError(f"No SAVEE training rows were loaded from {manifest_path}")
    return SaveeDataset(
        train=records_by_split.get("train", []),
        val=records_by_split.get("val", []),
        test=records_by_split.get("test", []),
        class_names=SAVEE_LABELS[:],
        manifest_path=manifest_path,
        raw_dir=raw_root,
    )


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    raise ValueError(f"Unsupported SAVEE label: {value!r}")


def normalize_split(value: Any) -> str:
    text = str(value or "train").strip().lower()
    if text in {"valid", "validation", "dev"}:
        return "val"
    if text in {"eval", "evaluation"}:
        return "test"
    return text if text in {"train", "val", "test"} else "train"


def resolve_audio_path(value: str, raw_dir: Path, manifest_dir: Path) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.extend(
        [
            raw_dir / path,
            manifest_dir / path,
            raw_dir / "ALL" / path.name,
            Path(r"E:\emotion_recognition_internship\data\raw") / path,
            Path(r"E:\emotion_recognition_internship\data\raw\ALL") / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve SAVEE audio path: {value!r}")


def records_to_features(records: list[SaveeRecord], sample_rate: int = 16_000) -> np.ndarray:
    rows = [extract_audio_features(record.audio_path, sample_rate=sample_rate) for record in records]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def records_to_labels(records: list[SaveeRecord]) -> np.ndarray:
    return np.asarray([record.label_id for record in records], dtype=np.int64)


def extract_audio_features(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    librosa = _import_librosa()
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = np.nan_to_num(waveform.astype(np.float32, copy=False))

    features: list[np.ndarray] = []
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for matrix in (mfcc, delta, delta2):
        features.extend([np.mean(matrix, axis=1), np.std(matrix, axis=1)])

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=32)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    features.extend([np.mean(log_mel, axis=1), np.std(log_mel, axis=1)])

    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rms = librosa.feature.rms(y=waveform)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    for matrix in (chroma, contrast, zcr, rms, centroid, bandwidth, rolloff):
        features.extend([np.mean(matrix, axis=1), np.std(matrix, axis=1)])

    duration = np.asarray([waveform.shape[0] / float(sr)], dtype=np.float32)
    energy = np.asarray([float(np.mean(waveform**2)), float(np.max(np.abs(waveform)))], dtype=np.float32)
    return np.concatenate([*features, duration, energy]).astype(np.float32, copy=False)


def _import_librosa() -> object:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for SAVEE audio features. Install requirements.txt first.") from exc
    return librosa
