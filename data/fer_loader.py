"""FER2013 loading and lightweight feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


FER_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

FER_NUMERIC_TO_LABEL = {index: label for index, label in enumerate(FER_LABELS)}
FER_USAGE_TO_SPLIT = {
    "training": "train",
    "publictest": "val",
    "privatetest": "test",
}
BinaryMode = Literal["fer7", "binary"]


@dataclass(frozen=True)
class FERSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FERDataset:
    train: FERSplit
    val: FERSplit
    test: FERSplit
    class_names: list[str]
    source: Path
    task: BinaryMode


def find_default_fer_csv() -> Path | None:
    candidates = [
        Path("datasets/fer2013.csv"),
        Path("datasets/fer/fer2013.csv"),
        Path(r"E:\emotion_recognition_internship\data\raw\fer2013.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_fer2013_csv(path: str | Path, task: BinaryMode = "fer7") -> FERDataset:
    pd = _import_pandas()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found: {path}")

    frame = pd.read_csv(path, usecols=["emotion", "pixels", "Usage"])
    split_images: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    class_names = ["negative", "positive"] if task == "binary" else FER_LABELS[:]

    for row in frame.itertuples(index=False):
        split = FER_USAGE_TO_SPLIT.get(str(row.Usage).strip().lower())
        if split is None:
            continue
        label = int(row.emotion)
        mapped_label = map_label(label, task)
        if mapped_label is None:
            continue
        split_images[split].append(pixels_to_image(row.pixels))
        split_labels[split].append(mapped_label)

    splits = {
        name: FERSplit(
            images=np.stack(images, axis=0).astype(np.uint8, copy=False),
            labels=np.asarray(split_labels[name], dtype=np.int64),
        )
        for name, images in split_images.items()
    }
    if not all(split.images.size for split in splits.values()):
        sizes = {name: int(split.labels.shape[0]) for name, split in splits.items()}
        raise ValueError(f"FER2013 split loading failed; split sizes: {sizes}")

    return FERDataset(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        class_names=class_names,
        source=path,
        task=task,
    )


def map_label(label: int, task: BinaryMode) -> int | None:
    if task == "fer7":
        return label
    label_name = FER_NUMERIC_TO_LABEL[label]
    if label_name in {"happiness", "surprise"}:
        return 1
    if label_name in {"anger", "disgust", "fear", "sadness"}:
        return 0
    return None


def pixels_to_image(pixels: str) -> np.ndarray:
    values = np.fromstring(pixels, dtype=np.uint8, sep=" ")
    side = int(np.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"FER pixel row has {values.size} values; expected a square image.")
    return values.reshape(side, side)


def image_file_to_array(path: str | Path, image_size: int = 48) -> np.ndarray:
    cv2 = _import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return resize_gray(image, image_size)


def frame_to_face_or_gray(frame: np.ndarray, image_size: int = 48) -> np.ndarray:
    cv2 = _import_cv2()
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(frame, dtype=np.uint8)

    face = detect_largest_face(gray)
    if face is not None:
        gray = face
    return resize_gray(gray, image_size)


def detect_largest_face(gray: np.ndarray) -> np.ndarray | None:
    cv2 = _import_cv2()
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return gray[y : y + h, x : x + w]


def resize_gray(image: np.ndarray, image_size: int = 48) -> np.ndarray:
    cv2 = _import_cv2()
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (image_size, image_size):
        array = cv2.resize(array, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return array


def extract_fer_features(images: np.ndarray, feature_size: int = 24) -> np.ndarray:
    """Return compact raw-pixel plus edge features for CPU-friendly classifiers."""
    cv2 = _import_cv2()
    rows: list[np.ndarray] = []
    for image in images:
        gray = resize_gray(image, 48)
        small = cv2.resize(gray, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32) / 255.0

        equalized = cv2.equalizeHist(gray)
        equalized = cv2.resize(equalized, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
        equalized = equalized.astype(np.float32) / 255.0

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.resize(magnitude, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
        max_value = float(np.max(magnitude)) if magnitude.size else 0.0
        if max_value > 0:
            magnitude = magnitude / max_value

        rows.append(np.concatenate([small.reshape(-1), equalized.reshape(-1), magnitude.reshape(-1)]))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _import_pandas() -> object:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load FER2013 CSV files. Install requirements.txt first.") from exc
    return pd


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for FER feature extraction. Install requirements.txt first.") from exc
    return cv2
