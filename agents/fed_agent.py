"""Facial Emotion Detection agent.

Pipeline:
1. Detect faces in video frames with YOLOv8-Face.
2. Crop/normalize detected faces.
3. Extract deterministic 512-dim embeddings from ResNet-50 features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

ColorMode = Literal["bgr", "rgb"]


class FEDAgent:
    """Facial Emotion Detection agent using YOLOv8-Face and ResNet-50."""

    output_dim = 512

    def __init__(
        self,
        device: str | None = None,
        face_model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
        min_face_size: int = 16,
        image_size: int = 224,
        input_color: ColorMode = "bgr",
        resnet_weights: str | None = "DEFAULT",
    ) -> None:
        self.device = device or _default_device()
        self.face_model_path = _resolve_face_model_path(face_model_path)
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.image_size = image_size
        self.input_color = input_color
        self.resnet_weights = resnet_weights
        self._face_model = None
        self._backbone = None

    def extract(
        self,
        input: np.ndarray | str | Path | Sequence[np.ndarray | str | Path],
    ) -> np.ndarray:
        """Extract one 512-dim embedding per detected face.

        Parameters
        ----------
        input:
            A frame, path to one frame, or a sequence of frames/frame paths.

        Returns
        -------
        np.ndarray
            Shape ``(num_detected_faces, 512)``. Returns an empty array with
            shape ``(0, 512)`` when no faces are detected.
        """
        frames = _normalize_frame_inputs(input)
        if not frames:
            return np.empty((0, self.output_dim), dtype=np.float32)

        self._load_models()
        face_crops: list[np.ndarray] = []
        for frame in frames:
            face_crops.extend(self._detect_face_crops(frame))

        if not face_crops:
            LOGGER.info("FEDAgent found no faces in %d frame(s).", len(frames))
            return np.empty((0, self.output_dim), dtype=np.float32)

        embeddings = self._embed_faces(face_crops)
        LOGGER.info("FEDAgent extracted %d face embedding(s).", embeddings.shape[0])
        return embeddings

    def _load_models(self) -> None:
        if self._face_model is None:
            if self.face_model_path is None:
                raise FileNotFoundError(
                    "YOLOv8-Face weights were not found. Download a YOLOv8 face "
                    "model and pass face_model_path, or place one at "
                    "models/yolov8n-face.pt."
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError("ultralytics is required for FEDAgent. Install requirements.txt first.") from exc
            self._face_model = YOLO(str(self.face_model_path))

        if self._backbone is None:
            self._backbone = _build_resnet50_backbone(self.device, self.resnet_weights)

    def _detect_face_crops(self, frame: np.ndarray) -> list[np.ndarray]:
        assert self._face_model is not None
        results = self._face_model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        height, width = frame.shape[:2]
        crops: list[np.ndarray] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return crops

        for raw_box in boxes.xyxy.detach().cpu().numpy():
            x1, y1, x2, y2 = _clip_box(raw_box, width=width, height=height)
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue
            crops.append(frame[y1:y2, x1:x2].copy())
        return crops

    def _embed_faces(self, face_crops: Sequence[np.ndarray]) -> np.ndarray:
        torch = _import_torch()
        batch = np.stack([self._preprocess_face(crop) for crop in face_crops], axis=0)
        tensor = torch.from_numpy(batch).to(self.device)
        assert self._backbone is not None
        with torch.no_grad():
            features = self._backbone(tensor).flatten(1)
            embeddings = _compress_resnet_features(features, self.output_dim)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)

    def _preprocess_face(self, face: np.ndarray) -> np.ndarray:
        cv2 = _import_cv2()
        face_rgb = _to_rgb(face, self.input_color)
        resized = cv2.resize(face_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (array - mean) / std
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_face_model_path(face_model_path: str | Path | None) -> Path | None:
    if face_model_path is not None:
        return Path(face_model_path)
    for candidate in (
        Path("models/yolov8n-face.pt"),
        Path("weights/yolov8n-face.pt"),
        Path("yolov8n-face.pt"),
    ):
        if candidate.exists():
            return candidate
    return None


def _normalize_frame_inputs(
    input: np.ndarray | str | Path | Sequence[np.ndarray | str | Path],
) -> list[np.ndarray]:
    if isinstance(input, np.ndarray):
        return [input]
    if isinstance(input, (str, Path)):
        return [_read_frame(Path(input))]

    frames: list[np.ndarray] = []
    for item in input:
        if isinstance(item, np.ndarray):
            frames.append(item)
        else:
            frames.append(_read_frame(Path(item)))
    return frames


def _read_frame(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read frame image: {path}")
    return frame


def _build_resnet50_backbone(device: str, weights_name: str | None) -> object:
    torch = _import_torch()
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise ImportError("torchvision is required for FEDAgent. Install requirements.txt first.") from exc

    weights = None
    if weights_name:
        if weights_name == "DEFAULT":
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = ResNet50_Weights[weights_name]

    model = resnet50(weights=weights)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    backbone.to(device)
    backbone.eval()
    return backbone


def _compress_resnet_features(features: object, output_dim: int) -> object:
    torch = _import_torch()
    if features.shape[1] == output_dim:
        return features
    if features.shape[1] > output_dim and features.shape[1] % output_dim == 0:
        return features.reshape(features.shape[0], output_dim, -1).mean(dim=2)
    if features.shape[1] > output_dim:
        return features[:, :output_dim]
    padding = torch.zeros(
        features.shape[0],
        output_dim - features.shape[1],
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, padding], dim=1)


def _clip_box(raw_box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = raw_box[:4]
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    return left, top, right, bottom


def _to_rgb(image: np.ndarray, color_mode: ColorMode) -> np.ndarray:
    if color_mode == "rgb":
        return image
    return image[..., ::-1]


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for FEDAgent. Install requirements.txt first.") from exc
    return torch


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for FEDAgent. Install requirements-core.txt first.") from exc
    return cv2
