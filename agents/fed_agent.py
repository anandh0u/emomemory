"""Facial Emotion Detection agent using transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class FacialEmotionDetector:
    """Facial Emotion Detection using HuggingFace transformers - emo1 model."""

    # Emotion labels for common datasets
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, device: str | None = None, model_name: str = "dima806/facial_emotions_image_detection"):
        """Initialize facial emotion detector - emo1 model.
        
        Args:
            device: Device to run model on (cuda/cpu)
            model_name: HuggingFace model name for facial emotion recognition
        """
        self.device = device or self._get_device()
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self._load_model()

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        """Load the facial emotion recognition model."""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch
            
            LOGGER.info(f"Loading facial emotion model: {self.model_name}")
            
            # Try to load the model with error handling
            try:
                self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                LOGGER.info("Facial emotion model loaded successfully")
            except Exception as model_error:
                LOGGER.error(f"Failed to load model {self.model_name}: {model_error}")
                # Fallback to a lighter model
                LOGGER.info("Falling back to lighter model: RickyIM/vit-base-patch16-224-facial-emotion-recognition")
                self.model_name = "RickyIM/vit-base-patch16-224-facial-emotion-recognition"
                self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                LOGGER.info("Fallback facial emotion model loaded successfully")
            
        except ImportError as e:
            LOGGER.error(f"Failed to load transformers: {e}")
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise

    def predict(self, image_path: str) -> Dict[str, Any]:
        """Predict emotion from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with emotion, confidence, and all emotion scores
        """
        if not image_path:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS}
            }
        
        try:
            from PIL import Image
            import torch
            import numpy as np
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            inputs = self.feature_extractor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get results
            probs = probabilities[0].cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            # Map to emotion labels
            id2label = self.model.config.id2label
            emotion = id2label[predicted_class]
            
            # Create all emotions dict
            all_emotions = {id2label[i]: float(probs[i]) for i in range(len(probs))}
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "all_emotions": all_emotions
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting facial emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS},
                "error": str(e)
            }


# Backward compatibility alias
FEDAgent = FacialEmotionDetector


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
