"""Speech Emotion Recognition agent using transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class SpeechEmotionRecognizer:
    """Speech Emotion Recognition using HuggingFace transformers."""

    # Emotion labels for common datasets
    EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    def __init__(self, device: str | None = None, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """Initialize speech emotion detector.
        
        Args:
            device: Device to run model on (cuda/cpu)
            model_name: HuggingFace model name for speech emotion recognition
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
        """Load the speech emotion recognition model."""
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            import torch
            
            LOGGER.info(f"Loading speech emotion model: {self.model_name}")
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            LOGGER.info("Speech emotion model loaded successfully")
        except ImportError as e:
            LOGGER.error(f"Failed to load transformers: {e}")
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Predict emotion from audio file.
        
        Args:
            audio_path: Path to WAV audio file
            
        Returns:
            Dictionary with emotion, confidence, and all emotion scores
        """
        if not audio_path:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS}
            }
        
        try:
            from transformers import AutoFeatureExtractor
            import torch
            import numpy as np
            
            # For now, return mock result since librosa is not available in Streamlit Cloud
            # In production, would use librosa to load audio
            LOGGER.warning("Speech emotion requires librosa - returning mock result for demo")
            
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": {e: 0.125 for e in self.EMOTIONS},
                "note": "Speech emotion requires librosa - not available in Streamlit Cloud"
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting speech emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS},
                "error": str(e)
            }


# Backward compatibility alias
SERAgent = SpeechEmotionRecognizer
