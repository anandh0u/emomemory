"""Speech Emotion Recognition agent using transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class SpeechEmotionRecognizer:
    """Speech Emotion Recognition using HuggingFace transformers - emo1 model."""

    # Emotion labels for common datasets
    EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    def __init__(self, device: str | None = None, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """Initialize speech emotion detector - emo1 model.
        
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
            
            # Try to load audio with librosa if available
            try:
                import librosa
                import soundfile as sf
                
                # Load audio file
                audio, sr = librosa.load(audio_path, sr=16000)
                
                # Extract features
                inputs = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
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
                
            except ImportError:
                LOGGER.warning("librosa not available - using torch audio loading")
                # Fallback: use torch audio if librosa not available
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000
                
                # Extract features
                inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
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
            LOGGER.error(f"Error predicting speech emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS},
                "error": str(e)
            }


# Backward compatibility alias
SERAgent = SpeechEmotionRecognizer
