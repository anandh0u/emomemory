"""Text Emotion Detection agent using transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class TextEmotionDetector:
    """Text Emotion Detection using HuggingFace transformers - emo1 model."""

    # Emotion labels for common datasets
    EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    # Alternative emotion set
    EMOTIONS_ALT = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]

    def __init__(self, device: str | None = None, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """Initialize text emotion detector - emo1 model.
        
        Args:
            device: Device to run model on (cuda/cpu)
            model_name: HuggingFace model name for emotion detection
        """
        self.device = device or self._get_device()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        """Load the emotion detection model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            LOGGER.info(f"Loading emotion model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            LOGGER.info("Emotion model loaded successfully")
        except ImportError as e:
            LOGGER.error(f"Failed to load transformers: {e}")
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotion from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion, confidence, and all emotion scores
        """
        if not text or not isinstance(text, str):
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS_ALT}
            }
        
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
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
                "all_emotions": all_emotions,
                "sentiment_score": self._calculate_sentiment(all_emotions)
            }
            
        except Exception as e:
            LOGGER.error(f"Error predicting emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in self.EMOTIONS_ALT},
                "error": str(e)
            }

    def _calculate_sentiment(self, emotions: Dict[str, float]) -> float:
        """Calculate sentiment score from emotions."""
        # Positive emotions: joy, love, happy, surprise
        # Negative emotions: sadness, anger, fear, disgust
        positive = emotions.get("joy", 0) + emotions.get("love", 0) + emotions.get("happy", 0)
        negative = emotions.get("sadness", 0) + emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("disgust", 0)
        
        if positive + negative == 0:
            return 0.0
        
        return (positive - negative) / (positive + negative)


# Backward compatibility alias
TEDAgent = TextEmotionDetector
