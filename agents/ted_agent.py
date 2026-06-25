"""Text Emotion Detection agent stub."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TEDAgent:
    """Text Emotion Detection agent: Whisper ASR followed by FRIDA embeddings."""

    output_dim = 768

    def __init__(self, device: str | None = None) -> None:
        self.device = device or _default_device()

    def extract(self, input: str | Path) -> np.ndarray:
        """Transcribe audio and extract text emotion embeddings."""
        raise NotImplementedError(
            "TEDAgent is scaffolded. Next step: transcribe with "
            "openai/whisper-large-v3-turbo and embed text with ai-forever/FRIDA."
        )


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
