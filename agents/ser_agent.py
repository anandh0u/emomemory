"""Speech Emotion Recognition agent stub."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SERAgent:
    """Speech Emotion Recognition agent using emotion2vec+."""

    output_dim = 256

    def __init__(self, device: str | None = None) -> None:
        self.device = device or _default_device()

    def extract(self, input: str | Path) -> np.ndarray:
        """Extract audio emotion embeddings from a WAV file."""
        raise NotImplementedError(
            "SERAgent is scaffolded. Next step: load emotion2vec/emotion2vec_plus_large "
            "through funasr/HuggingFace and return 256-dim embeddings."
        )


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
