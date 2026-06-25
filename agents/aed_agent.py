"""Audio Event Detection auxiliary agent stub."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AEDAgent:
    """Auxiliary Audio Event Detection agent based on PANNs CNN-14."""

    output_dim = 527

    def __init__(self, device: str | None = None) -> None:
        self.device = device or _default_device()

    def extract(self, input: str | Path) -> np.ndarray:
        """Extract AudioSet event logits or probabilities from a WAV file."""
        raise NotImplementedError(
            "AEDAgent is scaffolded. Next step: add CNN-14 AudioSet inference."
        )

    def has_speech(self, input: str | Path, threshold: float = 0.5) -> bool:
        """Return whether the audio likely contains speech."""
        _ = threshold
        raise NotImplementedError(
            "Speech gating will be implemented after AEDAgent.extract is available."
        )


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
