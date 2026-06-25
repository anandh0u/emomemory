"""Video preprocessing utilities for multimodal inference."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoAssets:
    """Extracted video assets used by the agent pipeline."""

    video_path: Path
    output_dir: Path
    frames_dir: Path
    audio_path: Path
    frame_paths: list[Path]
    source_fps: float
    duration_seconds: float
    sample_rate: int


class VideoProcessor:
    """Extract frames and mono WAV audio from an input video."""

    def __init__(
        self,
        frame_fps: float = 1.0,
        sample_rate: int = 16_000,
        image_ext: str = ".jpg",
    ) -> None:
        if frame_fps <= 0:
            raise ValueError("frame_fps must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not image_ext.startswith("."):
            image_ext = f".{image_ext}"
        self.frame_fps = frame_fps
        self.sample_rate = sample_rate
        self.image_ext = image_ext.lower()

    def process(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        overwrite: bool = True,
    ) -> VideoAssets:
        """Extract 1fps frames by default and a 16kHz mono WAV audio file."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames_dir = output_dir / "frames"
        audio_path = output_dir / f"{video_path.stem}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            _remove_matching_files(frames_dir, f"*{self.image_ext}")
            if audio_path.exists():
                audio_path.unlink()

        frame_paths, source_fps, duration = self.extract_frames(video_path, frames_dir)
        self.extract_audio(video_path, audio_path)
        return VideoAssets(
            video_path=video_path,
            output_dir=output_dir,
            frames_dir=frames_dir,
            audio_path=audio_path,
            frame_paths=frame_paths,
            source_fps=source_fps,
            duration_seconds=duration,
            sample_rate=self.sample_rate,
        )

    def extract_frames(self, video_path: Path, frames_dir: Path) -> tuple[list[Path], float, float]:
        """Extract frames at the configured sampling rate."""
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if source_fps <= 0:
                raise RuntimeError(f"Could not determine FPS for video: {video_path}")
            duration = frame_count / source_fps if frame_count > 0 else 0.0
            if duration <= 0:
                duration = _estimate_duration_by_reading(capture, source_fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_paths: list[Path] = []
            timestamp = 0.0
            index = 0
            while timestamp <= duration + 1e-6:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    break
                frame_path = frames_dir / f"frame_{index:06d}{self.image_ext}"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")
                frame_paths.append(frame_path)
                index += 1
                timestamp = index / self.frame_fps
        finally:
            capture.release()

        if not frame_paths:
            raise RuntimeError(f"No frames were extracted from video: {video_path}")
        LOGGER.info("Extracted %d frames from %s", len(frame_paths), video_path)
        return frame_paths, source_fps, duration

    def extract_audio(self, video_path: Path, audio_path: Path) -> Path:
        """Extract a mono WAV track using ffmpeg, with moviepy as a fallback."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-f",
                "wav",
                str(audio_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0 and audio_path.exists():
                LOGGER.info("Extracted audio to %s with ffmpeg", audio_path)
                return audio_path
            LOGGER.warning("ffmpeg audio extraction failed: %s", result.stderr.strip())

        self._extract_audio_with_moviepy(video_path, audio_path)
        if not audio_path.exists():
            raise RuntimeError(f"Audio extraction did not create expected file: {audio_path}")
        LOGGER.info("Extracted audio to %s with moviepy", audio_path)
        return audio_path

    def _extract_audio_with_moviepy(self, video_path: Path, audio_path: Path) -> None:
        try:
            from moviepy import VideoFileClip
        except Exception:
            from moviepy.editor import VideoFileClip

        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                raise RuntimeError(f"Video has no audio track: {video_path}")
            clip.audio.write_audiofile(
                str(audio_path),
                fps=self.sample_rate,
                nbytes=2,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],
                logger=None,
            )


def extract_video_assets(
    video_path: str | Path,
    output_dir: str | Path,
    frame_fps: float = 1.0,
    sample_rate: int = 16_000,
    overwrite: bool = True,
) -> VideoAssets:
    """Convenience wrapper around ``VideoProcessor``."""
    return VideoProcessor(frame_fps=frame_fps, sample_rate=sample_rate).process(
        video_path=video_path,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for frame extraction. Run: pip install -r requirements.txt") from exc
    return cv2


def _estimate_duration_by_reading(capture: object, source_fps: float) -> float:
    frame_count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frame_count += 1
    return frame_count / source_fps


def _remove_matching_files(directory: Path, pattern: str) -> None:
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()
