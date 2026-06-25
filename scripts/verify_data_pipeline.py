"""Quick smoke test for MOSEI loading and video preprocessing.

This script is intentionally lightweight: it validates that the current data
pipeline can read CMU-MOSEI CSD files and, optionally, extract frames/audio
from a single video file before the agent stack is wired in.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MoseiLoader
from data.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the MMER data pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CMU-MOSEI .csd files.")
    parser.add_argument("--split-dir", type=Path, default=None, help="Optional directory with train/val/test split files.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "acoustic", "text"],
        default=["visual", "acoustic", "text"],
        help="Modalities to verify.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["aligned", "pooled", "eager"],
        default="aligned",
        help="How MOSEI features should be loaded.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "mean_std"],
        default="mean",
        help="Temporal pooling used for non-aligned features.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail when a requested modality is missing.")
    parser.add_argument("--video", type=Path, default=None, help="Optional video file to preprocess as a smoke test.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke_test"), help="Output directory for extracted assets.")
    parser.add_argument("--frame-fps", type=float, default=1.0, help="Frame sampling rate for video extraction.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sample rate for extracted WAV files.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    loader = MoseiLoader(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        modalities=args.modalities,
        feature_mode=args.feature_mode,
        pooling=args.pooling,
        strict=args.strict,
    )
    dataset = loader.load()

    print(f"MOSEI split source: {dataset.split_source}")
    print(dataset.summary().to_string(index=False))

    if args.video is not None:
        processor = VideoProcessor(frame_fps=args.frame_fps, sample_rate=args.sample_rate)
        video_path = args.video
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths, source_fps, duration_seconds = processor.extract_frames(video_path, frames_dir)
        print(f"Video frames: {len(frame_paths)}")
        print(f"Frame directory: {frames_dir}")
        print(f"Source FPS: {source_fps:.3f}")
        print(f"Duration: {duration_seconds:.3f}s")

        audio_path = output_dir / f"{video_path.stem}.wav"
        try:
            processor.extract_audio(video_path, audio_path)
            print(f"Audio path: {audio_path}")
        except Exception as exc:
            print(f"Audio extraction warning: {exc}")


if __name__ == "__main__":
    main()