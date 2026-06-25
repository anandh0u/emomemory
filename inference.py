"""Single-video inference entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data import extract_video_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMER inference on one video.")
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"), help="Asset output directory.")
    parser.add_argument("--frame-fps", type=float, default=1.0, help="Frame extraction rate.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    assets = extract_video_assets(args.video, args.output_dir, frame_fps=args.frame_fps)
    print(f"Extracted {len(assets.frame_paths)} frames to {assets.frames_dir}")
    print(f"Extracted audio to {assets.audio_path}")
    raise NotImplementedError("Full inference will be enabled after the FED/SER/TED/AED agents are implemented.")


if __name__ == "__main__":
    main()
