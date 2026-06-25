"""Download CMU-MOSEI CSD files with retry/resume support.

The official CMU links sometimes time out. This script can use the official
CMU Multimodal SDK URLs, an unofficial Hugging Face mirror, or try both.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Literal

import requests
from tqdm import tqdm


DEFAULT_MODALITIES = ("glove_vectors", "COVAREP", "OpenFace_2")
HF_MIRROR_BASE = "https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq/resolve/main/data"
Source = Literal["auto", "official", "huggingface"]

CANONICAL_FILENAMES = {
    "glove_vectors": "CMU_MOSEI_TimestampedWordVectors.csd",
    "COVAREP": "CMU_MOSEI_COVAREP.csd",
    "OpenFace_2": "CMU_MOSEI_OpenFace2.csd",
    "FACET 4.2": "CMU_MOSEI_VisualFacet42.csd",
    "All Labels": "CMU_MOSEI_Labels.csd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CMU-MOSEI CSD feature files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/mosei_csd"),
        help="Directory where .csd files will be downloaded.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help="CMU-MOSEI highlevel modality keys to download.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "official", "huggingface"],
        default="auto",
        help="Download source. auto tries official first, then Hugging Face mirror.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=10, help="Download retry attempts per file/source.")
    parser.add_argument("--chunk-mb", type=int, default=4, help="Streaming chunk size in MiB.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mmsdk import mmdatasdk
    except ImportError as exc:
        raise SystemExit(
            "mmsdk is not installed. Run: "
            r".\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .\third_party\CMU-MultimodalSDK"
        ) from exc

    highlevel_urls = {
        key: value
        for key, value in mmdatasdk.cmu_mosei.highlevel.items()
        if key in set(args.modalities)
    }
    missing = sorted(set(args.modalities) - set(highlevel_urls))
    if missing:
        available = ", ".join(sorted(mmdatasdk.cmu_mosei.highlevel))
        raise SystemExit(f"Unknown modality key(s): {missing}. Available: {available}")

    downloads = {**highlevel_urls, **mmdatasdk.cmu_mosei.labels}
    logging.info("Downloading: %s", ", ".join(downloads))
    for key, official_url in downloads.items():
        filename = CANONICAL_FILENAMES[key]
        destination = args.output_dir / filename
        urls = _candidate_urls(
            source=args.source,
            official_url=official_url,
            filename=filename,
        )
        _download_from_candidates(
            name=key,
            urls=urls,
            destination=destination,
            timeout=args.timeout,
            retries=args.retries,
            chunk_size=args.chunk_mb * 1024 * 1024,
            force=args.force,
        )

    logging.info("Download complete. Files in %s:", args.output_dir)
    for path in sorted(args.output_dir.glob("*.csd")):
        logging.info("  %s", path.name)


def _candidate_urls(source: Source, official_url: str, filename: str) -> list[str]:
    huggingface_url = f"{HF_MIRROR_BASE}/{filename}?download=true"
    if source == "official":
        return [official_url]
    if source == "huggingface":
        return [huggingface_url]
    return [official_url, huggingface_url]


def _download_from_candidates(
    name: str,
    urls: list[str],
    destination: Path,
    timeout: int,
    retries: int,
    chunk_size: int,
    force: bool,
) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        logging.info("Skipping %s; file already exists at %s", name, destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            _download_with_retries(
                url=url,
                destination=destination,
                timeout=timeout,
                retries=retries,
                chunk_size=chunk_size,
                force=force,
            )
            return
        except Exception as exc:
            last_error = exc
            logging.warning("Failed downloading %s from %s: %s", name, url, exc)

    raise SystemExit(f"Could not download {name}. Last error: {last_error}")


def _download_with_retries(
    url: str,
    destination: Path,
    timeout: int,
    retries: int,
    chunk_size: int,
    force: bool,
) -> None:
    part_path = destination.with_suffix(destination.suffix + ".part")
    if force:
        if destination.exists():
            destination.unlink()
        if part_path.exists():
            part_path.unlink()

    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            _download_once(
                session=session,
                url=url,
                destination=destination,
                part_path=part_path,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            return
        except Exception:
            if attempt == retries:
                raise
            sleep_seconds = min(60, 2**attempt)
            logging.info("Retrying in %d seconds (%d/%d)", sleep_seconds, attempt, retries)
            time.sleep(sleep_seconds)


def _download_once(
    session: requests.Session,
    url: str,
    destination: Path,
    part_path: Path,
    timeout: int,
    chunk_size: int,
) -> None:
    resume_at = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with session.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
        if response.status_code == 416:
            part_path.replace(destination)
            logging.info("Download already complete: %s", destination)
            return
        response.raise_for_status()

        mode = "ab" if resume_at and response.status_code == 206 else "wb"
        if mode == "wb":
            resume_at = 0

        content_length = int(response.headers.get("content-length", 0))
        total = resume_at + content_length if content_length else None
        logging.info("Downloading %s", destination.name)
        with open(part_path, mode + "") as file_handle:
            with tqdm(
                total=total,
                initial=resume_at,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    progress.update(len(chunk))

    if part_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {part_path}")
    part_path.replace(destination)
    logging.info("Saved %s", destination)


if __name__ == "__main__":
    main()
