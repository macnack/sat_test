#!/usr/bin/env python3
"""Extract frames from all videos in a directory.

Default behavior:
- input videos:  ./test_videos/*.mp4
- output frames: ./test_images/<video_id>/000000.jpg
- GPS sync:      ./test_gps/<video_id>.txt -> ./test_images/<video_id>/gps_metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def extract_with_ffmpeg(video_path: Path, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd += ["-i", str(video_path), "-start_number", "0", str(pattern)]
    proc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL)
    return proc.returncode


@dataclass
class VideoResult:
    video_path: Path
    out_dir: Path
    code: int
    metadata_path: Path | None = None
    metadata_error: str | None = None


def _image_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), path.name
    return 10**12, path.name


def _load_gps_rows(gps_path: Path) -> np.ndarray:
    data = np.loadtxt(gps_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"GPS file must have columns: time,lat,lon,alt. Got {data.shape[1]} in {gps_path}")
    return data


def write_synced_metadata(
    seq_name: str,
    image_dir: Path,
    gps_dir: Path,
    metadata_name: str,
    metadata_indent: int | None,
) -> Path:
    gps_path = gps_dir / f"{seq_name}.txt"
    if not gps_path.exists():
        raise FileNotFoundError(f"GPS file missing for sequence '{seq_name}': {gps_path}")

    image_files = sorted((p for p in image_dir.glob("*.jpg") if p.is_file()), key=_image_sort_key)
    if not image_files:
        raise FileNotFoundError(f"No .jpg frames found in {image_dir}")

    gps = _load_gps_rows(gps_path)
    if len(image_files) != len(gps):
        raise RuntimeError(
            "Cannot sync by index because counts differ for sequence "
            f"'{seq_name}': images={len(image_files)}, gps_rows={len(gps)}"
        )

    items: list[dict[str, Any]] = []
    for idx, (img_path, gps_row) in enumerate(zip(image_files, gps, strict=True)):
        gps_time, lat, lon, alt = gps_row[:4]
        items.append(
            {
                "image": img_path.name,
                "image_index": idx,
                "gps_index": idx,
                "gps_time": float(gps_time),
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": float(alt),
            }
        )

    payload = {
        "sequence": seq_name,
        "sync_method": "index",
        "gps_source": str(gps_path),
        "image_dir": str(image_dir),
        "num_images": len(image_files),
        "num_gps_rows": len(gps),
        "items": items,
    }

    out_path = image_dir / metadata_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=metadata_indent)
        f.write("\n")
    return out_path


def process_video(
    video_path: Path,
    output_dir: Path,
    overwrite: bool,
    no_gps_sync: bool,
    gps_input: Path,
    metadata_name: str,
    metadata_indent: int | None,
) -> VideoResult:
    seq_name = video_path.stem
    out_dir = output_dir / seq_name
    code = extract_with_ffmpeg(video_path, out_dir, overwrite=overwrite)
    if code != 0:
        return VideoResult(video_path=video_path, out_dir=out_dir, code=code)

    if no_gps_sync:
        return VideoResult(video_path=video_path, out_dir=out_dir, code=code)

    try:
        metadata_path = write_synced_metadata(
            seq_name=seq_name,
            image_dir=out_dir,
            gps_dir=gps_input,
            metadata_name=metadata_name,
            metadata_indent=metadata_indent,
        )
        return VideoResult(video_path=video_path, out_dir=out_dir, code=code, metadata_path=metadata_path)
    except Exception as exc:
        return VideoResult(video_path=video_path, out_dir=out_dir, code=code, metadata_error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all video frames to per-video folders.")
    parser.add_argument("--input", type=Path, default=Path("test_videos"), help="Directory containing videos")
    parser.add_argument("--output", type=Path, default=Path("test_images"), help="Directory for extracted frames")
    parser.add_argument("--glob", type=str, default="*.mp4", help="Video glob pattern")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of videos to process in parallel (default: 1).",
    )
    parser.add_argument(
        "--gps-input",
        type=Path,
        default=Path("test_gps"),
        help="GPS directory used to sync poses to frames (default: test_gps).",
    )
    parser.add_argument(
        "--no-gps-sync",
        action="store_true",
        help="Disable GPS sync and metadata generation.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default="gps_metadata.json",
        help="Metadata filename written in each sequence output folder when --gps-input is set.",
    )
    parser.add_argument(
        "--metadata-indent",
        type=int,
        default=2,
        help="JSON indentation for metadata output. Set to 0 for compact output.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    if args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1, got {args.jobs}")
    if args.metadata_indent < 0:
        raise ValueError(f"--metadata-indent must be >= 0, got {args.metadata_indent}")

    if not args.no_gps_sync and not args.gps_input.exists():
        raise FileNotFoundError(f"GPS directory does not exist: {args.gps_input}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in PATH. Install ffmpeg first.")

    videos = sorted(p for p in args.input.glob(args.glob) if p.is_file())
    if not videos:
        raise FileNotFoundError(f"No videos matching '{args.glob}' found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} videos in {args.input}")
    workers = min(args.jobs, len(videos))
    print(f"Parallel workers: {workers}")
    failures = 0
    metadata_failures = 0
    metadata_written = 0
    metadata_indent = args.metadata_indent or None

    futures: list[Future[VideoResult]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for video_path in videos:
            futures.append(
                executor.submit(
                    process_video,
                    video_path,
                    args.output,
                    args.overwrite,
                    args.no_gps_sync,
                    args.gps_input,
                    args.metadata_name,
                    metadata_indent,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            if result.code != 0:
                failures += 1
                print(f"[FAIL] {result.video_path}")
                continue

            print(f"[ OK ] {result.video_path} -> {result.out_dir}")
            if result.metadata_path is not None:
                metadata_written += 1
                print(f"[META] {result.metadata_path}")
            elif result.metadata_error is not None:
                metadata_failures += 1
                print(f"[META-FAIL] {result.video_path.stem}: {result.metadata_error}")

    if failures:
        raise RuntimeError(f"Extraction finished with {failures} failure(s).")
    if metadata_failures:
        raise RuntimeError(
            f"Extraction finished, but metadata syncing failed for {metadata_failures} sequence(s). "
            f"Metadata written for {metadata_written} sequence(s)."
        )
    print("Extraction completed for all videos.")


if __name__ == "__main__":
    main()
