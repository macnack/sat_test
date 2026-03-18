#!/usr/bin/env python3
"""Run the full local pipeline:
1) extract frames (+ GPS metadata sync)
2) visualize GPS tracks (+ stats txt)
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run videos_to_frames + vis_gps_on_osm in sequence.")
    parser.add_argument("--videos-input", type=Path, default=Path("test_videos"))
    parser.add_argument("--images-output", type=Path, default=Path("test_images"))
    parser.add_argument("--gps-input", type=Path, default=Path("test_gps"))
    parser.add_argument("--videos-glob", type=str, default="*.mp4")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--no-gps-sync", action="store_true")
    parser.add_argument("--metadata-name", type=str, default="gps_metadata.json")
    parser.add_argument("--metadata-indent", type=int, default=2)

    parser.add_argument("--traj-output", type=Path, default=Path("test_trajectories"))
    parser.add_argument("--stats-output", type=Path, default=Path("test_trajectories_stats"))
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--tile-url", type=str, default="https://tile.openstreetmap.org/{z}/{x}/{y}.png")
    parser.add_argument("--user-agent", type=str, default="up-count-track-gps-visualizer/1.0")
    parser.add_argument("--tile-cache", type=Path, default=Path("test_trajectories/.tile_cache"))
    parser.add_argument("--viewport-pad-px", type=float, default=120.0)
    parser.add_argument("--viewport-min-span-px", type=float, default=420.0)
    args = parser.parse_args()

    py = sys.executable

    step1 = [
        py,
        "videos_to_frames.py",
        "--input",
        str(args.videos_input),
        "--output",
        str(args.images_output),
        "--glob",
        args.videos_glob,
        "--jobs",
        str(args.jobs),
        "--gps-input",
        str(args.gps_input),
        "--metadata-name",
        args.metadata_name,
        "--metadata-indent",
        str(args.metadata_indent),
    ]
    if args.overwrite_frames:
        step1.append("--overwrite")
    if args.no_gps_sync:
        step1.append("--no-gps-sync")

    step2 = [
        py,
        "vis_gps_on_osm.py",
        "--input",
        str(args.gps_input),
        "--output",
        str(args.traj_output),
        "--stats-output",
        str(args.stats_output),
        "--zoom",
        str(args.zoom),
        "--tile-url",
        args.tile_url,
        "--user-agent",
        args.user_agent,
        "--tile-cache",
        str(args.tile_cache),
        "--viewport-pad-px",
        str(args.viewport_pad_px),
        "--viewport-min-span-px",
        str(args.viewport_min_span_px),
    ]

    run_cmd(step1)
    run_cmd(step2)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
