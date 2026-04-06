#!/usr/bin/env python3
"""Visualize GPS trajectories on OpenStreetMap and save Matplotlib figures.

Input format (per line):
    time_sec,latitude,longitude,altitude

By default this script reads all .txt files from ./test_gps and saves plots to
./test_trajectories.
"""

from __future__ import annotations

import argparse
import math
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np

TILE_SIZE = 256


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def lonlat_to_global_pixels(lon: np.ndarray, lat: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = np.radians(lat)
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def _tile_cache_path(cache_dir: Path, zoom: int, x: int, y: int) -> Path:
    return cache_dir / str(zoom) / str(x) / f"{y}.png"


def fetch_tile(
    x: int,
    y: int,
    zoom: int,
    tile_url: str,
    user_agent: str,
    cache_dir: Path,
    timeout: float = 10.0,
) -> np.ndarray:
    cache_path = _tile_cache_path(cache_dir, zoom, x, y)
    if cache_path.exists():
        data = cache_path.read_bytes()
    else:
        url = tile_url.format(z=zoom, x=x, y=y)
        req = Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "image/png,image/*;q=0.8,*/*;q=0.1",
            },
        )
        with urlopen(req, timeout=timeout) as response:
            data = response.read()
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise URLError(f"Unexpected content type for tile {zoom}/{x}/{y}: {content_type}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)

    tile = plt.imread(BytesIO(data), format="png")
    if tile.dtype != np.uint8:
        tile = (tile * 255).astype(np.uint8)
    if tile.shape[-1] == 3:
        alpha = np.full((tile.shape[0], tile.shape[1], 1), 255, dtype=np.uint8)
        tile = np.concatenate([tile, alpha], axis=-1)
    return tile


def build_basemap(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    tile_url: str,
    user_agent: str,
    cache_dir: Path,
    pad_tiles: int = 1,
):
    x0, y1 = deg2num(max_lat, min_lon, zoom)
    x1, y0 = deg2num(min_lat, max_lon, zoom)

    x0 = max(0, x0 - pad_tiles)
    y0 = y0 + pad_tiles
    x1 = x1 + pad_tiles
    y1 = max(0, y1 - pad_tiles)

    width_tiles = x1 - x0 + 1
    height_tiles = y0 - y1 + 1
    canvas = np.zeros((height_tiles * TILE_SIZE, width_tiles * TILE_SIZE, 4), dtype=np.uint8)

    for tx in range(x0, x1 + 1):
        for ty in range(y1, y0 + 1):
            try:
                tile = fetch_tile(tx, ty, zoom, tile_url=tile_url, user_agent=user_agent, cache_dir=cache_dir)
            except (HTTPError, URLError, ValueError, OSError):
                tile = np.full((TILE_SIZE, TILE_SIZE, 4), 235, dtype=np.uint8)
                tile[:, :, 3] = 255
            ox = (tx - x0) * TILE_SIZE
            oy = (ty - y1) * TILE_SIZE
            canvas[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE, :] = tile

    return canvas, x0, y1


def read_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"File {path} must have at least 4 columns: time,lat,lon,alt")
    t = data[:, 0]
    lat = data[:, 1]
    lon = data[:, 2]
    alt = data[:, 3]
    return t, lat, lon, alt


def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    earth_radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_m * c


def trajectory_length_m(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 2:
        return 0.0
    seg = haversine_distance_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.sum(seg))


def write_stats_txt(track_path: Path, out_dir: Path, t: np.ndarray, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> Path:
    out_path = out_dir / f"{track_path.stem}.txt"
    duration_sec = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    length_m = trajectory_length_m(lat, lon)
    center_lat = float(np.mean(lat))
    center_lon = float(np.mean(lon))
    min_lat = float(np.min(lat))
    max_lat = float(np.max(lat))
    min_lon = float(np.min(lon))
    max_lon = float(np.max(lon))
    lat_range_deg = max_lat - min_lat
    lon_range_deg = max_lon - min_lon
    area_height_m = float(haversine_distance_m(np.array([min_lat]), np.array([center_lon]), np.array([max_lat]), np.array([center_lon]))[0])
    area_width_m = float(haversine_distance_m(np.array([center_lat]), np.array([min_lon]), np.array([center_lat]), np.array([max_lon]))[0])
    area_bbox_m2 = area_width_m * area_height_m
    min_alt = float(np.min(alt))
    max_alt = float(np.max(alt))
    median_alt = float(np.median(alt))

    lines = [
        f"sequence={track_path.stem}",
        f"points={len(t)}",
        f"duration_sec={duration_sec:.6f}",
        f"trajectory_length_m={length_m:.3f}",
        f"trajectory_center_lat={center_lat:.8f}",
        f"trajectory_center_lon={center_lon:.8f}",
        f"area_range_lat_deg={lat_range_deg:.8f}",
        f"area_range_lon_deg={lon_range_deg:.8f}",
        f"area_range_height_m={area_height_m:.3f}",
        f"area_range_width_m={area_width_m:.3f}",
        f"area_range_bbox_m2={area_bbox_m2:.3f}",
        f"altitude_min_m={min_alt:.6f}",
        f"altitude_max_m={max_alt:.6f}",
        f"altitude_median_m={median_alt:.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def compute_viewport(px: np.ndarray, py: np.ndarray, map_w: int, map_h: int, pad_px: float = 120.0, min_span_px: float = 420.0):
    x_min = float(np.min(px))
    x_max = float(np.max(px))
    y_min = float(np.min(py))
    y_max = float(np.max(py))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    span_x = max(x_max - x_min + 2.0 * pad_px, min_span_px)
    span_y = max(y_max - y_min + 2.0 * pad_px, min_span_px)
    span = max(span_x, span_y)

    half = 0.5 * span
    left = cx - half
    right = cx + half
    top = cy - half
    bottom = cy + half

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > map_w:
        left -= right - map_w
        right = map_w
    if bottom > map_h:
        top -= bottom - map_h
        bottom = map_h

    left = max(0.0, left)
    top = max(0.0, top)

    return left, right, top, bottom


def plot_track(
    track_path: Path,
    out_dir: Path,
    zoom: int,
    tile_url: str,
    user_agent: str,
    cache_dir: Path,
    viewport_pad_px: float,
    viewport_min_span_px: float,
) -> Path:
    t, lat, lon, _alt = read_track(track_path)

    min_lat, max_lat = float(np.min(lat)), float(np.max(lat))
    min_lon, max_lon = float(np.min(lon)), float(np.max(lon))

    if min_lat == max_lat:
        min_lat -= 1e-4
        max_lat += 1e-4
    if min_lon == max_lon:
        min_lon -= 1e-4
        max_lon += 1e-4

    basemap, tile_x0, tile_y0 = build_basemap(
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        zoom=zoom,
        tile_url=tile_url,
        user_agent=user_agent,
        cache_dir=cache_dir,
    )

    gx, gy = lonlat_to_global_pixels(lon, lat, zoom)
    px = gx - tile_x0 * TILE_SIZE
    py = gy - tile_y0 * TILE_SIZE

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(basemap)
    ax.plot(px, py, color="red", linewidth=2.2, alpha=0.9, label="trajectory")
    ax.scatter(px[0], py[0], s=40, color="lime", edgecolor="black", linewidth=0.6, label="start", zorder=3)
    ax.scatter(px[-1], py[-1], s=40, color="yellow", edgecolor="black", linewidth=0.6, label="end", zorder=3)

    map_h, map_w = basemap.shape[:2]
    left, right, top, bottom = compute_viewport(
        px,
        py,
        map_w=map_w,
        map_h=map_h,
        pad_px=viewport_pad_px,
        min_span_px=viewport_min_span_px,
    )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    ax.set_title(f"{track_path.stem} | points={len(t)} | zoom={zoom}")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.85)
    plt.tight_layout()

    out_path = out_dir / f"{track_path.stem}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DJI GPS trajectories on OpenStreetMap")
    parser.add_argument("--input", type=Path, default=Path("test_gps"), help="Directory with .txt GPS files")
    parser.add_argument("--output", type=Path, default=Path("test_trajectories"), help="Output directory for .png plots")
    parser.add_argument("--zoom", type=int, default=18, help="OSM tile zoom level (typical: 15-19)")
    parser.add_argument(
        "--tile-url",
        type=str,
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="Tile URL template with placeholders: {z}, {x}, {y}",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="up-count-track-gps-visualizer/1.0",
        help="HTTP User-Agent sent to tile server (set your own if needed)",
    )
    parser.add_argument(
        "--tile-cache",
        type=Path,
        default=Path("test_trajectories/.tile_cache"),
        help="Directory for cached map tiles",
    )
    parser.add_argument(
        "--viewport-pad-px",
        type=float,
        default=120.0,
        help="Extra margin around trajectory in output image (pixels)",
    )
    parser.add_argument(
        "--viewport-min-span-px",
        type=float,
        default=420.0,
        help="Minimum square viewport size in pixels",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Directory for per-trajectory stats .txt files (default: same as --output).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    stats_dir = args.stats_output if args.stats_output is not None else args.output
    stats_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(args.input))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {args.input}")

    print(f"Found {len(files)} GPS files in {args.input}")
    for file_path in files:
        t, lat, lon, alt = read_track(file_path)
        stats_path = write_stats_txt(file_path, stats_dir, t=t, lat=lat, lon=lon, alt=alt)
        out_path = plot_track(
            file_path,
            args.output,
            zoom=args.zoom,
            tile_url=args.tile_url,
            user_agent=args.user_agent,
            cache_dir=args.tile_cache,
            viewport_pad_px=args.viewport_pad_px,
            viewport_min_span_px=args.viewport_min_span_px,
        )
        print(f"Saved stats: {stats_path}")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
