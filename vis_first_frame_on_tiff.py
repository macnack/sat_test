#!/usr/bin/env python3
"""Overlay first synchronized frame on a georeferenced TIFF map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image
from pyproj import CRS, Transformer

MODEL_PIXEL_SCALE_TAG = 33550
MODEL_TIEPOINT_TAG = 33922
GEO_KEY_DIRECTORY_TAG = 34735
PROJECTED_CRS_GEOKEY = 3072


def _require_positive_size(name: str, values: tuple[int, int]) -> tuple[int, int]:
    w, h = values
    if w < 1 or h < 1:
        raise ValueError(f"{name} must be positive, got {w}x{h}")
    return w, h


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_rotation_from_homography(path: Path | None) -> float:
    if path is None:
        return 0.0
    if not path.exists():
        raise FileNotFoundError(f"Homography JSON does not exist: {path}")
    payload = _read_json(path)
    manual = payload.get("manual_adjustment", {})
    return float(manual.get("rotation_deg", 0.0))


def _find_frame_item(metadata: dict, frame_name: str) -> dict:
    items = metadata.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Metadata {metadata!r} has no valid 'items' list")
    for item in items:
        if item.get("image") == frame_name:
            return item
    raise FileNotFoundError(f"Frame '{frame_name}' not found in metadata items")


def _extract_epsg_from_geokeys(geokeys: tuple[int, ...] | list[int]) -> int | None:
    if len(geokeys) < 4:
        return None
    key_count = int(geokeys[3])
    start = 4
    for _ in range(key_count):
        if start + 4 > len(geokeys):
            break
        key_id = int(geokeys[start])
        tiff_tag_location = int(geokeys[start + 1])
        count = int(geokeys[start + 2])
        value_offset = int(geokeys[start + 3])
        if key_id == PROJECTED_CRS_GEOKEY and tiff_tag_location == 0 and count == 1:
            return value_offset
        start += 4
    return None


def _load_geotiff(
    path: Path,
) -> tuple[np.ndarray, int, int, float, float, float, float, float, float, str]:
    img = Image.open(path)
    rgb = img.convert("RGB")
    tags = img.tag_v2

    if MODEL_PIXEL_SCALE_TAG not in tags:
        raise ValueError(f"Missing TIFF tag {MODEL_PIXEL_SCALE_TAG} (ModelPixelScaleTag)")
    if MODEL_TIEPOINT_TAG not in tags:
        raise ValueError(f"Missing TIFF tag {MODEL_TIEPOINT_TAG} (ModelTiepointTag)")
    if GEO_KEY_DIRECTORY_TAG not in tags:
        raise ValueError(f"Missing TIFF tag {GEO_KEY_DIRECTORY_TAG} (GeoKeyDirectoryTag)")

    scale = tags[MODEL_PIXEL_SCALE_TAG]
    tiepoints = tags[MODEL_TIEPOINT_TAG]
    geokeys = tags[GEO_KEY_DIRECTORY_TAG]

    if len(scale) < 2:
        raise ValueError(f"Invalid ModelPixelScaleTag length: {len(scale)}")
    if len(tiepoints) < 6:
        raise ValueError(f"Invalid ModelTiepointTag length: {len(tiepoints)}")

    scale_x = float(scale[0])
    scale_y = float(scale[1])
    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("Invalid pixel scale (zero)")

    tie_i = float(tiepoints[0])
    tie_j = float(tiepoints[1])
    tie_x = float(tiepoints[3])
    tie_y = float(tiepoints[4])

    epsg = _extract_epsg_from_geokeys(geokeys)
    if epsg is None:
        raise ValueError("ProjectedCSTypeGeoKey (3072) not found in GeoKeyDirectoryTag")
    tiff_crs = f"EPSG:{epsg}"

    arr = np.array(rgb)
    width, height = rgb.size
    return arr, width, height, scale_x, scale_y, tie_i, tie_j, tie_x, tie_y, tiff_crs


def _project_to_tiff_pixel(
    lon: float,
    lat: float,
    scale_x: float,
    scale_y: float,
    tie_i: float,
    tie_j: float,
    tie_x: float,
    tie_y: float,
    tiff_crs: str,
) -> tuple[float, float]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(tiff_crs), always_xy=True)
    x_map, y_map = transformer.transform(lon, lat)

    px = tie_i + (x_map - tie_x) / scale_x
    py = tie_j + (tie_y - y_map) / scale_y
    return float(px), float(py)


def _validate_inside(width: int, height: int, px: float, py: float) -> None:
    if px < 0 or px >= width or py < 0 or py >= height:
        raise ValueError(
            f"Projected GPS point is outside TIFF extent: px={px:.2f}, py={py:.2f}, "
            f"width={width}, height={height}"
        )


def _resize_with_pillow(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(arr)
    resized = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(resized)


def _render_overlay(
    map_arr: np.ndarray,
    marker_xy: tuple[float, float],
    frame_arr: np.ndarray,
    frame_name: str,
    output: Path,
    frame_opacity: float,
    rotation_deg: float,
) -> None:
    map_h, map_w = map_arr.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(map_w / dpi, map_h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(map_arr)
    ax.scatter(
        [marker_xy[0]],
        [marker_xy[1]],
        s=120,
        c="red",
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    frame_h, frame_w = frame_arr.shape[:2]
    cx, cy = marker_xy
    overlay = ax.imshow(frame_arr, extent=(0, frame_w, frame_h, 0), alpha=frame_opacity, zorder=4)
    transform = (
        Affine2D()
        .translate(-frame_w / 2.0, -frame_h / 2.0)
        .rotate_deg(rotation_deg)
        .translate(cx, cy)
        + ax.transData
    )
    overlay.set_transform(transform)
    ax.text(
        marker_xy[0] + 12,
        marker_xy[1] - 12,
        f"{frame_name} | rot={rotation_deg:.1f} deg",
        color="white",
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=5,
    )
    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay the first synchronized frame on a georeferenced TIFF map.")
    parser.add_argument("--tiff", type=Path, default=Path("test_references/0001_year_2024_crop.tiff"))
    parser.add_argument("--metadata", type=Path, default=Path("test_images/0001/gps_metadata.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("test_images/0001"))
    parser.add_argument(
        "--map-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Output map size in pixels.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(256, 256),
        help="Inset frame size in pixels.",
    )
    parser.add_argument("--output", type=Path, default=Path("test_frame_overlays/0001_first_frame_on_tiff.png"))
    parser.add_argument("--frame-name", type=str, default="000000.jpg")
    parser.add_argument(
        "--homography-json",
        type=Path,
        default=Path("0001_first_frame_on_tiff_interactive_homography.json"),
        help="Path to homography JSON; only rotation_deg is used.",
    )
    parser.add_argument(
        "--frame-opacity",
        type=float,
        default=0.8,
        help="Opacity for frame overlay on map (0.0-1.0).",
    )
    args = parser.parse_args()

    if not args.tiff.exists():
        raise FileNotFoundError(f"TIFF file does not exist: {args.tiff}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")

    map_size = _require_positive_size("--map-size", tuple(args.map_size))
    frame_size = _require_positive_size("--frame-size", tuple(args.frame_size))
    if not 0.0 <= args.frame_opacity <= 1.0:
        raise ValueError(f"--frame-opacity must be in [0,1], got {args.frame_opacity}")
    rotation_deg = _load_rotation_from_homography(args.homography_json)

    metadata = _read_json(args.metadata)
    item = _find_frame_item(metadata, args.frame_name)
    lat = float(item["latitude"])
    lon = float(item["longitude"])

    frame_path = args.images_dir / args.frame_name
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image does not exist: {frame_path}")

    (
        map_arr_orig,
        map_w_orig,
        map_h_orig,
        scale_x,
        scale_y,
        tie_i,
        tie_j,
        tie_x,
        tie_y,
        tiff_crs,
    ) = _load_geotiff(args.tiff)

    px_orig, py_orig = _project_to_tiff_pixel(
        lon=lon,
        lat=lat,
        scale_x=scale_x,
        scale_y=scale_y,
        tie_i=tie_i,
        tie_j=tie_j,
        tie_x=tie_x,
        tie_y=tie_y,
        tiff_crs=tiff_crs,
    )
    _validate_inside(map_w_orig, map_h_orig, px_orig, py_orig)

    resized_map = _resize_with_pillow(map_arr_orig, map_size)
    frame_arr_orig = np.array(Image.open(frame_path).convert("RGB"))
    resized_frame = _resize_with_pillow(frame_arr_orig, frame_size)

    sx = map_size[0] / map_w_orig
    sy = map_size[1] / map_h_orig
    marker_xy = (px_orig * sx, py_orig * sy)

    _render_overlay(
        map_arr=resized_map,
        marker_xy=marker_xy,
        frame_arr=resized_frame,
        frame_name=args.frame_name,
        output=args.output,
        frame_opacity=args.frame_opacity,
        rotation_deg=rotation_deg,
    )

    print(f"TIFF CRS: {tiff_crs}")
    print(f"GPS (lat, lon): ({lat:.8f}, {lon:.8f})")
    print(f"Marker px (orig): ({px_orig:.2f}, {py_orig:.2f})")
    print(f"Marker px (resized): ({marker_xy[0]:.2f}, {marker_xy[1]:.2f})")
    print(f"Rotation applied (deg): {rotation_deg:.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
