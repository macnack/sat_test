#!/usr/bin/env python3
"""Interactive tool to manually rotate/translate a frame over a georeferenced TIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, Slider
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


def _find_frame_item(metadata: dict, frame_name: str) -> dict:
    items = metadata.get("items")
    if not isinstance(items, list):
        raise ValueError("Metadata has no valid 'items' list")
    for item in items:
        if item.get("image") == frame_name:
            return item
    raise FileNotFoundError(f"Frame '{frame_name}' not found in metadata items")


def _extract_epsg_from_geokeys(geokeys: tuple[int, ...] | list[int]) -> int | None:
    if len(geokeys) < 4:
        return None
    key_count = int(geokeys[3])
    idx = 4
    for _ in range(key_count):
        if idx + 4 > len(geokeys):
            break
        key_id = int(geokeys[idx])
        tiff_tag_location = int(geokeys[idx + 1])
        count = int(geokeys[idx + 2])
        value_offset = int(geokeys[idx + 3])
        if key_id == PROJECTED_CRS_GEOKEY and tiff_tag_location == 0 and count == 1:
            return value_offset
        idx += 4
    return None


def _load_geotiff(path: Path) -> tuple[np.ndarray, int, int, float, float, float, float, float, float, str]:
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

    arr = np.array(rgb)
    width, height = rgb.size
    return arr, width, height, scale_x, scale_y, tie_i, tie_j, tie_x, tie_y, f"EPSG:{epsg}"


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


def _resize_with_pillow(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(arr)
    resized = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(resized)


def _frame_to_map_homography(cx: float, cy: float, dx: float, dy: float, angle_deg: float, frame_w: int, frame_h: int) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    tx = float(cx + dx)
    ty = float(cy + dy)

    t_center = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    t_local = np.array([[1.0, 0.0, -frame_w / 2.0], [0.0, 1.0, -frame_h / 2.0], [0.0, 0.0, 1.0]], dtype=float)
    return t_center @ r @ t_local


def _frame_to_map_homography_scaled(
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    angle_deg: float,
    scale_x_factor: float,
    scale_y_factor: float,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    tx = float(cx + dx)
    ty = float(cy + dy)

    t_center = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    sxy = np.array([[scale_x_factor, 0.0, 0.0], [0.0, scale_y_factor, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    t_local = np.array([[1.0, 0.0, -frame_w / 2.0], [0.0, 1.0, -frame_h / 2.0], [0.0, 0.0, 1.0]], dtype=float)
    return t_center @ r @ sxy @ t_local


def _add_scale_bar(ax, map_w: int, map_h: int, meters_per_px: float) -> None:
    if meters_per_px <= 0:
        return
    candidates_m = [20.0, 50.0, 100.0, 200.0, 500.0]
    target_px = map_w * 0.18
    best_m = candidates_m[0]
    best_err = float("inf")
    for m in candidates_m:
        px_len = m / meters_per_px
        err = abs(px_len - target_px)
        if err < best_err:
            best_err = err
            best_m = m
    px_len = best_m / meters_per_px
    margin = 24.0
    x0 = margin
    x1 = margin + px_len
    y = map_h - margin
    ax.plot([x0, x1], [y, y], color="white", linewidth=4.0, solid_capstyle="butt", zorder=9)
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.2, solid_capstyle="butt", zorder=10)
    ax.text(
        (x0 + x1) * 0.5,
        y - 10.0,
        f"{int(best_m)} m",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
        zorder=11,
    )


def _render_to_file(
    map_arr: np.ndarray,
    frame_arr: np.ndarray,
    marker_xy_gt: tuple[float, float],
    dx: float,
    dy: float,
    angle_deg: float,
    scale_x_factor: float,
    scale_y_factor: float,
    opacity: float,
    frame_name: str,
    output: Path,
    meters_per_px: float,
) -> None:
    map_h, map_w = map_arr.shape[:2]
    frame_h, frame_w = frame_arr.shape[:2]
    cx_gt, cy_gt = marker_xy_gt

    fig = plt.figure(figsize=(map_w / 100, map_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(map_arr)

    overlay = ax.imshow(frame_arr, extent=(0, frame_w, frame_h, 0), alpha=opacity, zorder=3)
    transform = (
        Affine2D()
        .translate(-frame_w / 2.0, -frame_h / 2.0)
        .scale(scale_x_factor, scale_y_factor)
        .rotate_deg(angle_deg)
        .translate(cx_gt + dx, cy_gt + dy)
        + ax.transData
    )
    overlay.set_transform(transform)
    _add_scale_bar(ax, map_w=map_w, map_h=map_h, meters_per_px=meters_per_px)

    ax.scatter([cx_gt], [cy_gt], s=120, c="red", edgecolors="white", linewidths=1.8, zorder=4)
    ax.scatter([cx_gt + dx], [cy_gt + dy], s=80, c="cyan", edgecolors="black", linewidths=1.0, zorder=5)
    ax.text(
        cx_gt + 12,
        cy_gt - 12,
        f"{frame_name} | rot={angle_deg:.1f} | dx={dx:.1f} | dy={dy:.1f} | sx={scale_x_factor:.2f} sy={scale_y_factor:.2f}",
        color="white",
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=6,
    )
    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _write_pose_json(
    output_json: Path,
    frame_name: str,
    tiff_crs: str,
    map_size: tuple[int, int],
    frame_size: tuple[int, int],
    gt_center: tuple[float, float],
    dx: float,
    dy: float,
    angle_deg: float,
    scale_x_factor: float,
    scale_y_factor: float,
    opacity: float,
    homography: np.ndarray,
) -> None:
    payload = {
        "frame_name": frame_name,
        "tiff_crs": tiff_crs,
        "map_size": {"width": map_size[0], "height": map_size[1]},
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "gt_center_px": {"x": gt_center[0], "y": gt_center[1]},
        "manual_adjustment": {
            "dx_px": dx,
            "dy_px": dy,
            "rotation_deg": angle_deg,
            "scale_x": scale_x_factor,
            "scale_y": scale_y_factor,
            "opacity": opacity,
        },
        "overlay_center_px": {"x": gt_center[0] + dx, "y": gt_center[1] + dy},
        "homography_frame_to_map": homography.tolist(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual rotation/translation of frame overlay on georeferenced TIFF.")
    parser.add_argument("--tiff", type=Path, default=Path("test_references/0001_year_2024_crop.tiff"))
    parser.add_argument("--metadata", type=Path, default=Path("test_images/0001/gps_metadata.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("test_images/0001"))
    parser.add_argument("--frame-name", type=str, default="000000.jpg")
    parser.add_argument("--map-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1024, 1024))
    parser.add_argument("--frame-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(256, 256))
    parser.add_argument("--frame-opacity", type=float, default=0.8)
    parser.add_argument("--initial-angle", type=float, default=0.0)
    parser.add_argument("--initial-scale-x", type=float, default=1.0, help="Initial X scale factor for frame.")
    parser.add_argument("--initial-scale-y", type=float, default=1.0, help="Initial Y scale factor for frame.")
    parser.add_argument("--initial-dx", type=float, default=0.0, help="Initial X shift in map pixels.")
    parser.add_argument("--initial-dy", type=float, default=0.0, help="Initial Y shift in map pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_frame_overlays/0001_first_frame_on_tiff_interactive.png"),
    )
    parser.add_argument(
        "--homography-output",
        type=Path,
        default=Path("test_frame_overlays/0001_first_frame_on_tiff_interactive_homography.json"),
    )
    args = parser.parse_args()

    if not args.tiff.exists():
        raise FileNotFoundError(f"TIFF file does not exist: {args.tiff}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if not 0.0 <= args.frame_opacity <= 1.0:
        raise ValueError(f"--frame-opacity must be in [0,1], got {args.frame_opacity}")
    if args.initial_scale_x <= 0 or args.initial_scale_y <= 0:
        raise ValueError(
            f"--initial-scale-x/--initial-scale-y must be > 0, got {args.initial_scale_x}, {args.initial_scale_y}"
        )

    map_size = _require_positive_size("--map-size", tuple(args.map_size))
    frame_size = _require_positive_size("--frame-size", tuple(args.frame_size))

    metadata = _read_json(args.metadata)
    item = _find_frame_item(metadata, args.frame_name)
    lat = float(item["latitude"])
    lon = float(item["longitude"])

    frame_path = args.images_dir / args.frame_name
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image does not exist: {frame_path}")

    map_arr_orig, map_w_orig, map_h_orig, scale_x, scale_y, tie_i, tie_j, tie_x, tie_y, tiff_crs = _load_geotiff(args.tiff)
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
    if px_orig < 0 or px_orig >= map_w_orig or py_orig < 0 or py_orig >= map_h_orig:
        raise ValueError(f"Projected point outside TIFF: px={px_orig:.2f}, py={py_orig:.2f}")

    map_arr = _resize_with_pillow(map_arr_orig, map_size)
    frame_arr = _resize_with_pillow(np.array(Image.open(frame_path).convert("RGB")), frame_size)

    sx = map_size[0] / map_w_orig
    sy = map_size[1] / map_h_orig
    cx = px_orig * sx
    cy = py_orig * sy
    meters_per_px_x = scale_x / sx
    meters_per_px_y = scale_y / sy
    meters_per_px = 0.5 * (meters_per_px_x + meters_per_px_y)

    map_h, map_w = map_arr.shape[:2]
    frame_h, frame_w = frame_arr.shape[:2]

    fig = plt.figure(figsize=(map_w / 100, map_h / 100), dpi=100)
    ax = fig.add_axes([0.0, 0.22, 1.0, 0.78])
    ax.imshow(map_arr)
    overlay = ax.imshow(frame_arr, extent=(0, frame_w, frame_h, 0), alpha=args.frame_opacity, zorder=3)
    gt_marker = ax.scatter([cx], [cy], s=120, c="red", edgecolors="white", linewidths=1.8, zorder=4)
    overlay_marker = ax.scatter(
        [cx + args.initial_dx],
        [cy + args.initial_dy],
        s=80,
        c="cyan",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )
    label = ax.text(
        cx + 12,
        cy - 12,
        "",
        color="white",
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=6,
    )
    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.set_axis_off()
    _add_scale_bar(ax, map_w=map_w, map_h=map_h, meters_per_px=meters_per_px)

    ax_rot = fig.add_axes([0.12, 0.15, 0.62, 0.025])
    rotation_slider = Slider(ax_rot, "Rotation", -180.0, 180.0, valinit=args.initial_angle, valstep=0.1)

    offset_limit = float(max(map_w, map_h))
    ax_dx = fig.add_axes([0.12, 0.12, 0.62, 0.025])
    dx_slider = Slider(ax_dx, "Offset X", -offset_limit, offset_limit, valinit=args.initial_dx, valstep=1.0)

    ax_dy = fig.add_axes([0.12, 0.09, 0.62, 0.025])
    dy_slider = Slider(ax_dy, "Offset Y", -offset_limit, offset_limit, valinit=args.initial_dy, valstep=1.0)

    ax_op = fig.add_axes([0.12, 0.06, 0.62, 0.025])
    opacity_slider = Slider(ax_op, "Opacity", 0.0, 1.0, valinit=args.frame_opacity, valstep=0.01)

    ax_sx = fig.add_axes([0.12, 0.03, 0.62, 0.022])
    sx_slider = Slider(ax_sx, "Scale X", 0.3, 3.0, valinit=args.initial_scale_x, valstep=0.01)

    ax_sy = fig.add_axes([0.12, 0.002, 0.62, 0.022])
    sy_slider = Slider(ax_sy, "Scale Y", 0.3, 3.0, valinit=args.initial_scale_y, valstep=0.01)

    ax_save = fig.add_axes([0.78, 0.02, 0.18, 0.06])
    save_btn = Button(ax_save, "Save PNG+JSON")

    def _redraw() -> None:
        angle = float(rotation_slider.val)
        dx = float(dx_slider.val)
        dy = float(dy_slider.val)
        scale_x_factor = float(sx_slider.val)
        scale_y_factor = float(sy_slider.val)
        opacity = float(opacity_slider.val)

        overlay.set_alpha(opacity)
        tr = (
            Affine2D()
            .translate(-frame_w / 2.0, -frame_h / 2.0)
            .scale(scale_x_factor, scale_y_factor)
            .rotate_deg(angle)
            .translate(cx + dx, cy + dy)
            + ax.transData
        )
        overlay.set_transform(tr)
        gt_marker.set_offsets(np.array([[cx, cy]]))
        overlay_marker.set_offsets(np.array([[cx + dx, cy + dy]]))
        label.set_text(
            f"{args.frame_name} | rot={angle:.1f} | dx={dx:.1f} | dy={dy:.1f} | sx={scale_x_factor:.2f} sy={scale_y_factor:.2f}"
        )
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        if event.key == "left":
            rotation_slider.set_val(rotation_slider.val - 1.0)
        elif event.key == "right":
            rotation_slider.set_val(rotation_slider.val + 1.0)
        elif event.key == "shift+left":
            rotation_slider.set_val(rotation_slider.val - 5.0)
        elif event.key == "shift+right":
            rotation_slider.set_val(rotation_slider.val + 5.0)
        elif event.key == "up":
            dy_slider.set_val(dy_slider.val - 1.0)
        elif event.key == "down":
            dy_slider.set_val(dy_slider.val + 1.0)
        elif event.key and event.key.lower() == "a":
            dx_slider.set_val(dx_slider.val - 1.0)
        elif event.key and event.key.lower() == "d":
            dx_slider.set_val(dx_slider.val + 1.0)
        elif event.key and event.key.lower() == "s":
            _save_current()

    def _save_current(_event=None) -> None:
        angle = float(rotation_slider.val)
        dx = float(dx_slider.val)
        dy = float(dy_slider.val)
        scale_x_factor = float(sx_slider.val)
        scale_y_factor = float(sy_slider.val)
        opacity = float(opacity_slider.val)

        _render_to_file(
            map_arr=map_arr,
            frame_arr=frame_arr,
            marker_xy_gt=(cx, cy),
            dx=dx,
            dy=dy,
            angle_deg=angle,
            scale_x_factor=scale_x_factor,
            scale_y_factor=scale_y_factor,
            opacity=opacity,
            frame_name=args.frame_name,
            output=args.output,
            meters_per_px=meters_per_px,
        )

        h = _frame_to_map_homography_scaled(
            cx=cx,
            cy=cy,
            dx=dx,
            dy=dy,
            angle_deg=angle,
            scale_x_factor=scale_x_factor,
            scale_y_factor=scale_y_factor,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        _write_pose_json(
            output_json=args.homography_output,
            frame_name=args.frame_name,
            tiff_crs=tiff_crs,
            map_size=map_size,
            frame_size=frame_size,
            gt_center=(cx, cy),
            dx=dx,
            dy=dy,
            angle_deg=angle,
            scale_x_factor=scale_x_factor,
            scale_y_factor=scale_y_factor,
            opacity=opacity,
            homography=h,
        )
        print(
            f"Saved PNG: {args.output} | JSON: {args.homography_output} | "
            f"rot={angle:.2f} dx={dx:.2f} dy={dy:.2f} sx={scale_x_factor:.2f} sy={scale_y_factor:.2f}"
        )

    rotation_slider.on_changed(lambda _v: _redraw())
    dx_slider.on_changed(lambda _v: _redraw())
    dy_slider.on_changed(lambda _v: _redraw())
    sx_slider.on_changed(lambda _v: _redraw())
    sy_slider.on_changed(lambda _v: _redraw())
    opacity_slider.on_changed(lambda _v: _redraw())
    save_btn.on_clicked(_save_current)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    _redraw()
    print("Interactive controls:")
    print(" - Rotation slider or Left/Right arrows")
    print(" - Offset X slider or A/D keys")
    print(" - Offset Y slider or Up/Down arrows")
    print(" - Scale X / Scale Y sliders for anisotropic scaling")
    print(" - Opacity slider")
    print(" - Press 's' or click Save PNG+JSON")
    print(f"PNG output: {args.output}")
    print(f"JSON output: {args.homography_output}")
    plt.show()


if __name__ == "__main__":
    main()
