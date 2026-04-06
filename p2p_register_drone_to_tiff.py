#!/usr/bin/env python3
"""Interactive point-to-point registration: drone image -> georeferenced TIFF map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, Slider
from PIL import Image

from vis_first_frame_on_tiff import (
    _find_frame_item,
    _load_geotiff,
    _project_to_tiff_pixel,
    _read_json,
    _require_positive_size,
    _resize_with_pillow,
)


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _rgba_without_black_background(frame_rgb: np.ndarray, black_thr: int) -> np.ndarray:
    frame = frame_rgb.astype(np.uint8, copy=False)
    alpha = np.where(np.max(frame, axis=2) <= int(black_thr), 0, 255).astype(np.uint8)
    return np.dstack((frame, alpha))


def _derive_vfov_deg(hfov_deg: float, frame_w: int, frame_h: int) -> float:
    hfov_rad = math.radians(hfov_deg)
    aspect = float(frame_h) / float(frame_w)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * aspect)
    return math.degrees(vfov_rad)


def _extract_patch(
    map_rgb: np.ndarray,
    cx: float,
    cy: float,
    patch_w: int,
    patch_h: int,
    out_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    map_h, map_w = map_rgb.shape[:2]
    left = int(round(cx - patch_w / 2.0))
    top = int(round(cy - patch_h / 2.0))
    right = left + patch_w
    bottom = top + patch_h

    pad_l = max(0, -left)
    pad_t = max(0, -top)
    pad_r = max(0, right - map_w)
    pad_b = max(0, bottom - map_h)

    l0 = max(0, left)
    t0 = max(0, top)
    r0 = min(map_w, right)
    b0 = min(map_h, bottom)
    patch = map_rgb[t0:b0, l0:r0]
    if pad_l or pad_t or pad_r or pad_b:
        patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    patch_resized = _resize_with_pillow(patch, out_size)
    meta = {
        "left": float(left),
        "top": float(top),
        "patch_w": float(patch_w),
        "patch_h": float(patch_h),
        "view_w": float(patch_resized.shape[1]),
        "view_h": float(patch_resized.shape[0]),
    }
    return patch_resized, meta


def _patch_view_to_global(meta: dict[str, float], x_view: float, y_view: float) -> tuple[float, float]:
    sx = meta["patch_w"] / max(1.0, meta["view_w"])
    sy = meta["patch_h"] / max(1.0, meta["view_h"])
    gx = meta["left"] + x_view * sx
    gy = meta["top"] + y_view * sy
    return float(gx), float(gy)


def _load_pairs_json(path: Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs_raw = payload.get("pairs", [])
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if not isinstance(pairs_raw, list):
        return out
    for p in pairs_raw:
        if not isinstance(p, dict):
            continue
        fp = p.get("frame_xy")
        mp = p.get("map_xy")
        if not isinstance(fp, list) or not isinstance(mp, list) or len(fp) != 2 or len(mp) != 2:
            continue
        out.append(((float(fp[0]), float(fp[1])), (float(mp[0]), float(mp[1]))))
    return out


def _save_pairs_json(
    path: Path,
    frame_name: str,
    center_x: float,
    center_y: float,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> None:
    payload = {
        "frame_name": frame_name,
        "center_map_px": {"x": center_x, "y": center_y},
        "pairs_count": len(pairs),
        "pairs": [
            {"frame_xy": [fp[0], fp[1]], "map_xy": [mp[0], mp[1]]}
            for fp, mp in pairs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resize_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = float(max_side) / float(max(h, w))
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    return _resize_with_pillow(arr, (nw, nh))


def _fit_similarity_closed_form(
    src_uv: np.ndarray,
    dst_xy: np.ndarray,
) -> tuple[float, float]:
    """Fit dst ~= s * R(theta) * src, returning (rot_deg, scale)."""
    sum_u2 = float(np.sum(src_uv[:, 0] ** 2 + src_uv[:, 1] ** 2))
    if sum_u2 < 1e-12:
        raise ValueError("Degenerate source geometry.")
    sum_dot = float(np.sum(dst_xy[:, 0] * src_uv[:, 0] + dst_xy[:, 1] * src_uv[:, 1]))
    sum_cross = float(np.sum(dst_xy[:, 1] * src_uv[:, 0] - dst_xy[:, 0] * src_uv[:, 1]))
    alpha = sum_dot / sum_u2
    beta = sum_cross / sum_u2
    scale = max(0.01, float(math.hypot(alpha, beta)))
    rot_deg = _wrap_angle_deg(math.degrees(math.atan2(beta, alpha)))
    return rot_deg, scale


def _similarity_predict(src_uv: np.ndarray, rot_deg: float, scale: float) -> np.ndarray:
    ang = math.radians(rot_deg)
    c = math.cos(ang)
    s = math.sin(ang)
    x = scale * (c * src_uv[:, 0] - s * src_uv[:, 1])
    y = scale * (s * src_uv[:, 0] + c * src_uv[:, 1])
    return np.column_stack([x, y]).astype(float)


def _fit_similarity_ransac(
    src_uv: np.ndarray,
    dst_xy: np.ndarray,
    iterations: int,
    thresh_px: float,
    min_inliers: int,
    seed: int = 0,
) -> tuple[float, float, np.ndarray, float]:
    n = int(src_uv.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 points.")
    if n == 2:
        rot, scale = _fit_similarity_closed_form(src_uv, dst_xy)
        pred = _similarity_predict(src_uv, rot, scale)
        err = np.linalg.norm(pred - dst_xy, axis=1)
        return rot, scale, np.ones(n, dtype=bool), float(np.sqrt(np.mean(np.square(err))))

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_rmse = float("inf")
    best_rot = 0.0
    best_scale = 1.0

    for _ in range(max(1, iterations)):
        idx = rng.choice(n, size=2, replace=False)
        try:
            rot_try, scale_try = _fit_similarity_closed_form(src_uv[idx], dst_xy[idx])
        except ValueError:
            continue
        pred = _similarity_predict(src_uv, rot_try, scale_try)
        err = np.linalg.norm(pred - dst_xy, axis=1)
        inliers = err <= thresh_px
        count = int(np.sum(inliers))
        if count == 0:
            continue
        rmse_in = float(np.sqrt(np.mean(np.square(err[inliers]))))
        if count > best_count or (count == best_count and rmse_in < best_rmse):
            best_count = count
            best_rmse = rmse_in
            best_inliers = inliers
            best_rot = rot_try
            best_scale = scale_try

    if best_count < max(2, min_inliers):
        # fallback to all-points LS if RANSAC could not find stable inliers
        rot, scale = _fit_similarity_closed_form(src_uv, dst_xy)
        pred = _similarity_predict(src_uv, rot, scale)
        err = np.linalg.norm(pred - dst_xy, axis=1)
        return rot, scale, np.ones(n, dtype=bool), float(np.sqrt(np.mean(np.square(err))))

    rot_refit, scale_refit = _fit_similarity_closed_form(src_uv[best_inliers], dst_xy[best_inliers])
    pred = _similarity_predict(src_uv, rot_refit, scale_refit)
    err = np.linalg.norm(pred - dst_xy, axis=1)
    rmse_all = float(np.sqrt(np.mean(np.square(err))))
    return rot_refit, scale_refit, best_inliers, rmse_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive P2P registration of drone image to GeoTIFF map.")
    parser.add_argument("--tiff", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--frame-name", type=str, required=True)
    parser.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"), default=(640, 640))
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0)
    parser.add_argument("--camera-vfov-deg", type=float, default=None)
    parser.add_argument("--altitude-offset-m", type=float, default=0.0)
    parser.add_argument("--init-rotation-deg", type=float, default=0.0)
    parser.add_argument("--init-scale-mult", type=float, default=1.0)
    parser.add_argument("--init-opacity", type=float, default=0.60)
    parser.add_argument("--black-transparent-threshold", type=int, default=8)
    parser.add_argument("--p2p-orb-max-pairs", type=int, default=24)
    parser.add_argument("--orb-max-side", type=int, default=1400)
    parser.add_argument("--ransac-iters", type=int, default=1200)
    parser.add_argument("--ransac-thresh-px", type=float, default=24.0)
    parser.add_argument("--h-ransac-thresh-px", type=float, default=4.0)
    parser.add_argument("--h-refine-lambda", type=float, default=0.35)
    parser.add_argument("--h-projective-damp", type=float, default=0.25)
    parser.add_argument("--points-json", type=Path, default=Path("test_frame_overlays/p2p_points.json"))
    parser.add_argument("--output-json", type=Path, default=Path("test_frame_overlays/p2p_registration_result.json"))
    args = parser.parse_args()

    if not args.tiff.exists():
        raise FileNotFoundError(f"TIFF file does not exist: {args.tiff}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    frame_size = _require_positive_size("--frame-size", tuple(args.frame_size))
    if args.camera_hfov_deg <= 0.0 or args.camera_hfov_deg >= 179.0:
        raise ValueError(f"--camera-hfov-deg must be in (0, 179), got {args.camera_hfov_deg}")

    metadata = _read_json(args.metadata)
    item = _find_frame_item(metadata, args.frame_name)
    lat = float(item["latitude"])
    lon = float(item["longitude"])
    alt = float(item.get("altitude", 0.0))

    frame_path = args.images_dir / args.frame_name
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image does not exist: {frame_path}")
    frame_rgb_raw = np.array(Image.open(frame_path).convert("RGB"))
    raw_h, raw_w = frame_rgb_raw.shape[:2]
    frame_rgb = _resize_with_pillow(frame_rgb_raw, frame_size)
    frame_rgba = _rgba_without_black_background(frame_rgb, args.black_transparent_threshold)
    frame_w, frame_h = frame_size
    frame_cx = 0.5 * (frame_w - 1.0)
    frame_cy = 0.5 * (frame_h - 1.0)

    map_rgb, map_w, map_h, scale_x, scale_y, tie_i, tie_j, tie_x, tie_y, tiff_crs = _load_geotiff(args.tiff)
    center_x, center_y = _project_to_tiff_pixel(
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
    if not (0.0 <= center_x < map_w and 0.0 <= center_y < map_h):
        raise ValueError(f"Projected image center is outside TIFF: ({center_x:.2f}, {center_y:.2f})")

    vfov_deg = float(args.camera_vfov_deg) if args.camera_vfov_deg is not None else _derive_vfov_deg(
        args.camera_hfov_deg, frame_w, frame_h
    )
    if vfov_deg <= 0.0 or vfov_deg >= 179.0:
        raise ValueError(f"--camera-vfov-deg must be in (0, 179), got {vfov_deg}")
    eff_alt_m = max(0.1, alt + float(args.altitude_offset_m))
    hfov_rad = math.radians(float(args.camera_hfov_deg))
    vfov_rad = math.radians(vfov_deg)
    base_w_px = (2.0 * eff_alt_m * math.tan(hfov_rad / 2.0)) / abs(float(scale_x))
    base_h_px = (2.0 * eff_alt_m * math.tan(vfov_rad / 2.0)) / abs(float(scale_y))
    base_w_px = max(1.0, float(base_w_px))
    base_h_px = max(1.0, float(base_h_px))
    base_sx = base_w_px / float(frame_w)
    base_sy = base_h_px / float(frame_h)

    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = _load_pairs_json(args.points_json)
    pending_frame_pt: list[tuple[float, float] | None] = [None]
    patch_meta: dict[str, float] = {}
    patch_cache: dict[str, np.ndarray] = {}
    homography_h: list[np.ndarray | None] = [None]
    use_homography: list[bool] = [False]

    fig_main = plt.figure(figsize=(12, 9), num="Registration Overlay (Figure 1)")
    ax_main = fig_main.add_axes([0.02, 0.23, 0.96, 0.75])
    ax_main.imshow(map_rgb)
    ax_main.set_xlim(0, map_w)
    ax_main.set_ylim(map_h, 0)
    ax_main.set_axis_off()
    center_marker = ax_main.scatter([center_x], [center_y], c="yellow", s=70, marker="x", linewidths=2.2, zorder=8)
    _ = center_marker
    gps_label = ax_main.text(
        center_x + 10.0,
        center_y - 10.0,
        f"GPS center\nlat={lat:.6f}\nlon={lon:.6f}",
        color="yellow",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=19,
    )
    _ = gps_label
    overlay_img = ax_main.imshow(frame_rgba, extent=(0, frame_w, frame_h, 0), alpha=args.init_opacity, zorder=6)
    homo_overlay_img = ax_main.imshow(np.zeros((map_h, map_w, 4), dtype=np.uint8), zorder=7, visible=False)
    info_text = ax_main.text(
        10,
        18,
        "",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=20,
    )
    loss_text = ax_main.text(
        10,
        44,
        "",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=20,
    )
    status_text = ax_main.text(
        10,
        70,
        "Status: ready",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        zorder=20,
    )
    resid_map_scatter = None
    resid_est_scatter = None
    resid_lines: list[Any] = []
    last_rmse: list[float | None] = [None]
    best_rmse: list[float | None] = [None]

    fig_p2p, (ax_frame, ax_map) = plt.subplots(1, 2, figsize=(12, 6), num="Point-to-Point Tool (Figure 2)")
    frame_artist = ax_frame.imshow(frame_rgb)
    _ = frame_artist
    frame_center_artist = ax_frame.scatter(
        [frame_cx],
        [frame_cy],
        c="yellow",
        s=70,
        marker="x",
        linewidths=2.0,
        zorder=12,
    )
    _ = frame_center_artist
    frame_center_label = ax_frame.text(
        frame_cx + 8.0,
        frame_cy - 8.0,
        "frame center",
        color="yellow",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
        zorder=12,
    )
    _ = frame_center_label
    map_patch_artist = ax_map.imshow(np.zeros_like(frame_rgb))
    _ = map_patch_artist
    ax_frame.set_title("Frame (click point)")
    ax_map.set_title("Map patch (click corresponding point)")
    ax_frame.set_xlim(0, frame_w)
    ax_frame.set_ylim(frame_h, 0)
    ax_map.set_xlim(0, frame_w)
    ax_map.set_ylim(frame_h, 0)
    frame_pts_artist = None
    map_pts_artist = None
    pending_artist = None

    ax_rot = fig_main.add_axes([0.08, 0.16, 0.58, 0.03])
    s_rot = Slider(ax_rot, "Rotation", -180.0, 180.0, valinit=float(args.init_rotation_deg), valstep=0.1)
    ax_scale = fig_main.add_axes([0.08, 0.12, 0.58, 0.03])
    s_scale = Slider(ax_scale, "Scale", 0.05, 6.0, valinit=float(args.init_scale_mult), valstep=0.01)
    ax_alpha = fig_main.add_axes([0.08, 0.08, 0.58, 0.03])
    s_alpha = Slider(ax_alpha, "Opacity", 0.0, 1.0, valinit=float(args.init_opacity), valstep=0.01)

    ax_btn_orb = fig_main.add_axes([0.72, 0.16, 0.24, 0.05])
    b_orb = Button(ax_btn_orb, "ORB Suggest (O)")
    ax_btn_h = fig_main.add_axes([0.72, 0.22, 0.24, 0.05])
    b_h = Button(ax_btn_h, "Fit Homography (H)")
    ax_btn_err = fig_main.add_axes([0.72, 0.28, 0.24, 0.05])
    b_err = Button(ax_btn_err, "Print Err (E)")
    ax_btn_two = fig_main.add_axes([0.72, 0.34, 0.24, 0.05])
    b_two = Button(ax_btn_two, "Two-Stage (T)")
    ax_btn_fit = fig_main.add_axes([0.72, 0.10, 0.24, 0.05])
    b_fit = Button(ax_btn_fit, "Fit Rot+Scale (R)")
    ax_btn_clear = fig_main.add_axes([0.72, 0.04, 0.11, 0.05])
    b_clear = Button(ax_btn_clear, "Clear (X)")
    ax_btn_save = fig_main.add_axes([0.85, 0.04, 0.11, 0.05])
    b_save = Button(ax_btn_save, "Save (S)")

    def _transform_for_display(rot_deg: float, scale_mult: float) -> Affine2D:
        return (
            Affine2D()
            .translate(-frame_cx, -frame_cy)
            .scale(base_sx * scale_mult, base_sy * scale_mult)
            .rotate_deg(rot_deg)
            .translate(center_x, center_y)
        )

    def _status(msg: str) -> None:
        status_text.set_text(f"Status: {msg}")
        fig_main.canvas.draw_idle()
        print(msg)

    def _predict_map_point(frame_pt: tuple[float, float], rot_deg: float, scale_mult: float) -> tuple[float, float]:
        px = float(frame_pt[0]) - frame_cx
        py = float(frame_pt[1]) - frame_cy
        vx = px * base_sx * scale_mult
        vy = py * base_sy * scale_mult
        ang = math.radians(rot_deg)
        rx = math.cos(ang) * vx - math.sin(ang) * vy
        ry = math.sin(ang) * vx + math.cos(ang) * vy
        return center_x + rx, center_y + ry

    def _print_pair_errors(rot_deg: float, scale_mult: float) -> None:
        if not pairs:
            print("Pairs: 0")
            return
        errs = []
        signs = []
        for idx, (fp, mp) in enumerate(pairs, start=1):
            ex, ey = _predict_map_point(fp, rot_deg, scale_mult)
            dx = mp[0] - ex
            dy = mp[1] - ey
            err = float(math.hypot(dx, dy))
            errs.append(err)
            v_img = np.array([fp[0] - frame_cx, fp[1] - frame_cy], dtype=float)
            v_map = np.array([mp[0] - center_x, mp[1] - center_y], dtype=float)
            r_img = float(np.hypot(v_img[0], v_img[1])) + 1e-9
            r_map = float(np.hypot(v_map[0], v_map[1]))
            signs.append(r_map - (r_img * (base_sx + base_sy) * 0.5 * scale_mult))
            print(f"pair {idx:02d}: err={err:.2f}px dx={dx:.2f} dy={dy:.2f}")
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        mean = float(np.mean(errs))
        trend = float(np.mean(signs))
        if trend > 1.0:
            hint = "scale likely too SMALL"
        elif trend < -1.0:
            hint = "scale likely too LARGE"
        else:
            hint = "scale near balanced"
        print(f"pairs={len(pairs)} mean={mean:.2f}px rmse={rmse:.2f}px -> {hint}")

    def _draw_residuals(rot_deg: float, scale_mult: float) -> float | None:
        nonlocal resid_map_scatter, resid_est_scatter, resid_lines
        if resid_map_scatter is not None:
            resid_map_scatter.remove()
            resid_map_scatter = None
        if resid_est_scatter is not None:
            resid_est_scatter.remove()
            resid_est_scatter = None
        for art in resid_lines:
            art.remove()
        resid_lines = []
        if not pairs:
            fig_main.canvas.draw_idle()
            return None
        map_pts = np.array([mp for _fp, mp in pairs], dtype=float)
        est_pts = np.array([_predict_map_point(fp, rot_deg, scale_mult) for fp, _mp in pairs], dtype=float)
        resid_map_scatter = ax_main.scatter(map_pts[:, 0], map_pts[:, 1], c="magenta", s=36, zorder=12)
        resid_est_scatter = ax_main.scatter(est_pts[:, 0], est_pts[:, 1], c="lime", s=30, marker="+", zorder=12)
        for (mx, my), (ex, ey) in zip(map_pts, est_pts, strict=True):
            resid_lines.append(ax_main.plot([mx, ex], [my, ey], color="white", linewidth=1.0, alpha=0.9, zorder=11)[0])
        rmse = float(np.sqrt(np.mean(np.square(np.linalg.norm(map_pts - est_pts, axis=1)))))
        fig_main.canvas.draw_idle()
        return rmse

    def _refresh_patch_view() -> float | None:
        rot = float(s_rot.val)
        scale_mult = float(s_scale.val)
        patch_w = max(8, int(round(base_w_px * scale_mult * 1.5)))
        patch_h = max(8, int(round(base_h_px * scale_mult * 1.5)))
        patch_rgb, meta = _extract_patch(map_rgb, center_x, center_y, patch_w, patch_h, out_size=frame_size)
        patch_meta.clear()
        patch_meta.update(meta)
        patch_cache["rgb"] = patch_rgb
        nonlocal map_patch_artist
        map_patch_artist.set_data(patch_rgb)
        ax_map.set_xlim(0, patch_rgb.shape[1])
        ax_map.set_ylim(patch_rgb.shape[0], 0)

        nonlocal frame_pts_artist, map_pts_artist, pending_artist
        if frame_pts_artist is not None:
            frame_pts_artist.remove()
            frame_pts_artist = None
        if map_pts_artist is not None:
            map_pts_artist.remove()
            map_pts_artist = None
        if pending_artist is not None:
            pending_artist.remove()
            pending_artist = None

        if pairs:
            frame_pts = np.array([fp for fp, _mp in pairs], dtype=float)
            map_pts_global = np.array([mp for _fp, mp in pairs], dtype=float)
            map_pts_view = np.column_stack(
                [
                    (map_pts_global[:, 0] - patch_meta["left"]) * patch_meta["view_w"] / patch_meta["patch_w"],
                    (map_pts_global[:, 1] - patch_meta["top"]) * patch_meta["view_h"] / patch_meta["patch_h"],
                ]
            )
            frame_pts_artist = ax_frame.scatter(frame_pts[:, 0], frame_pts[:, 1], c="cyan", s=38, marker="o")
            map_pts_artist = ax_map.scatter(map_pts_view[:, 0], map_pts_view[:, 1], c="cyan", s=38, marker="o")
        if pending_frame_pt[0] is not None:
            pending_artist = ax_frame.scatter(
                [pending_frame_pt[0][0]], [pending_frame_pt[0][1]], c="yellow", s=60, marker="x", linewidths=2.0
            )
        fig_p2p.canvas.draw_idle()
        return _draw_residuals(rot, scale_mult)

    def _on_change(_v: float) -> None:
        rot = float(s_rot.val)
        scale_mult = float(s_scale.val)
        alpha = float(s_alpha.val)
        if use_homography[0] and homography_h[0] is not None:
            overlay_img.set_visible(False)
            homo_overlay_img.set_visible(True)
            try:
                import cv2
            except Exception:
                use_homography[0] = False
                homo_overlay_img.set_visible(False)
                overlay_img.set_visible(True)
                print("Homography view disabled: OpenCV unavailable.")
            else:
                warped = cv2.warpPerspective(
                    frame_rgba,
                    homography_h[0],
                    (map_w, map_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                warped[:, :, 3] = np.clip(warped[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                homo_overlay_img.set_data(warped)
        else:
            homo_overlay_img.set_visible(False)
            overlay_img.set_visible(True)
            overlay_img.set_alpha(alpha)
            overlay_img.set_transform(_transform_for_display(rot, scale_mult) + ax_main.transData)
        info_text.set_text(
            f"{args.frame_name} | center=({center_x:.1f},{center_y:.1f}) | rot={rot:.2f} deg | "
            f"scale={scale_mult:.3f} | base={base_w_px:.1f}x{base_h_px:.1f}px | "
            f"mode={'H' if use_homography[0] and homography_h[0] is not None else 'AFF'}"
        )
        rmse = _refresh_patch_view()
        if rmse is None:
            last_rmse[0] = None
            loss_text.set_text("Loss RMSE: n/a px\nΔ: n/a\nbest: n/a px")
        else:
            prev = last_rmse[0]
            if best_rmse[0] is None or rmse < best_rmse[0]:
                best_rmse[0] = rmse
            if prev is None:
                delta_str = "init ~ same"
            else:
                delta = rmse - prev
                if delta < -1e-3:
                    delta_str = f"{delta:.2f}px ↓ better"
                elif delta > 1e-3:
                    delta_str = f"+{delta:.2f}px ↑ worse"
                else:
                    delta_str = f"{delta:.2f}px ~ same"
            last_rmse[0] = rmse
            loss_text.set_text(
                f"Loss RMSE: {rmse:.2f} px\n"
                f"Δ: {delta_str}\n"
                f"best: {best_rmse[0]:.2f} px"
            )
        fig_main.canvas.draw_idle()

    def _fit_rot_scale_from_pairs(_event=None) -> None:
        if len(pairs) < 2:
            _status("Fit Rot+Scale: need at least 2 pairs.")
            return

        src_uv = []
        dst_xy = []
        for fp, mp in pairs:
            ux = (float(fp[0]) - frame_cx) * base_sx
            uy = (float(fp[1]) - frame_cy) * base_sy
            vx = float(mp[0]) - center_x
            vy = float(mp[1]) - center_y
            src_uv.append([ux, uy])
            dst_xy.append([vx, vy])
        src_arr = np.array(src_uv, dtype=float)
        dst_arr = np.array(dst_xy, dtype=float)
        try:
            rot_est, scale_est, inliers, rmse_all = _fit_similarity_ransac(
                src_arr,
                dst_arr,
                iterations=max(1, int(args.ransac_iters)),
                thresh_px=max(0.1, float(args.ransac_thresh_px)),
                min_inliers=max(2, min(4, len(pairs))),
                seed=0,
            )
        except ValueError:
            _status("Fit Rot+Scale failed: degenerate geometry.")
            return

        s_rot.set_val(rot_est)
        s_scale.set_val(scale_est)
        _status(
            f"Fit Rot+Scale RANSAC -> rot={rot_est:.2f} deg, scale={scale_est:.4f}, "
            f"inliers={int(np.sum(inliers))}/{len(pairs)}, rmse={rmse_all:.2f}px"
        )
        _print_pair_errors(float(s_rot.val), float(s_scale.val))

    def _fit_homography_from_pairs(_event=None) -> None:
        if len(pairs) < 4:
            _status("Fit Homography: need at least 4 pairs.")
            return
        try:
            import cv2
        except Exception:
            _status("Fit Homography unavailable: OpenCV not installed.")
            return

        src = np.array([fp for fp, _mp in pairs], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([mp for _fp, mp in pairs], dtype=np.float32).reshape(-1, 1, 2)
        h, inliers = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(0.5, float(args.h_ransac_thresh_px)),
        )
        if h is None:
            _status("Fit Homography failed.")
            return

        center_src = np.array([[[frame_cx, frame_cy]]], dtype=np.float32)
        center_proj = cv2.perspectiveTransform(center_src, h)[0, 0]
        dx = center_x - float(center_proj[0])
        dy = center_y - float(center_proj[1])
        t = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        h = t @ h
        homography_h[0] = h
        use_homography[0] = True

        proj = cv2.perspectiveTransform(src, h).reshape(-1, 2)
        tgt = dst.reshape(-1, 2)
        err = np.linalg.norm(proj - tgt, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(err))))
        inl = int(np.sum(inliers)) if inliers is not None else len(pairs)
        _status(f"Fit Homography -> inliers={inl}/{len(pairs)} rmse={rmse:.2f}px")
        _on_change(0.0)

    def _fit_two_stage(_event=None) -> None:
        if len(pairs) < 4:
            _status("Two-Stage: need at least 4 pairs.")
            return
        try:
            import cv2
        except Exception:
            _status("Two-Stage unavailable: OpenCV not installed.")
            return

        # Stage 1: robust similarity.
        _fit_rot_scale_from_pairs()
        rot = float(s_rot.val)
        scale_mult = float(s_scale.val)

        src = np.array([fp for fp, _mp in pairs], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([mp for _fp, mp in pairs], dtype=np.float32).reshape(-1, 1, 2)
        h_raw, inliers = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(0.5, float(args.h_ransac_thresh_px)),
        )
        if h_raw is None:
            _status("Two-Stage: homography stage failed.")
            return

        # Similarity baseline as 3x3 map from frame pixels -> map pixels.
        tform = _transform_for_display(rot, scale_mult)
        m = tform.get_matrix()
        h_base = np.array(m, dtype=np.float64)
        h_base = h_base / (h_base[2, 2] if abs(h_base[2, 2]) > 1e-12 else 1.0)
        h_raw = np.array(h_raw, dtype=np.float64)
        h_raw = h_raw / (h_raw[2, 2] if abs(h_raw[2, 2]) > 1e-12 else 1.0)

        lam = float(np.clip(args.h_refine_lambda, 0.0, 1.0))
        h_mix = (1.0 - lam) * h_base + lam * h_raw
        damp = float(np.clip(args.h_projective_damp, 0.0, 1.0))
        h_mix[2, 0] *= damp
        h_mix[2, 1] *= damp
        h_mix[2, 2] = 1.0

        # Re-anchor center to fixed GPS center.
        center_src = np.array([[[frame_cx, frame_cy]]], dtype=np.float32)
        center_proj = cv2.perspectiveTransform(center_src, h_mix)[0, 0]
        dx = center_x - float(center_proj[0])
        dy = center_y - float(center_proj[1])
        t = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        h_final = t @ h_mix

        homography_h[0] = h_final
        use_homography[0] = True

        proj = cv2.perspectiveTransform(src, h_final).reshape(-1, 2)
        tgt = dst.reshape(-1, 2)
        err = np.linalg.norm(proj - tgt, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(err))))
        inl = int(np.sum(inliers)) if inliers is not None else len(pairs)
        _status(
            f"Two-Stage -> inliers={inl}/{len(pairs)} rmse={rmse:.2f}px "
            f"(lambda={lam:.2f}, proj_damp={damp:.2f})"
        )
        _on_change(0.0)

    def _clear_pairs(_event=None) -> None:
        pairs.clear()
        pending_frame_pt[0] = None
        homography_h[0] = None
        use_homography[0] = False
        _save_pairs_json(args.points_json, args.frame_name, center_x, center_y, pairs)
        _on_change(0.0)
        _status("Pairs cleared.")

    def _suggest_orb_pairs(_event=None) -> None:
        try:
            import cv2
        except Exception:
            _status("ORB unavailable: install opencv-python or opencv-python-headless.")
            return
        frame_orb = _resize_max_side(frame_rgb_raw, max(256, int(args.orb_max_side)))
        orb_h, orb_w = frame_orb.shape[:2]

        scale_mult = float(s_scale.val)
        patch_w = max(8, int(round(base_w_px * scale_mult * 1.5)))
        patch_h = max(8, int(round(base_h_px * scale_mult * 1.5)))
        patch_rgb, meta_orb = _extract_patch(
            map_rgb,
            center_x,
            center_y,
            patch_w,
            patch_h,
            out_size=(orb_w, orb_h),
        )

        frame_gray = cv2.cvtColor(frame_orb, cv2.COLOR_RGB2GRAY)
        patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=5000)
        kf, df = orb.detectAndCompute(frame_gray, None)
        km, dm = orb.detectAndCompute(patch_gray, None)
        if df is None or dm is None or len(kf) < 8 or len(km) < 8:
            _status("ORB: not enough keypoints.")
            return
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(df, dm, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 4:
            _status(f"ORB: too few matches ({len(good)}).")
            return
        good = sorted(good, key=lambda m: float(m.distance))[: max(4, int(args.p2p_orb_max_pairs))]
        pairs.clear()
        pending_frame_pt[0] = None
        for m in good:
            fx, fy = kf[m.queryIdx].pt
            mx_view, my_view = km[m.trainIdx].pt
            fx_disp = float(fx) * float(frame_w) / float(orb_w)
            fy_disp = float(fy) * float(frame_h) / float(orb_h)
            mx, my = _patch_view_to_global(meta_orb, float(mx_view), float(my_view))
            pairs.append(((fx_disp, fy_disp), (mx, my)))
        _save_pairs_json(args.points_json, args.frame_name, center_x, center_y, pairs)
        _on_change(0.0)
        _status(f"ORB suggested {len(pairs)} pairs (orb_res={orb_w}x{orb_h}).")
        _fit_rot_scale_from_pairs()

    def _save_result(_event=None) -> None:
        rot = float(s_rot.val)
        scale_mult = float(s_scale.val)
        alpha = float(s_alpha.val)
        errs = []
        for fp, mp in pairs:
            ex, ey = _predict_map_point(fp, rot, scale_mult)
            errs.append(float(math.hypot(mp[0] - ex, mp[1] - ey)))
        rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
        payload = {
            "frame_name": args.frame_name,
            "tiff": str(args.tiff),
            "center_map_px": {"x": center_x, "y": center_y},
            "gps": {"lat": lat, "lon": lon, "alt": alt},
            "base_size_px": {"width": base_w_px, "height": base_h_px},
            "rotation_deg": rot,
            "scale_mult": scale_mult,
            "opacity": alpha,
            "pairs_count": len(pairs),
            "pairs": [
                {"frame_xy": [fp[0], fp[1]], "map_xy": [mp[0], mp[1]]}
                for fp, mp in pairs
            ],
            "rmse_px": rmse,
            "mode": "homography" if use_homography[0] and homography_h[0] is not None else "affine",
            "homography_3x3": homography_h[0].tolist() if homography_h[0] is not None else None,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _status(f"Saved: {args.output_json}")
        if rmse is not None:
            print(f"Final RMSE: {rmse:.2f}px")

    def _on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes == ax_frame:
            pending_frame_pt[0] = (float(event.xdata), float(event.ydata))
            _refresh_patch_view()
            _status(f"Frame point selected: ({event.xdata:.1f}, {event.ydata:.1f})")
            return
        if event.inaxes != ax_map:
            return
        if pending_frame_pt[0] is None:
            _status("Click frame point first.")
            return
        mx, my = _patch_view_to_global(patch_meta, float(event.xdata), float(event.ydata))
        pairs.append((pending_frame_pt[0], (mx, my)))
        pending_frame_pt[0] = None
        _save_pairs_json(args.points_json, args.frame_name, center_x, center_y, pairs)
        _refresh_patch_view()
        _fit_rot_scale_from_pairs()

    def _on_key(event) -> None:
        if event.key == "left":
            s_rot.set_val(s_rot.val - 1.0)
        elif event.key == "right":
            s_rot.set_val(s_rot.val + 1.0)
        elif event.key == "[":
            s_scale.set_val(max(s_scale.valmin, s_scale.val - 0.01))
        elif event.key == "]":
            s_scale.set_val(min(s_scale.valmax, s_scale.val + 0.01))
        elif event.key and event.key.lower() == "r":
            _fit_rot_scale_from_pairs()
        elif event.key and event.key.lower() == "h":
            _fit_homography_from_pairs()
        elif event.key and event.key.lower() == "t":
            _fit_two_stage()
        elif event.key and event.key.lower() == "o":
            _suggest_orb_pairs()
        elif event.key and event.key.lower() == "x":
            _clear_pairs()
        elif event.key and event.key.lower() == "s":
            _save_result()
        elif event.key and event.key.lower() == "e":
            _print_pair_errors(float(s_rot.val), float(s_scale.val))
            _status("Printed pair residuals to terminal.")

    s_rot.on_changed(_on_change)
    s_scale.on_changed(_on_change)
    s_alpha.on_changed(_on_change)
    b_orb.on_clicked(_suggest_orb_pairs)
    b_h.on_clicked(_fit_homography_from_pairs)
    b_two.on_clicked(_fit_two_stage)
    b_err.on_clicked(lambda _e: (_print_pair_errors(float(s_rot.val), float(s_scale.val)), _status("Printed pair residuals to terminal.")))
    b_fit.on_clicked(_fit_rot_scale_from_pairs)
    b_clear.on_clicked(_clear_pairs)
    b_save.on_clicked(_save_result)
    fig_p2p.canvas.mpl_connect("button_press_event", _on_click)
    fig_main.canvas.mpl_connect("key_press_event", _on_key)
    fig_p2p.canvas.mpl_connect("key_press_event", _on_key)

    _on_change(0.0)
    print("Interactive controls:")
    print(" - Rotate: slider or Left/Right")
    print(" - Scale: slider or [ ]")
    print(" - Opacity: slider")
    print(" - P2P add: click FRAME then matching MAP point in Figure 2")
    print(" - ORB suggestions: button or 'o'")
    print(" - Fit rotation+scale from pairs: button or 'r'")
    print(" - Fit center-anchored homography: button or 'h' (needs >=4 pairs)")
    print(" - Two-stage (RANSAC similarity -> regularized homography): button or 't'")
    print(" - Print pair residuals: button or 'e'")
    print(" - Clear pairs: button or 'x'")
    print(" - Save result JSON: button or 's'")
    print(
        f"Fixed center (GPS anchor): ({center_x:.2f}, {center_y:.2f}) | "
        f"base_size_px={base_w_px:.2f}x{base_h_px:.2f}"
    )
    print(
        f"Resolutions -> frame_raw={raw_w}x{raw_h}, frame_display={frame_w}x{frame_h}, "
        f"map_tiff={map_w}x{map_h}, orb_max_side={args.orb_max_side}"
    )
    print(
        f"RANSAC settings -> iters={args.ransac_iters}, thresh_px={args.ransac_thresh_px:.1f}"
    )
    print(
        f"Homography refine -> h_thresh={args.h_ransac_thresh_px:.1f}, "
        f"lambda={args.h_refine_lambda:.2f}, proj_damp={args.h_projective_damp:.2f}"
    )
    print(f"Points JSON: {args.points_json} (loaded {len(pairs)} pairs)")
    plt.show()


if __name__ == "__main__":
    main()
