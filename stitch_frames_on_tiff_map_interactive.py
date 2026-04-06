#!/usr/bin/env python3
"""Interactive manual alignment for stitched trajectory overlays on TIFF map."""

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

from stitch_frames_on_tiff_map import (
    _compute_heading_deg,
    _estimate_rigid_kabsch,
    _ransac_rotation_deg,
    _read_json,
    _resolve_homography_path,
    _smooth_heading_deg,
    _sorted_items,
)
from vis_first_frame_on_tiff import _load_geotiff, _project_to_tiff_pixel, _require_positive_size, _resize_with_pillow


def _load_optional_manual_adjustment(path: Path | None) -> tuple[float, float, float, float, float, set[str]]:
    if path is None:
        return 0.0, 0.0, 0.0, 1.0, 0.55, set()
    payload = _read_json(path)
    manual = payload.get("manual_adjustment", {})
    dx = float(manual.get("dx_px", 0.0))
    dy = float(manual.get("dy_px", 0.0))
    rot = float(manual.get("rotation_deg", 0.0))
    scale = float(manual.get("scale", 1.0))
    opacity = float(manual.get("opacity", 0.55))
    visibility = payload.get("visibility", {})
    hidden_images = visibility.get("hidden_images", [])
    if not isinstance(hidden_images, list):
        hidden_images = []
    hidden_image_set = {str(x) for x in hidden_images}
    return dx, dy, rot, scale, opacity, hidden_image_set


def _write_manual_adjustment_json(
    out_path: Path,
    sequence: str,
    map_size: tuple[int, int],
    frame_size: tuple[int, int],
    sample_step: int,
    rotate_by_heading: bool,
    dx: float,
    dy: float,
    rot: float,
    scale: float,
    opacity: float,
    hidden_indices: list[int],
    hidden_images: list[str],
) -> None:
    payload = {
        "sequence": sequence,
        "map_size": {"width": map_size[0], "height": map_size[1]},
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "sample_step": sample_step,
        "rotate_by_heading": rotate_by_heading,
        "manual_adjustment": {
            "dx_px": dx,
            "dy_px": dy,
            "rotation_deg": rot,
            "scale": scale,
            "opacity": opacity,
        },
        "visibility": {
            "hidden_indices": hidden_indices,
            "hidden_images": hidden_images,
            "num_hidden": len(hidden_indices),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _rgba_without_black_background(frame_rgb: np.ndarray, black_thr: int) -> np.ndarray:
    frame = frame_rgb.astype(np.uint8, copy=False)
    alpha = np.where(np.max(frame, axis=2) <= int(black_thr), 0, 255).astype(np.uint8)
    return np.dstack((frame, alpha))


def _estimate_rotation_orb(frame_rgb: np.ndarray, map_rgb: np.ndarray) -> tuple[float | None, str]:
    try:
        import cv2
    except Exception:
        return None, "ORB unavailable (opencv-python not installed)"

    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    map_gray = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    k0, d0 = orb.detectAndCompute(frame_gray, None)
    k1, d1 = orb.detectAndCompute(map_gray, None)
    if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
        return None, "ORB: not enough keypoints"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(d0, d1, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 10:
        return None, f"ORB: too few matches ({len(good)})"

    src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine is None:
        return None, "ORB: affine estimation failed"

    angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    return _wrap_angle_deg(angle), f"ORB inliers={inlier_count}/{len(good)}"


def _estimate_rotation_loftr(
    frame_rgb: np.ndarray,
    map_rgb: np.ndarray,
    cache: dict[str, Any],
    max_side: int = 640,
) -> tuple[float | None, str]:
    try:
        import torch
        import torch.nn.functional as f
        from kornia.feature import LoFTR
    except Exception:
        return None, "LoFTR unavailable (kornia/torch missing)"

    def _to_gray_tensor(img_rgb: np.ndarray, device: Any) -> tuple[Any, float, float]:
        img = img_rgb.astype(np.float32) / 255.0
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        ten = torch.from_numpy(gray).to(device=device, dtype=torch.float32)[None, None]
        h, w = gray.shape
        sx = 1.0
        sy = 1.0
        m = max(h, w)
        if m > max_side:
            ratio = float(max_side) / float(m)
            nh = max(16, int(round(h * ratio)))
            nw = max(16, int(round(w * ratio)))
            ten = f.interpolate(ten, size=(nh, nw), mode="bilinear", align_corners=False)
            sx = float(w) / float(nw)
            sy = float(h) / float(nh)
        return ten, sx, sy

    if "loftr_matcher" not in cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache["loftr_device"] = device
        cache["loftr_matcher"] = LoFTR(pretrained="outdoor").to(device).eval()
    device = cache["loftr_device"]
    matcher = cache["loftr_matcher"]

    t0, sx0, sy0 = _to_gray_tensor(frame_rgb, device)
    t1, sx1, sy1 = _to_gray_tensor(map_rgb, device)
    with torch.no_grad():
        out = matcher({"image0": t0, "image1": t1})
    k0 = out.get("keypoints0", None)
    k1 = out.get("keypoints1", None)
    if k0 is None or k1 is None:
        return None, "LoFTR: no keypoints"
    if int(k0.shape[0]) < 8 or int(k1.shape[0]) < 8:
        return None, f"LoFTR: too few matches ({int(k0.shape[0])})"

    pts0 = k0.detach().cpu().numpy().astype(np.float64)
    pts1 = k1.detach().cpu().numpy().astype(np.float64)
    pts0[:, 0] *= sx0
    pts0[:, 1] *= sy0
    pts1[:, 0] *= sx1
    pts1[:, 1] *= sy1
    angle, inliers = _ransac_rotation_deg(
        pts0,
        pts1,
        iterations=700,
        reproj_thresh_px=4.0,
        min_inliers=12,
        seed=0,
    )
    if angle is None:
        return None, f"LoFTR: RANSAC failed (best_inliers={inliers})"
    return _wrap_angle_deg(float(angle)), f"LoFTR inliers={inliers}/{len(pts0)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual alignment for stitched trajectory on TIFF map.")
    parser.add_argument("--tiff", type=Path, default=Path("test_references/0001_year_2024_crop.tiff"))
    parser.add_argument("--metadata", type=Path, default=Path("test_images/0001/gps_metadata.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("test_images/0001"))
    parser.add_argument("--map-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--frame-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(256, 256))
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0)
    parser.add_argument("--camera-vfov-deg", type=float, default=None)
    parser.add_argument("--altitude-offset-m", type=float, default=0.0)
    parser.add_argument("--black-transparent-threshold", type=int, default=8)
    parser.add_argument("--p2p-orb-max-pairs", type=int, default=24)
    parser.add_argument("--rotation-matcher", type=str, default="auto", choices=["auto", "loftr", "orb", "off"])
    parser.add_argument("--sample-step", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--rotate-by-heading", action="store_true", default=False)
    parser.add_argument(
        "--heading-smooth-window",
        type=int,
        default=1,
        help="Odd window size for heading smoothing (1 disables smoothing).",
    )
    parser.add_argument("--homography-json", type=Path, default=Path("test_frame_overlays/0001_first_frame_on_tiff_interactive_homography.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_frame_overlays/0001_stitched_manual_interactive.png"),
    )
    parser.add_argument(
        "--manual-output-json",
        type=Path,
        default=Path("test_frame_overlays/0001_stitched_manual_adjustment.json"),
    )
    args = parser.parse_args()

    if not args.tiff.exists():
        raise FileNotFoundError(f"TIFF file does not exist: {args.tiff}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {args.metadata}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if args.sample_step < 1:
        raise ValueError(f"--sample-step must be >= 1, got {args.sample_step}")
    if args.max_frames < 1:
        raise ValueError(f"--max-frames must be >= 1, got {args.max_frames}")
    if args.heading_smooth_window < 1:
        raise ValueError(f"--heading-smooth-window must be >= 1, got {args.heading_smooth_window}")

    frame_size = _require_positive_size("--frame-size", tuple(args.frame_size))

    homography_path = _resolve_homography_path(args.homography_json)
    state_path = args.manual_output_json if args.manual_output_json.exists() else homography_path
    init_dx, init_dy, init_rot, init_scale, init_opacity, _init_hidden_images = _load_optional_manual_adjustment(
        state_path
    )

    metadata = _read_json(args.metadata)
    sequence = str(metadata.get("sequence", "unknown"))
    items = _sorted_items(metadata)

    map_arr_orig, map_w_orig, map_h_orig, scale_x, scale_y, tie_i, tie_j, tie_x, tie_y, tiff_crs = _load_geotiff(args.tiff)

    px_orig_list: list[float] = []
    py_orig_list: list[float] = []
    valid_items: list[dict] = []
    for item in items:
        lon = float(item["longitude"])
        lat = float(item["latitude"])
        px, py = _project_to_tiff_pixel(
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
        if 0.0 <= px < map_w_orig and 0.0 <= py < map_h_orig:
            px_orig_list.append(px)
            py_orig_list.append(py)
            valid_items.append(item)

    if not valid_items:
        raise RuntimeError("No GPS points project inside TIFF bounds")

    px_orig = np.array(px_orig_list, dtype=float)
    py_orig = np.array(py_orig_list, dtype=float)
    resized_map = map_arr_orig
    map_size = (map_w_orig, map_h_orig)
    px = px_orig
    py = py_orig

    sampled_idx = np.arange(0, len(valid_items), args.sample_step, dtype=int)
    if len(sampled_idx) > args.max_frames:
        sampled_idx = sampled_idx[: args.max_frames]

    heading_deg = _compute_heading_deg(px, py)
    heading_deg = _smooth_heading_deg(heading_deg, args.heading_smooth_window)

    sampled_frames: list[np.ndarray] = []
    sampled_frames_rgba: list[np.ndarray] = []
    sampled_images: list[str] = []
    sampled_cx: list[float] = []
    sampled_cy: list[float] = []
    sampled_heading: list[float] = []
    sampled_alt_m: list[float] = []
    for idx in sampled_idx:
        item = valid_items[int(idx)]
        img_path = args.images_dir / str(item["image"])
        if not img_path.exists():
            continue
        frame_arr = np.array(Image.open(img_path).convert("RGB"))
        frame_resized = _resize_with_pillow(frame_arr, frame_size)
        sampled_frames.append(frame_resized)
        sampled_frames_rgba.append(_rgba_without_black_background(frame_resized, args.black_transparent_threshold))
        sampled_images.append(str(item["image"]))
        sampled_cx.append(float(px[idx]))
        sampled_cy.append(float(py[idx]))
        sampled_heading.append(float(heading_deg[idx]))
        sampled_alt_m.append(float(item.get("altitude", 0.0)))

    if not sampled_frames:
        raise RuntimeError("No sampled image files found for stitching")

    map_h, map_w = resized_map.shape[:2]
    frame_w, frame_h = frame_size
    frame_aspect = frame_h / float(frame_w)

    hfov_deg = float(args.camera_hfov_deg)
    vfov_deg = float(args.camera_vfov_deg) if args.camera_vfov_deg is not None else None
    if hfov_deg <= 0.0 or hfov_deg >= 179.0:
        raise ValueError(f"--camera-hfov-deg must be in (0, 179), got {hfov_deg}")
    if vfov_deg is None:
        hfov_rad = math.radians(hfov_deg)
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * frame_aspect)
        vfov_deg = math.degrees(vfov_rad)
    if vfov_deg <= 0.0 or vfov_deg >= 179.0:
        raise ValueError(f"--camera-vfov-deg must be in (0, 179), got {vfov_deg}")

    map_mpp_x = abs(float(scale_x))
    map_mpp_y = abs(float(scale_y))
    if map_mpp_x <= 0.0 or map_mpp_y <= 0.0:
        raise ValueError(f"Invalid GeoTIFF meters-per-pixel: scale_x={scale_x}, scale_y={scale_y}")

    hfov_rad = math.radians(hfov_deg)
    vfov_rad = math.radians(vfov_deg)
    sampled_base_w_px: list[float] = []
    sampled_base_h_px: list[float] = []
    for alt_m in sampled_alt_m:
        effective_alt_m = max(0.1, alt_m + float(args.altitude_offset_m))
        ground_w_m = 2.0 * effective_alt_m * math.tan(hfov_rad / 2.0)
        ground_h_m = 2.0 * effective_alt_m * math.tan(vfov_rad / 2.0)
        sampled_base_w_px.append(max(1.0, ground_w_m / map_mpp_x))
        sampled_base_h_px.append(max(1.0, ground_h_m / map_mpp_y))

    fig = plt.figure(figsize=(map_w / 100.0, map_h / 100.0), dpi=100, num="Overlay (Figure 1)")
    ax = fig.add_axes([0.0, 0.20, 1.0, 0.80])
    ax.imshow(resized_map)
    ax.plot(px, py, color="yellow", linewidth=1.8, alpha=0.8, zorder=2)
    ax.scatter([px[0]], [py[0]], s=60, c="lime", edgecolors="black", linewidths=0.8, zorder=3)
    ax.scatter([px[-1]], [py[-1]], s=60, c="red", edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.set_axis_off()

    info_text = ax.text(
        10,
        20,
        "",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
        zorder=20,
    )

    fig_p2p, (ax_p2p_frame, ax_p2p_map) = plt.subplots(1, 2, figsize=(12, 6), num="Point-to-Point Tool (Figure 2)")
    ax_p2p_frame.set_title("Frame (click point)")
    ax_p2p_frame.set_xlabel("Frame X")
    ax_p2p_frame.set_ylabel("Frame Y")
    ax_p2p_map.set_title("Map patch (click corresponding point)")
    ax_p2p_map.set_xlabel("Patch X")
    ax_p2p_map.set_ylabel("Patch Y")

    overlay_artist = None
    center_marker = None
    p2p_overlay_map_pts = None
    p2p_overlay_est_pts = None
    p2p_overlay_lines = []
    matcher_cache: dict[str, Any] = {}
    p2p_mode = [False]
    p2p_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    p2p_pending_frame_pt: list[tuple[float, float] | None] = [None]
    p2p_patch_meta: dict[str, float] = {
        "left": 0.0,
        "top": 0.0,
        "patch_w": float(frame_w),
        "patch_h": float(frame_h),
        "view_w": float(frame_w),
        "view_h": float(frame_h),
    }
    p2p_frame_artist = None
    p2p_map_artist = None
    p2p_frame_pts = None
    p2p_map_pts = None
    p2p_pending_frame_pt_artist = None

    def _draw_overlays(frame_idx: int, dx: float, dy: float, rot: float, scale: float, opacity: float) -> None:
        nonlocal overlay_artist, center_marker
        if overlay_artist is not None:
            overlay_artist.remove()
            overlay_artist = None
        if center_marker is not None:
            center_marker.remove()
            center_marker = None

        i = int(np.clip(frame_idx, 0, len(sampled_frames) - 1))
        frame_arr = sampled_frames_rgba[i]
        cx0 = sampled_cx[i]
        cy0 = sampled_cy[i]
        head = sampled_heading[i]
        angle = rot + (head if args.rotate_by_heading else 0.0)
        base_w_px = sampled_base_w_px[i]
        base_h_px = sampled_base_h_px[i]
        eff_w_px = max(1.0, base_w_px * scale)
        eff_h_px = max(1.0, base_h_px * scale)

        overlay_artist = ax.imshow(frame_arr, extent=(0, eff_w_px, eff_h_px, 0), alpha=opacity, zorder=4)
        tr = (
            Affine2D()
            .translate(-eff_w_px / 2.0, -eff_h_px / 2.0)
            .rotate_deg(angle)
            .translate(cx0 + dx, cy0 + dy)
            + ax.transData
        )
        overlay_artist.set_transform(tr)
        center_marker = ax.scatter(
            [cx0 + dx],
            [cy0 + dy],
            s=35,
            c="cyan",
            marker="x",
            linewidths=1.4,
            zorder=7,
        )

        info_text.set_text(
            f"seq={sequence} | frame={i + 1}/{len(sampled_frames)} ({sampled_images[i]}) "
            f"| base={base_w_px:.1f}x{base_h_px:.1f}px eff={eff_w_px:.1f}x{eff_h_px:.1f}px "
            f"| scale={scale:.3f} dx={dx:.1f} dy={dy:.1f} rot={rot:.1f} op={opacity:.2f}"
        )
        _draw_p2p_overlay_diagnostics()
        fig.canvas.draw_idle()

    ax_idx = fig.add_axes([0.12, 0.17, 0.62, 0.03])
    s_idx = Slider(ax_idx, "Frame", 0, len(sampled_frames) - 1, valinit=0, valstep=1)

    ax_scale = fig.add_axes([0.12, 0.13, 0.62, 0.03])
    s_scale = Slider(ax_scale, "Scale", 0.10, 5.0, valinit=init_scale, valstep=0.01)

    ax_rot = fig.add_axes([0.12, 0.09, 0.62, 0.03])
    s_rot = Slider(ax_rot, "Rotation", -180.0, 180.0, valinit=init_rot, valstep=0.1)

    offset_limit = float(max(map_w, map_h))
    ax_dx = fig.add_axes([0.12, 0.05, 0.62, 0.03])
    s_dx = Slider(ax_dx, "Offset X", -offset_limit, offset_limit, valinit=init_dx, valstep=1.0)

    ax_dy = fig.add_axes([0.12, 0.01, 0.62, 0.03])
    s_dy = Slider(ax_dy, "Offset Y", -offset_limit, offset_limit, valinit=init_dy, valstep=1.0)

    ax_op = fig.add_axes([0.76, 0.21, 0.20, 0.03])
    s_op = Slider(ax_op, "Opacity", 0.0, 1.0, valinit=init_opacity, valstep=0.01)

    ax_save = fig.add_axes([0.78, 0.09, 0.18, 0.06])
    b_save = Button(ax_save, "Save PNG+JSON")
    ax_match = fig.add_axes([0.78, 0.01, 0.18, 0.06])
    b_match = Button(ax_match, "Auto Rot (M)")
    ax_p2p_btn = fig.add_axes([0.78, 0.13, 0.18, 0.04])
    b_p2p = Button(ax_p2p_btn, "P2P Rot (P)")
    ax_p2p_clear = fig.add_axes([0.78, 0.18, 0.18, 0.04])
    b_p2p_clear = Button(ax_p2p_clear, "Clear P2P (X)")
    ax_p2p_orb = fig.add_axes([0.78, 0.23, 0.18, 0.04])
    b_p2p_orb = Button(ax_p2p_orb, "P2P ORB (O)")

    def _on_change(_v: float) -> None:
        _draw_overlays(
            int(s_idx.val),
            float(s_dx.val),
            float(s_dy.val),
            float(s_rot.val),
            float(s_scale.val),
            float(s_op.val),
        )
        _refresh_p2p_tool_view()

    def _save(_event=None) -> None:
        dx = float(s_dx.val)
        dy = float(s_dy.val)
        rot = float(s_rot.val)
        scale = float(s_scale.val)
        opacity = float(s_op.val)
        _draw_overlays(int(s_idx.val), dx, dy, rot, scale, opacity)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=100, bbox_inches=None, pad_inches=0)
        _write_manual_adjustment_json(
            out_path=args.manual_output_json,
            sequence=sequence,
            map_size=map_size,
            frame_size=frame_size,
            sample_step=args.sample_step,
            rotate_by_heading=args.rotate_by_heading,
            dx=dx,
            dy=dy,
            rot=rot,
            scale=scale,
            opacity=opacity,
            hidden_indices=[],
            hidden_images=[],
        )
        print(f"Saved PNG: {args.output}")
        print(f"Saved manual JSON: {args.manual_output_json}")

    def _extract_map_patch_rgb(cx: float, cy: float, w: float, h: float, margin: float = 1.35) -> np.ndarray:
        patch_w = max(8, int(round(w * margin)))
        patch_h = max(8, int(round(h * margin)))
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
        patch = resized_map[t0:b0, l0:r0]
        if pad_l or pad_t or pad_r or pad_b:
            patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        return _resize_with_pillow(patch, frame_size)

    def _auto_match_rotation(_event=None) -> None:
        if args.rotation_matcher == "off":
            print("Auto-rotation matcher is disabled (--rotation-matcher off).")
            return

        i = int(np.clip(int(s_idx.val), 0, len(sampled_frames) - 1))
        frame_arr = sampled_frames[i]
        cx = sampled_cx[i] + float(s_dx.val)
        cy = sampled_cy[i] + float(s_dy.val)
        base_w_px = sampled_base_w_px[i] * float(s_scale.val)
        base_h_px = sampled_base_h_px[i] * float(s_scale.val)
        map_patch = _extract_map_patch_rgb(cx, cy, base_w_px, base_h_px)

        requested = args.rotation_matcher
        est_angle: float | None = None
        details = ""
        used = requested
        if requested in {"auto", "loftr"}:
            est_angle, details = _estimate_rotation_loftr(frame_arr, map_patch, matcher_cache)
            used = "loftr"
        if est_angle is None and requested in {"auto", "orb"}:
            est_angle, details = _estimate_rotation_orb(frame_arr, map_patch)
            used = "orb"
        if est_angle is None:
            print(f"Auto-rotation failed ({requested}): {details}")
            return

        head = sampled_heading[i] if args.rotate_by_heading else 0.0
        target_rot = _wrap_angle_deg(float(est_angle) - float(head))
        s_rot.set_val(target_rot)
        print(
            f"Auto-rotation [{used}] frame={i + 1}/{len(sampled_frames)} -> "
            f"rot={target_rot:.2f} deg (est={float(est_angle):.2f}, head={head:.2f}) | {details}"
        )

    def _clear_p2p_overlay_diagnostics() -> None:
        nonlocal p2p_overlay_map_pts, p2p_overlay_est_pts, p2p_overlay_lines
        if p2p_overlay_map_pts is not None:
            p2p_overlay_map_pts.remove()
            p2p_overlay_map_pts = None
        if p2p_overlay_est_pts is not None:
            p2p_overlay_est_pts.remove()
            p2p_overlay_est_pts = None
        for art in p2p_overlay_lines:
            art.remove()
        p2p_overlay_lines = []

    def _draw_p2p_overlay_diagnostics() -> None:
        _clear_p2p_overlay_diagnostics()
        if not p2p_pairs:
            fig.canvas.draw_idle()
            return
        i = int(np.clip(int(s_idx.val), 0, len(sampled_frames) - 1))
        head = sampled_heading[i] if args.rotate_by_heading else 0.0
        angle = math.radians(float(s_rot.val) + float(head))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cx = sampled_cx[i] + float(s_dx.val)
        cy = sampled_cy[i] + float(s_dy.val)
        eff_w = max(1.0, sampled_base_w_px[i] * float(s_scale.val))
        eff_h = max(1.0, sampled_base_h_px[i] * float(s_scale.val))

        map_pts = np.array([mp for _fp, mp in p2p_pairs], dtype=float)
        est_pts = []
        for fp, _mp in p2p_pairs:
            fx, fy = fp
            lx = (fx / float(frame_w) - 0.5) * eff_w
            ly = (fy / float(frame_h) - 0.5) * eff_h
            ex = cx + (cos_a * lx - sin_a * ly)
            ey = cy + (sin_a * lx + cos_a * ly)
            est_pts.append((ex, ey))
        est_pts_arr = np.array(est_pts, dtype=float)

        p2p_overlay_map_pts = ax.scatter(map_pts[:, 0], map_pts[:, 1], c="magenta", s=32, marker="o", zorder=14)
        p2p_overlay_est_pts = ax.scatter(est_pts_arr[:, 0], est_pts_arr[:, 1], c="lime", s=28, marker="+", zorder=14)
        for (mx, my), (ex, ey) in zip(map_pts, est_pts_arr, strict=True):
            p2p_overlay_lines.append(ax.plot([mx, ex], [my, ey], color="white", linewidth=1.0, alpha=0.9, zorder=13)[0])
        fig.canvas.draw_idle()

    def _clear_p2p_tool_artists() -> None:
        nonlocal p2p_frame_artist, p2p_map_artist, p2p_frame_pts, p2p_map_pts, p2p_pending_frame_pt_artist
        if p2p_frame_artist is not None:
            p2p_frame_artist.remove()
            p2p_frame_artist = None
        if p2p_map_artist is not None:
            p2p_map_artist.remove()
            p2p_map_artist = None
        if p2p_frame_pts is not None:
            p2p_frame_pts.remove()
            p2p_frame_pts = None
        if p2p_map_pts is not None:
            p2p_map_pts.remove()
            p2p_map_pts = None
        if p2p_pending_frame_pt_artist is not None:
            p2p_pending_frame_pt_artist.remove()
            p2p_pending_frame_pt_artist = None

    def _extract_map_patch_for_p2p(cx: float, cy: float, w: float, h: float, margin: float = 1.35) -> np.ndarray:
        patch_w = max(8, int(round(w * margin)))
        patch_h = max(8, int(round(h * margin)))
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
        patch = resized_map[t0:b0, l0:r0]
        if pad_l or pad_t or pad_r or pad_b:
            patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        resized = _resize_with_pillow(patch, frame_size)
        p2p_patch_meta["left"] = float(left)
        p2p_patch_meta["top"] = float(top)
        p2p_patch_meta["patch_w"] = float(patch_w)
        p2p_patch_meta["patch_h"] = float(patch_h)
        p2p_patch_meta["view_w"] = float(resized.shape[1])
        p2p_patch_meta["view_h"] = float(resized.shape[0])
        return resized

    def _map_patch_view_to_global(px_view: float, py_view: float) -> tuple[float, float]:
        sx = p2p_patch_meta["patch_w"] / max(1.0, p2p_patch_meta["view_w"])
        sy = p2p_patch_meta["patch_h"] / max(1.0, p2p_patch_meta["view_h"])
        gx = p2p_patch_meta["left"] + float(px_view) * sx
        gy = p2p_patch_meta["top"] + float(py_view) * sy
        return gx, gy

    def _refresh_p2p_tool_view() -> None:
        _clear_p2p_tool_artists()
        i = int(np.clip(int(s_idx.val), 0, len(sampled_frames) - 1))
        frame_img = sampled_frames[i]
        cx = sampled_cx[i] + float(s_dx.val)
        cy = sampled_cy[i] + float(s_dy.val)
        eff_w = max(1.0, sampled_base_w_px[i] * float(s_scale.val))
        eff_h = max(1.0, sampled_base_h_px[i] * float(s_scale.val))
        map_patch = _extract_map_patch_for_p2p(cx, cy, eff_w, eff_h)

        nonlocal p2p_frame_artist, p2p_map_artist, p2p_frame_pts, p2p_map_pts, p2p_pending_frame_pt_artist
        p2p_frame_artist = ax_p2p_frame.imshow(frame_img)
        p2p_map_artist = ax_p2p_map.imshow(map_patch)
        ax_p2p_frame.set_xlim(0, frame_img.shape[1])
        ax_p2p_frame.set_ylim(frame_img.shape[0], 0)
        ax_p2p_map.set_xlim(0, map_patch.shape[1])
        ax_p2p_map.set_ylim(map_patch.shape[0], 0)
        if p2p_pairs:
            frame_pts = np.array([fp for fp, _mp in p2p_pairs], dtype=float)
            map_pts_global = np.array([mp for _fp, mp in p2p_pairs], dtype=float)
            map_pts_view = np.column_stack(
                [
                    (map_pts_global[:, 0] - p2p_patch_meta["left"]) * max(1.0, p2p_patch_meta["view_w"]) / max(1.0, p2p_patch_meta["patch_w"]),
                    (map_pts_global[:, 1] - p2p_patch_meta["top"]) * max(1.0, p2p_patch_meta["view_h"]) / max(1.0, p2p_patch_meta["patch_h"]),
                ]
            )
            p2p_frame_pts = ax_p2p_frame.scatter(frame_pts[:, 0], frame_pts[:, 1], c="cyan", s=38, marker="o")
            p2p_map_pts = ax_p2p_map.scatter(map_pts_view[:, 0], map_pts_view[:, 1], c="cyan", s=38, marker="o")
        if p2p_pending_frame_pt[0] is not None:
            pfx, pfy = p2p_pending_frame_pt[0]
            p2p_pending_frame_pt_artist = ax_p2p_frame.scatter([pfx], [pfy], c="yellow", s=56, marker="x")
        fig_p2p.canvas.draw_idle()

    def _apply_p2p_solution() -> None:
        if len(p2p_pairs) < 2:
            print("P2P: need at least 2 frame-map pairs.")
            return
        i = int(np.clip(int(s_idx.val), 0, len(sampled_frames) - 1))
        head = sampled_heading[i] if args.rotate_by_heading else 0.0
        eff_w = max(1.0, sampled_base_w_px[i] * float(s_scale.val))
        eff_h = max(1.0, sampled_base_h_px[i] * float(s_scale.val))

        src_local = []
        dst_map = []
        for (fx, fy), (mx, my) in p2p_pairs:
            src_local.append([(fx / float(frame_w) - 0.5) * eff_w, (fy / float(frame_h) - 0.5) * eff_h])
            dst_map.append([mx, my])
        src_arr = np.array(src_local, dtype=float)
        dst_arr = np.array(dst_map, dtype=float)
        r_mat, t_vec = _estimate_rigid_kabsch(src_arr, dst_arr)
        total_angle = math.degrees(math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0])))
        target_rot = _wrap_angle_deg(total_angle - float(head))
        target_dx = float(t_vec[0]) - float(sampled_cx[i])
        target_dy = float(t_vec[1]) - float(sampled_cy[i])
        s_rot.set_val(target_rot)
        s_dx.set_val(target_dx)
        s_dy.set_val(target_dy)
        _draw_p2p_overlay_diagnostics()
        print(
            f"P2P applied: pairs={len(p2p_pairs)} rot={target_rot:.2f} dx={target_dx:.2f} dy={target_dy:.2f}"
        )

    def _suggest_p2p_pairs_orb(_event=None) -> None:
        try:
            import cv2
        except Exception:
            print("P2P ORB unavailable: install opencv-python or opencv-python-headless.")
            return

        i = int(np.clip(int(s_idx.val), 0, len(sampled_frames) - 1))
        frame_img = sampled_frames[i]
        cx = sampled_cx[i] + float(s_dx.val)
        cy = sampled_cy[i] + float(s_dy.val)
        eff_w = max(1.0, sampled_base_w_px[i] * float(s_scale.val))
        eff_h = max(1.0, sampled_base_h_px[i] * float(s_scale.val))
        map_patch = _extract_map_patch_for_p2p(cx, cy, eff_w, eff_h)

        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
        patch_gray = cv2.cvtColor(map_patch, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=2500)
        kf, df = orb.detectAndCompute(frame_gray, None)
        km, dm = orb.detectAndCompute(patch_gray, None)
        if df is None or dm is None or len(kf) < 8 or len(km) < 8:
            print("P2P ORB: not enough keypoints.")
            return

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(df, dm, k=2)
        good: list[Any] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 4:
            print(f"P2P ORB: too few matches ({len(good)}).")
            return

        good = sorted(good, key=lambda m: float(m.distance))
        max_pairs = max(4, int(args.p2p_orb_max_pairs))
        picked = good[:max_pairs]
        p2p_pairs.clear()
        p2p_pending_frame_pt[0] = None
        for m in picked:
            fx, fy = kf[m.queryIdx].pt
            mx_view, my_view = km[m.trainIdx].pt
            mx, my = _map_patch_view_to_global(mx_view, my_view)
            p2p_pairs.append(((float(fx), float(fy)), (float(mx), float(my))))

        p2p_mode[0] = True
        _refresh_p2p_tool_view()
        _draw_p2p_overlay_diagnostics()
        _apply_p2p_solution()
        print(f"P2P ORB: loaded {len(p2p_pairs)} suggested pairs.")

    def _toggle_p2p_mode(_event=None) -> None:
        p2p_mode[0] = not p2p_mode[0]
        p2p_pairs.clear()
        p2p_pending_frame_pt[0] = None
        _clear_p2p_tool_artists()
        _clear_p2p_overlay_diagnostics()
        fig.canvas.draw_idle()
        fig_p2p.canvas.draw_idle()
        state = "ON" if p2p_mode[0] else "OFF"
        if p2p_mode[0]:
            fig_p2p.show()
            _refresh_p2p_tool_view()
        print(f"P2P mode: {state}. Click FRAME point then MAP point in Figure 2.")

    def _clear_p2p_pairs(_event=None) -> None:
        p2p_pairs.clear()
        p2p_pending_frame_pt[0] = None
        _clear_p2p_tool_artists()
        _clear_p2p_overlay_diagnostics()
        if p2p_mode[0]:
            _refresh_p2p_tool_view()
        fig.canvas.draw_idle()
        fig_p2p.canvas.draw_idle()
        print("P2P pairs cleared.")

    def _on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes in {ax_p2p_frame, ax_p2p_map} and not p2p_mode[0]:
            p2p_mode[0] = True
            print("P2P mode auto-enabled (clicked in Figure 2).")
        if not p2p_mode[0]:
            return
        if event.inaxes == ax_p2p_frame:
            p2p_pending_frame_pt[0] = (float(event.xdata), float(event.ydata))
            _refresh_p2p_tool_view()
            print(f"P2P frame point selected: ({event.xdata:.1f}, {event.ydata:.1f}). Now click map point.")
            return
        if event.inaxes != ax_p2p_map:
            return
        if p2p_pending_frame_pt[0] is None:
            print("P2P: click a frame point first.")
            return
        map_global = _map_patch_view_to_global(float(event.xdata), float(event.ydata))
        p2p_pairs.append((p2p_pending_frame_pt[0], map_global))
        p2p_pending_frame_pt[0] = None
        _refresh_p2p_tool_view()
        _draw_p2p_overlay_diagnostics()
        _apply_p2p_solution()

    def _on_key(event) -> None:
        if event.key == "left":
            s_idx.set_val(max(s_idx.valmin, s_idx.val - 1.0))
        elif event.key == "right":
            s_idx.set_val(min(s_idx.valmax, s_idx.val + 1.0))
        elif event.key == "shift+left":
            s_rot.set_val(s_rot.val - 5.0)
        elif event.key == "shift+right":
            s_rot.set_val(s_rot.val + 5.0)
        elif event.key == "[":
            s_scale.set_val(max(s_scale.valmin, s_scale.val - 0.01))
        elif event.key == "]":
            s_scale.set_val(min(s_scale.valmax, s_scale.val + 0.01))
        elif event.key == "{":
            s_scale.set_val(max(s_scale.valmin, s_scale.val - 0.05))
        elif event.key == "}":
            s_scale.set_val(min(s_scale.valmax, s_scale.val + 0.05))
        elif event.key == "up":
            s_dy.set_val(s_dy.val - 1.0)
        elif event.key == "down":
            s_dy.set_val(s_dy.val + 1.0)
        elif event.key and event.key.lower() == "a":
            s_dx.set_val(s_dx.val - 1.0)
        elif event.key and event.key.lower() == "d":
            s_dx.set_val(s_dx.val + 1.0)
        elif event.key and event.key.lower() == "m":
            _auto_match_rotation()
        elif event.key and event.key.lower() == "p":
            _toggle_p2p_mode()
        elif event.key and event.key.lower() == "o":
            _suggest_p2p_pairs_orb()
        elif event.key and event.key.lower() == "x":
            _clear_p2p_pairs()
        elif event.key and event.key.lower() == "s":
            _save()

    s_idx.on_changed(_on_change)
    s_scale.on_changed(_on_change)
    s_rot.on_changed(_on_change)
    s_dx.on_changed(_on_change)
    s_dy.on_changed(_on_change)
    s_op.on_changed(_on_change)
    b_save.on_clicked(_save)
    b_match.on_clicked(_auto_match_rotation)
    b_p2p.on_clicked(_toggle_p2p_mode)
    b_p2p_clear.on_clicked(_clear_p2p_pairs)
    b_p2p_orb.on_clicked(_suggest_p2p_pairs_orb)
    fig_p2p.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig_p2p.canvas.mpl_connect("key_press_event", _on_key)

    _draw_overlays(0, init_dx, init_dy, init_rot, init_scale, init_opacity)
    _refresh_p2p_tool_view()
    _draw_p2p_overlay_diagnostics()
    print("Interactive controls:")
    print(" - Frame slider or Left/Right")
    print(" - Scale slider or [ ] (fine), { } (coarse)")
    print(" - Rotation slider or Shift+Left/Shift+Right")
    print(" - Offset X slider or A/D")
    print(" - Offset Y slider or Up/Down")
    print(" - Opacity slider")
    print(" - Press 'm' or click Auto Rot (M) to estimate rotation from map patch")
    print(" - Press 'p' or click P2P Rot (P): enable Figure 2 point matching")
    print(" - In Figure 2: click FRAME point then corresponding MAP point (repeat)")
    print(" - Press 'o' or click P2P ORB (O): auto-suggest P2P pairs from ORB")
    print(" - Press 'x' or click Clear P2P (X): clear matched pairs")
    print(" - Press 's' or click Save PNG+JSON")
    print(f"Initial manual source: {state_path}")
    print(f"Frame size used: {frame_size[0]}x{frame_size[1]}")
    print(
        f"Camera FOV used: hfov={hfov_deg:.2f} deg, vfov={vfov_deg:.2f} deg, altitude_offset={args.altitude_offset_m:.2f} m"
    )
    print(f"Black transparency threshold: {args.black_transparent_threshold}")
    print(f"Rotation matcher: {args.rotation_matcher}")
    print(f"Map shown at original TIFF size: {map_w_orig}x{map_h_orig}")
    print(f"Output PNG: {args.output}")
    print(f"Output manual JSON: {args.manual_output_json}")
    plt.show()


if __name__ == "__main__":
    main()
