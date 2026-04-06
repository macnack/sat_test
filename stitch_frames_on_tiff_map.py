#!/usr/bin/env python3
"""Stitch sequence frames along GPS trajectory and visualize on georeferenced TIFF map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image

from vis_first_frame_on_tiff import _load_geotiff, _project_to_tiff_pixel, _require_positive_size, _resize_with_pillow


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_homography_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    if path.is_absolute():
        raise FileNotFoundError(f"Homography JSON does not exist: {path}")
    alt = Path("test_frame_overlays") / path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Homography JSON does not exist: {path} (or {alt})")


def _load_optional_manual_adjustment(path: Path | None) -> tuple[float, float, float]:
    if path is None:
        return 0.0, 0.0, 0.0
    payload = _read_json(path)
    manual = payload.get("manual_adjustment", {})
    dx = float(manual.get("dx_px", 0.0))
    dy = float(manual.get("dy_px", 0.0))
    rot = float(manual.get("rotation_deg", 0.0))
    return dx, dy, rot


def _sorted_items(metadata: dict) -> list[dict]:
    items = metadata.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Metadata must contain non-empty 'items' list")
    return sorted(items, key=lambda x: int(x.get("image_index", 10**9)))


def _compute_heading_deg(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    if len(px) == 1:
        return np.array([0.0], dtype=float)
    dx = np.gradient(px)
    dy = np.gradient(py)
    return np.degrees(np.arctan2(dy, dx))


def _smooth_heading_deg(heading_deg: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(heading_deg) <= 2:
        return heading_deg
    if window % 2 == 0:
        window += 1
    pad = window // 2
    # Unwrap angles in radians before smoothing to avoid wraparound artifacts.
    heading_rad = np.unwrap(np.deg2rad(heading_deg))
    padded = np.pad(heading_rad, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.rad2deg(smoothed)


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _blend_angles_deg(base_deg: float, est_deg: float, alpha: float) -> float:
    a = np.array([math.cos(math.radians(base_deg)), math.sin(math.radians(base_deg))], dtype=float)
    b = np.array([math.cos(math.radians(est_deg)), math.sin(math.radians(est_deg))], dtype=float)
    v = (1.0 - alpha) * a + alpha * b
    if float(np.hypot(v[0], v[1])) < 1e-9:
        return _wrap_angle_deg(base_deg)
    return _wrap_angle_deg(math.degrees(math.atan2(v[1], v[0])))


def _estimate_rigid_kabsch(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = np.mean(src_xy, axis=0)
    dst_mean = np.mean(dst_xy, axis=0)
    src_centered = src_xy - src_mean
    dst_centered = dst_xy - dst_mean

    h = src_centered.T @ dst_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_mean - (r @ src_mean)
    return r, t


def _ransac_rotation_deg(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    iterations: int,
    reproj_thresh_px: float,
    min_inliers: int,
    seed: int = 0,
) -> tuple[float | None, int]:
    n = len(src_xy)
    if n < 2:
        return None, 0

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(iterations):
        sample_idx = rng.choice(n, size=2, replace=False)
        r_try, t_try = _estimate_rigid_kabsch(src_xy[sample_idx], dst_xy[sample_idx])
        pred = (src_xy @ r_try.T) + t_try
        err = np.linalg.norm(pred - dst_xy, axis=1)
        inliers = err <= reproj_thresh_px
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_count < max(2, min_inliers):
        return None, best_count

    r_best, _t_best = _estimate_rigid_kabsch(src_xy[best_inliers], dst_xy[best_inliers])
    angle = math.degrees(math.atan2(float(r_best[1, 0]), float(r_best[0, 0])))
    return _wrap_angle_deg(angle), best_count


def _estimate_delta_rotations_loftr(
    sampled_frames: list[np.ndarray],
    sampled_base_rot: np.ndarray,
    loftr_max_side: int,
    ransac_iters: int,
    ransac_thresh_px: float,
    ransac_min_inliers: int,
    alpha: float,
) -> tuple[np.ndarray, int, int]:
    if len(sampled_frames) <= 1:
        return sampled_base_rot.copy(), 0, 0

    try:
        import torch
        import torch.nn.functional as f
        from kornia.feature import LoFTR
    except Exception as exc:  # pragma: no cover - optional runtime path
        raise RuntimeError(
            "Kornia/torch not available. Install dependencies and retry with --experimental-kornia-rot-correct."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained="outdoor").to(device).eval()

    def _to_gray_tensor(img_rgb: np.ndarray) -> tuple[torch.Tensor, float, float]:
        img = img_rgb.astype(np.float32) / 255.0
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        ten = torch.from_numpy(gray).to(device=device, dtype=torch.float32)[None, None]
        h, w = gray.shape
        scale_x = 1.0
        scale_y = 1.0
        max_side = max(h, w)
        if max_side > loftr_max_side:
            ratio = float(loftr_max_side) / float(max_side)
            nh = max(16, int(round(h * ratio)))
            nw = max(16, int(round(w * ratio)))
            ten = f.interpolate(ten, size=(nh, nw), mode="bilinear", align_corners=False)
            scale_x = float(w) / float(nw)
            scale_y = float(h) / float(nh)
        return ten, scale_x, scale_y

    corrected = sampled_base_rot.copy()
    used_pairs = 0
    total_pairs = max(0, len(sampled_frames) - 1)

    for i in range(total_pairs):
        img0 = sampled_frames[i]
        img1 = sampled_frames[i + 1]
        t0, sx0, sy0 = _to_gray_tensor(img0)
        t1, sx1, sy1 = _to_gray_tensor(img1)

        with torch.no_grad():
            out = matcher({"image0": t0, "image1": t1})

        k0 = out.get("keypoints0", None)
        k1 = out.get("keypoints1", None)
        if k0 is None or k1 is None:
            continue
        if k0.shape[0] < 8 or k1.shape[0] < 8:
            continue

        pts0 = k0.detach().cpu().numpy().astype(np.float64)
        pts1 = k1.detach().cpu().numpy().astype(np.float64)
        pts0[:, 0] *= sx0
        pts0[:, 1] *= sy0
        pts1[:, 0] *= sx1
        pts1[:, 1] *= sy1

        delta_deg, inliers = _ransac_rotation_deg(
            pts0,
            pts1,
            iterations=ransac_iters,
            reproj_thresh_px=ransac_thresh_px,
            min_inliers=ransac_min_inliers,
            seed=i,
        )
        if delta_deg is None:
            continue

        pred_next = _wrap_angle_deg(corrected[i] + delta_deg)
        corrected[i + 1] = _blend_angles_deg(sampled_base_rot[i + 1], pred_next, alpha=alpha)
        used_pairs += 1

    return corrected, used_pairs, total_pairs


def _experimental_correct_centers(
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    heading_deg: np.ndarray,
    min_step_px: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(centers_x) == 0:
        return centers_x, centers_y, 0
    out_x = centers_x.copy()
    out_y = centers_y.copy()
    corrected = 0
    for i in range(1, len(out_x)):
        dx = float(out_x[i] - out_x[i - 1])
        dy = float(out_y[i] - out_y[i - 1])
        dist = float(np.hypot(dx, dy))
        if dist >= min_step_px:
            continue

        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ang = float(np.deg2rad(heading_deg[i]))
            ux, uy = float(np.cos(ang)), float(np.sin(ang))

        out_x[i] = out_x[i - 1] + ux * min_step_px
        out_y[i] = out_y[i - 1] + uy * min_step_px
        corrected += 1

    return out_x, out_y, corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch sampled frames along trajectory on TIFF map.")
    parser.add_argument("--tiff", type=Path, default=Path("test_references/0001_year_2024_crop.tiff"))
    parser.add_argument("--metadata", type=Path, default=Path("test_images/0001/gps_metadata.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("test_images/0001"))
    parser.add_argument("--map-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1024, 1024))
    parser.add_argument("--frame-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(256, 256))
    parser.add_argument("--sample-step", type=int, default=40, help="Use every N-th frame for stitching.")
    parser.add_argument("--max-frames", type=int, default=120, help="Hard cap on stitched frames after sampling.")
    parser.add_argument("--frame-opacity", type=float, default=0.55)
    parser.add_argument("--draw-trajectory", action="store_true", default=True)
    parser.add_argument("--no-draw-trajectory", action="store_false", dest="draw_trajectory")
    parser.add_argument("--rotate-by-heading", action="store_true", default=False)
    parser.add_argument("--no-rotate-by-heading", action="store_false", dest="rotate_by_heading")
    parser.add_argument(
        "--heading-smooth-window",
        type=int,
        default=1,
        help="Odd window size for heading smoothing (1 disables smoothing).",
    )
    parser.add_argument(
        "--homography-json",
        type=Path,
        default=Path("test_frame_overlays/0001_first_frame_on_tiff_interactive_homography.json"),
        help="Optional homography JSON filename/path (for example: 0001_first_frame_on_tiff_interactive_homography.json). "
        "rotation_deg + dx_px + dy_px are read from manual_adjustment and applied to all frames.",
    )
    parser.add_argument(
        "--manual-adjustment-json",
        type=Path,
        default=None,
        help="Backward-compatible alias of --homography-json.",
    )
    parser.add_argument(
        "--experimental-kornia-rot-correct",
        action="store_true",
        help="Experimental: estimate frame-to-frame relative rotation with Kornia LoFTR + RANSAC and refine angles.",
    )
    parser.add_argument("--kornia-loftr-max-side", type=int, default=320, help="Max side for LoFTR inputs.")
    parser.add_argument("--kornia-ransac-iters", type=int, default=300, help="RANSAC iterations per frame pair.")
    parser.add_argument("--kornia-ransac-thresh-px", type=float, default=3.0, help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--kornia-ransac-min-inliers", type=int, default=12, help="Minimum inliers to accept pair rotation.")
    parser.add_argument(
        "--kornia-rotation-alpha",
        type=float,
        default=0.7,
        help="Blend weight for LoFTR rotation chain vs base rotation [0..1].",
    )
    parser.add_argument(
        "--experimental-drift-correct",
        action="store_true",
        help="Experimental: reduce overlap by enforcing minimum distance between consecutive stitched frame centers.",
    )
    parser.add_argument(
        "--experimental-min-step-px",
        type=float,
        default=None,
        help="Experimental with --experimental-drift-correct: minimum center spacing in map pixels "
        "(default: 0.35 * min(frame_width, frame_height)).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_frame_overlays/0001_stitched_trajectory_on_tiff.png"),
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
    if not 0.0 <= args.frame_opacity <= 1.0:
        raise ValueError(f"--frame-opacity must be in [0,1], got {args.frame_opacity}")
    if args.heading_smooth_window < 1:
        raise ValueError(f"--heading-smooth-window must be >= 1, got {args.heading_smooth_window}")
    if args.kornia_loftr_max_side < 32:
        raise ValueError(f"--kornia-loftr-max-side must be >= 32, got {args.kornia_loftr_max_side}")
    if args.kornia_ransac_iters < 10:
        raise ValueError(f"--kornia-ransac-iters must be >= 10, got {args.kornia_ransac_iters}")
    if args.kornia_ransac_thresh_px <= 0:
        raise ValueError(f"--kornia-ransac-thresh-px must be > 0, got {args.kornia_ransac_thresh_px}")
    if args.kornia_ransac_min_inliers < 2:
        raise ValueError(f"--kornia-ransac-min-inliers must be >= 2, got {args.kornia_ransac_min_inliers}")
    if not 0.0 <= args.kornia_rotation_alpha <= 1.0:
        raise ValueError(f"--kornia-rotation-alpha must be in [0,1], got {args.kornia_rotation_alpha}")
    if args.experimental_min_step_px is not None and args.experimental_min_step_px <= 0:
        raise ValueError(f"--experimental-min-step-px must be > 0, got {args.experimental_min_step_px}")

    map_size = _require_positive_size("--map-size", tuple(args.map_size))
    frame_size = _require_positive_size("--frame-size", tuple(args.frame_size))
    homography_arg = args.homography_json if args.homography_json is not None else args.manual_adjustment_json
    homography_path = _resolve_homography_path(homography_arg)
    manual_dx, manual_dy, manual_rot = _load_optional_manual_adjustment(homography_path)

    metadata = _read_json(args.metadata)
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

    resized_map = _resize_with_pillow(map_arr_orig, map_size)
    sx = map_size[0] / map_w_orig
    sy = map_size[1] / map_h_orig
    px = px_orig * sx
    py = py_orig * sy

    sampled_idx = np.arange(0, len(valid_items), args.sample_step, dtype=int)
    if len(sampled_idx) > args.max_frames:
        sampled_idx = sampled_idx[: args.max_frames]
    heading_deg = _compute_heading_deg(px, py)
    heading_deg = _smooth_heading_deg(heading_deg, args.heading_smooth_window)

    map_h, map_w = resized_map.shape[:2]
    frame_h, frame_w = frame_size[1], frame_size[0]

    fig = plt.figure(figsize=(map_w / 100.0, map_h / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(resized_map)

    if args.draw_trajectory:
        ax.plot(px, py, color="yellow", linewidth=2.0, alpha=0.9, zorder=2)
        ax.scatter([px[0]], [py[0]], s=60, c="lime", edgecolors="black", linewidths=0.8, zorder=3)
        ax.scatter([px[-1]], [py[-1]], s=60, c="red", edgecolors="black", linewidths=0.8, zorder=3)

    sampled_px = px[sampled_idx] + manual_dx
    sampled_py = py[sampled_idx] + manual_dy
    corrected_count = 0
    if args.experimental_drift_correct:
        min_step = (
            float(args.experimental_min_step_px)
            if args.experimental_min_step_px is not None
            else 0.35 * float(min(frame_size[0], frame_size[1]))
        )
        sampled_px, sampled_py, corrected_count = _experimental_correct_centers(
            sampled_px,
            sampled_py,
            heading_deg[sampled_idx],
            min_step_px=min_step,
        )

    sampled_frames: list[np.ndarray] = []
    sampled_valid_idx: list[int] = []
    for idx in sampled_idx:
        item = valid_items[int(idx)]
        img_name = str(item["image"])
        img_path = args.images_dir / img_name
        if not img_path.exists():
            continue

        frame_arr = np.array(Image.open(img_path).convert("RGB"))
        frame_arr = _resize_with_pillow(frame_arr, frame_size)
        sampled_frames.append(frame_arr)
        sampled_valid_idx.append(int(idx))

    sampled_base_rot = np.array(
        [
            _wrap_angle_deg(manual_rot + (float(heading_deg[i]) if args.rotate_by_heading else 0.0))
            for i in sampled_valid_idx
        ],
        dtype=float,
    )
    sampled_rot = sampled_base_rot.copy()
    kornia_pairs_used = 0
    kornia_pairs_total = 0
    if args.experimental_kornia_rot_correct and len(sampled_frames) > 1:
        sampled_rot, kornia_pairs_used, kornia_pairs_total = _estimate_delta_rotations_loftr(
            sampled_frames=sampled_frames,
            sampled_base_rot=sampled_base_rot,
            loftr_max_side=args.kornia_loftr_max_side,
            ransac_iters=args.kornia_ransac_iters,
            ransac_thresh_px=args.kornia_ransac_thresh_px,
            ransac_min_inliers=args.kornia_ransac_min_inliers,
            alpha=args.kornia_rotation_alpha,
        )

    stitched_count = 0
    for local_i, idx in enumerate(sampled_valid_idx):
        frame_arr = sampled_frames[local_i]

        cx = float(sampled_px[local_i])
        cy = float(sampled_py[local_i])
        angle = float(sampled_rot[local_i])

        artist = ax.imshow(frame_arr, extent=(0, frame_w, frame_h, 0), alpha=args.frame_opacity, zorder=4)
        transform = (
            Affine2D()
            .translate(-frame_w / 2.0, -frame_h / 2.0)
            .rotate_deg(angle)
            .translate(cx, cy)
            + ax.transData
        )
        artist.set_transform(transform)
        stitched_count += 1

    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.set_axis_off()

    ax.text(
        10,
        20,
        (
            f"seq={metadata.get('sequence', 'unknown')} | stitched={stitched_count} | "
            f"sample_step={args.sample_step} | heading_rot={args.rotate_by_heading} "
            f"| heading_win={args.heading_smooth_window}"
        ),
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
        zorder=10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    print(f"TIFF CRS: {tiff_crs}")
    print(f"Total GPS points in bounds: {len(valid_items)}")
    print(f"Stitched frames: {stitched_count}")
    print(f"Frame size used: {frame_size[0]}x{frame_size[1]}")
    if args.experimental_drift_correct:
        print(f"Experimental drift correction enabled: corrected {corrected_count} frame centers")
    if args.experimental_kornia_rot_correct:
        print(
            f"Experimental Kornia rotation enabled: used {kornia_pairs_used}/{kornia_pairs_total} frame pairs "
            f"(alpha={args.kornia_rotation_alpha:.2f})"
        )
    if homography_path is not None:
        print(
            f"Manual adjustment source: {homography_path} "
            f"(dx={manual_dx:.2f}, dy={manual_dy:.2f}, rot={manual_rot:.2f})"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
