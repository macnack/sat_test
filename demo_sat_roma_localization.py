"""Measure SAT-RoMa localization error against GPS ground truth.

Loads drone images + GPS-centered satellite map crops via the
``sat_test_up_count_track`` dataloader, runs SAT-RoMa (DINOv2 or DINOv3)
inference via ``SatRomaMatcher`` from the ablation experiment, and computes
the localization error in metres between the estimated position and the
GPS ground truth.

Three estimation methods are reported:
  1. **warp_center** — read the dense warp at the drone-image center pixel
     to get the corresponding map pixel directly.
  2. **homography** — Gaussian-fit correspondences from coarse decoder +
     RANSAC + Mahalanobis-weighted least-squares (same as ablation).
  3. **weighted_centroid** — certainty-weighted centroid of all sampled
     match positions on the map (baseline sanity check).

Usage example (DINOv3):
    python demo_sat_roma_localization.py \\
        --scene 0007 \\
        --images-root test_images \\
        --gps-root test_gps \\
        --references-root test_references \\
        --checkpoint-path ../../workspace/checkpoints/train_roma_sat_<RUN_ID>_best_latest.pth \\
        --encoder-type dinov3 \\
        --pretrain-model vit_large_patch16_dinov3 \\
        --output-dir outputs/localization_0007

Usage example (DINOv2):
    python demo_sat_roma_localization.py \\
        --scene 0007 \\
        --checkpoint-path ../../workspace/checkpoints/<dinov2_ckpt>.pth \\
        --encoder-type dinov2 \\
        --output-dir outputs/localization_0007
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# -- Ensure the experiments directory is on sys.path so we can import
#    SatRomaMatcher and get_model
ROOT = Path(__file__).resolve().parents[2]  # sat_roma repo root
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from sat_test_up_count_track import create_test_dataloader
from ablation_experiment_feature_matchers_sat import SatRomaMatcher
from romatch.datasets.sat import TFWTransform, _read_tfw


# ---------------------------------------------------------------------------
# Pixel → world CRS via TFWTransform
# ---------------------------------------------------------------------------

def _find_tfw(tiff_path: Path) -> Path:
    """Locate a .tfw world file next to the reference TIFF."""
    for suffix in (".tfw", ".tifw", ".tiffw"):
        candidate = tiff_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    # Try <stem>.tfw alongside .tiff/.tif
    candidate = tiff_path.parent / (tiff_path.stem + ".tfw")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No .tfw world file found for {tiff_path}. "
        "Expected e.g. " + str(tiff_path.with_suffix(".tfw"))
    )


def _im_b_pixel_to_world(
    tfw: TFWTransform,
    col: float,
    row: float,
    map_crop_origin_xy: tuple[int, int],
    map_crop_size_wh: tuple[int, int],
    im_b_width_px: int,
    im_b_height_px: int,
) -> tuple[float, float]:
    """Convert an im_B eval-pixel to world CRS coords via TFWTransform."""
    return tfw.im_b_pixel_to_world(
        col=col,
        row=row,
        x_map=map_crop_origin_xy[0],
        y_map=map_crop_origin_xy[1],
        map_sample_size_px=map_crop_size_wh[0],
        im_b_width_px=im_b_width_px,
        im_b_height_px=im_b_height_px,
    )


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def _estimate_warp_center(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    h_B: int, w_B: int,
) -> tuple[float, float, float]:
    """Read the dense warp at the drone-image center → map pixel coords.

    warp: [H, W, 4] — (imA_x, imA_y, imB_x, imB_y) all in [-1, 1]
    Returns (est_px_x, est_px_y, center_certainty).
    """
    warp_h, warp_w = warp.shape[:2]
    cy, cx = warp_h // 2, warp_w // 2
    imB_x_norm = warp[cy, cx, 2].item()
    imB_y_norm = warp[cy, cx, 3].item()
    est_px_x = (imB_x_norm + 1.0) * w_B / 2.0
    est_px_y = (imB_y_norm + 1.0) * h_B / 2.0
    cert = certainty[cy, cx].item()
    return est_px_x, est_px_y, cert


def _estimate_weighted_centroid(
    kpts_B: torch.Tensor,
    cert: torch.Tensor,
) -> tuple[float, float]:
    """Certainty-weighted centroid of matched map keypoints."""
    weights = cert / cert.sum()
    est_px_x = (kpts_B[:, 0] * weights).sum().item()
    est_px_y = (kpts_B[:, 1] * weights).sum().item()
    return est_px_x, est_px_y


def _project_center_through_H(
    H: torch.Tensor | np.ndarray,
    h_A: int, w_A: int,
) -> tuple[float, float, bool]:
    """Project the drone-image center through a pixel homography."""
    if isinstance(H, torch.Tensor):
        H = H.cpu().numpy()
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all():
        return 0.0, 0.0, False
    center = np.array([w_A / 2.0, h_A / 2.0, 1.0])
    projected = H @ center
    if abs(projected[2]) < 1e-8:
        return 0.0, 0.0, False
    return float(projected[0] / projected[2]), float(projected[1] / projected[2]), True


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float [0,1] tensor → [H,W,C] uint8 BGR numpy for cv2."""
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)[..., ::-1]  # RGB→BGR


def _draw_cross(img: np.ndarray, x: float, y: float, color: tuple,
                size: int = 8, thickness: int = 2) -> None:
    ix, iy = int(round(x)), int(round(y))
    cv2.line(img, (ix - size, iy), (ix + size, iy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (ix, iy - size), (ix, iy + size), color, thickness, cv2.LINE_AA)


def save_matched_viz(
    im_A: torch.Tensor,
    im_B: torch.Tensor,
    kpts_A: torch.Tensor,
    kpts_B: torch.Tensor,
    cert: torch.Tensor,
    gt_px: tuple[float, float],
    est_warp: tuple[float, float],
    est_hom: tuple[float, float] | None,
    est_centroid: tuple[float, float],
    error_warp: float,
    error_hom: float,
    error_centroid: float,
    image_name: str,
    save_path: Path,
    max_matches_draw: int = 200,
) -> None:
    """Save a side-by-side match visualization with position markers.

    Layout: [drone image | satellite map crop]
    - Green cross: GT position (map center)
    - Red cross: warp_center estimate
    - Blue cross: homography estimate
    - Yellow cross: centroid estimate
    - Lines: sampled matches coloured by certainty
    """
    vis_A = _tensor_to_uint8(im_A)
    vis_B = _tensor_to_uint8(im_B)

    h_A, w_A = vis_A.shape[:2]
    h_B, w_B = vis_B.shape[:2]

    canvas_h = max(h_A, h_B)
    if h_A != canvas_h:
        scale = canvas_h / h_A
        vis_A = cv2.resize(vis_A, (int(w_A * scale), canvas_h), interpolation=cv2.INTER_LINEAR)
    if h_B != canvas_h:
        scale = canvas_h / h_B
        vis_B = cv2.resize(vis_B, (int(w_B * scale), canvas_h), interpolation=cv2.INTER_LINEAR)

    disp_w_A = vis_A.shape[1]
    scale_A_y = vis_A.shape[0] / h_A
    scale_A_x = disp_w_A / w_A
    scale_B_y = vis_B.shape[0] / h_B
    scale_B_x = vis_B.shape[1] / w_B

    canvas = np.concatenate([vis_A, vis_B], axis=1)
    offset_x = disp_w_A

    # Draw match lines (subsample for readability)
    n = kpts_A.shape[0]
    cert_np = cert.cpu().numpy()
    indices = np.random.choice(n, min(max_matches_draw, n), replace=False) if n > 0 else np.array([], dtype=int)

    kA = kpts_A.cpu().numpy()
    kB = kpts_B.cpu().numpy()
    cert_min, cert_max = (cert_np.min(), cert_np.max()) if n > 0 else (0.0, 1.0)
    cert_range = max(cert_max - cert_min, 1e-6)

    for i in indices:
        t = (cert_np[i] - cert_min) / cert_range
        color = (0, int(255 * t), int(255 * (1 - t)))
        pt_a = (int(round(kA[i, 0] * scale_A_x)), int(round(kA[i, 1] * scale_A_y)))
        pt_b = (int(round(kB[i, 0] * scale_B_x)) + offset_x, int(round(kB[i, 1] * scale_B_y)))
        cv2.line(canvas, pt_a, pt_b, color, 1, cv2.LINE_AA)

    for i in indices:
        t = (cert_np[i] - cert_min) / cert_range
        color = (0, int(255 * t), int(255 * (1 - t)))
        pt_a = (int(round(kA[i, 0] * scale_A_x)), int(round(kA[i, 1] * scale_A_y)))
        pt_b = (int(round(kB[i, 0] * scale_B_x)) + offset_x, int(round(kB[i, 1] * scale_B_y)))
        cv2.circle(canvas, pt_a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_b, 2, color, -1, cv2.LINE_AA)

    # Position markers on the map (right panel)
    _draw_cross(canvas, gt_px[0] * scale_B_x + offset_x, gt_px[1] * scale_B_y,
                (0, 255, 0), size=12, thickness=3)
    _draw_cross(canvas, est_warp[0] * scale_B_x + offset_x, est_warp[1] * scale_B_y,
                (0, 0, 255), size=10, thickness=2)
    if est_hom is not None:
        _draw_cross(canvas, est_hom[0] * scale_B_x + offset_x, est_hom[1] * scale_B_y,
                    (255, 128, 0), size=10, thickness=2)
    _draw_cross(canvas, est_centroid[0] * scale_B_x + offset_x, est_centroid[1] * scale_B_y,
                (0, 255, 255), size=10, thickness=2)

    # Legend
    y0, line_h = 15, 18
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    cv2.putText(canvas, image_name, (5, y0), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    y0 += line_h
    cv2.putText(canvas, "[green]  GT (map center)", (5, y0), font, fs, (0, 255, 0), 1, cv2.LINE_AA)
    y0 += line_h
    cv2.putText(canvas, f"[red]    warp_center: {error_warp:.2f}m", (5, y0), font, fs, (0, 0, 255), 1, cv2.LINE_AA)
    y0 += line_h
    hom_label = f"{error_hom:.2f}m" if np.isfinite(error_hom) else "FAIL"
    cv2.putText(canvas, f"[blue]   homography:  {hom_label}", (5, y0), font, fs, (255, 128, 0), 1, cv2.LINE_AA)
    y0 += line_h
    cv2.putText(canvas, f"[yellow] centroid:    {error_centroid:.2f}m", (5, y0), font, fs, (0, 255, 255), 1, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_evaluation(
    matcher: SatRomaMatcher,
    loader: Any,
    device: torch.device,
    sample_num: int,
    tfw: TFWTransform,
    viz_dir: Path | None = None,
    max_matches_draw: int = 200,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model = matcher.model

    def _px_to_world(col: float, row: float, crop_origin: tuple[int, int],
                     crop_size: tuple[int, int], im_b_w: int, im_b_h: int) -> tuple[float, float]:
        return _im_b_pixel_to_world(tfw, col, row, crop_origin, crop_size, im_b_w, im_b_h)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            im_A = batch["im_A"].to(device)       # [1,3,H_A,W_A]
            im_B = batch["im_B"].to(device)       # [1,3,H_B,W_B]
            gps = batch["gps"][0]                  # [lat, lon, alt]
            image_name = batch["image_name"][0]
            crop_origin = batch["map_crop_origin_xy"][0]  # (left_px, top_px)
            crop_size = batch["map_crop_size_wh"][0]      # (crop_w, crop_h)

            h_A, w_A = im_A.shape[-2:]
            h_B, w_B = im_B.shape[-2:]

            # GT = map center (crop is centred on the GPS position)
            gt_px_x, gt_px_y = w_B / 2.0, h_B / 2.0
            gt_crs_x, gt_crs_y = _px_to_world(gt_px_x, gt_px_y, crop_origin, crop_size, w_B, h_B)

            # Run SatRomaMatcher — does dense match + coarse decoder forward
            # + find_gaussians + optimize_homography + convert_to_pixel_homography
            matched = matcher.match(im_A[0], im_B[0])

            # --- Method 1: warp center (from dense warp) ---
            if matched.warp is not None:
                warp = matched.warp   # [H,W,4]
                cert_map = matched.certainty  # [H,W]
                wc_px_x, wc_px_y, wc_cert = _estimate_warp_center(warp, cert_map, h_B, w_B)
            else:
                wc_px_x, wc_px_y, wc_cert = w_B / 2.0, h_B / 2.0, 0.0
            wc_crs_x, wc_crs_y = _px_to_world(wc_px_x, wc_px_y, crop_origin, crop_size, w_B, h_B)
            wc_err = math.sqrt((wc_crs_x - gt_crs_x) ** 2 + (wc_crs_y - gt_crs_y) ** 2)

            # --- Method 2: homography from ransac_multimodel ---
            H_est = matched.homography_est
            H_init = matched.homography_init
            H_pixel = H_est if H_est is not None else H_init
            hom_px_x, hom_px_y, hom_ok = _project_center_through_H(H_pixel, h_A, w_A)
            if hom_ok:
                hom_crs_x, hom_crs_y = _px_to_world(hom_px_x, hom_px_y, crop_origin, crop_size, w_B, h_B)
                hom_err = math.sqrt((hom_crs_x - gt_crs_x) ** 2 + (hom_crs_y - gt_crs_y) ** 2)
            else:
                hom_crs_x, hom_crs_y, hom_err = float("nan"), float("nan"), float("nan")

            # --- Method 3: weighted centroid (from sparse warp samples) ---
            kpts_A = matched.keypoints0
            kpts_B = matched.keypoints1
            cert_sparse = matched.confidence
            if kpts_B is not None and kpts_B.shape[0] > 0 and cert_sparse is not None:
                wt_px_x, wt_px_y = _estimate_weighted_centroid(kpts_B, cert_sparse)
            elif matched.warp is not None:
                warp_matches, warp_cert = model.sample(
                    matched.warp, matched.certainty, num=sample_num,
                )
                _, fallback_kpts_B = model.to_pixel_coordinates(
                    warp_matches, h_A, w_A, h_B, w_B,
                )
                wt_px_x, wt_px_y = _estimate_weighted_centroid(fallback_kpts_B, warp_cert)
                kpts_A = model.to_pixel_coordinates(
                    warp_matches[..., :2], h_A, w_A,
                )
                kpts_B = fallback_kpts_B
                cert_sparse = warp_cert
            else:
                wt_px_x, wt_px_y = w_B / 2.0, h_B / 2.0
            wt_crs_x, wt_crs_y = _px_to_world(wt_px_x, wt_px_y, crop_origin, crop_size, w_B, h_B)
            wt_err = math.sqrt((wt_crs_x - gt_crs_x) ** 2 + (wt_crs_y - gt_crs_y) ** 2)

            # --- Visualization ---
            if viz_dir is not None and kpts_A is not None and kpts_B is not None:
                stem = Path(image_name).stem
                viz_cert = cert_sparse if cert_sparse is not None else torch.ones(kpts_A.shape[0])
                save_matched_viz(
                    im_A=im_A[0].cpu(),
                    im_B=im_B[0].cpu(),
                    kpts_A=kpts_A,
                    kpts_B=kpts_B,
                    cert=viz_cert,
                    gt_px=(gt_px_x, gt_px_y),
                    est_warp=(wc_px_x, wc_px_y),
                    est_hom=(hom_px_x, hom_px_y) if hom_ok else None,
                    est_centroid=(wt_px_x, wt_px_y),
                    error_warp=wc_err,
                    error_hom=hom_err,
                    error_centroid=wt_err,
                    image_name=image_name,
                    save_path=viz_dir / f"{batch_idx:06d}_{stem}.jpg",
                    max_matches_draw=max_matches_draw,
                )

            row = {
                "index": batch_idx,
                "image_name": image_name,
                "gps_lat": float(gps[0]),
                "gps_lon": float(gps[1]),
                "gps_alt": float(gps[2]),
                "gt_crs_x": gt_crs_x,
                "gt_crs_y": gt_crs_y,
                "warp_center_error_m": wc_err,
                "warp_center_crs_x": wc_crs_x,
                "warp_center_crs_y": wc_crs_y,
                "warp_center_certainty": wc_cert,
                "homography_error_m": hom_err,
                "homography_ok": hom_ok,
                "homography_crs_x": hom_crs_x if hom_ok else None,
                "homography_crs_y": hom_crs_y if hom_ok else None,
                "centroid_error_m": wt_err,
                "centroid_crs_x": wt_crs_x,
                "centroid_crs_y": wt_crs_y,
                "num_matches": int(kpts_A.shape[0]) if kpts_A is not None else 0,
                "mean_certainty": float(cert_sparse.mean().item()) if cert_sparse is not None and cert_sparse.numel() > 0 else 0.0,
            }
            results.append(row)

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(
                    f"  [{batch_idx+1:>4d}/{len(loader)}] {image_name}  "
                    f"warp={wc_err:.2f}m  hom={'FAIL' if not hom_ok else f'{hom_err:.2f}m'}  "
                    f"centroid={wt_err:.2f}m"
                )

    return results


def print_summary(
    results: list[dict[str, Any]],
    scene: str,
    encoder_type: str,
) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    methods = [
        ("warp_center", "warp_center_error_m"),
        ("homography", "homography_error_m"),
        ("centroid", "centroid_error_m"),
    ]

    print(f"\n{'='*70}")
    print(f"Scene: {scene} | Encoder: {encoder_type} | N={n}")
    print(f"{'='*70}")

    for label, key in methods:
        errs = np.array([r[key] for r in results])
        valid = errs[np.isfinite(errs)]
        if len(valid) == 0:
            print(f"\n  [{label}] — no valid estimates")
            continue
        print(f"\n  [{label}]  (valid: {len(valid)}/{n})")
        print(f"    Mean:   {valid.mean():.3f} m")
        print(f"    Median: {np.median(valid):.3f} m")
        print(f"    Std:    {valid.std():.3f} m")
        print(f"    Min:    {valid.min():.3f} m  |  Max: {valid.max():.3f} m")
        for tau in [1.0, 2.0, 5.0, 10.0, 20.0]:
            pct = 100.0 * (valid < tau).sum() / len(valid)
            print(f"    Success @{tau:>4.0f}m: {pct:5.1f}%")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAT-RoMa localization error benchmark against GPS ground truth."
    )
    # Data
    parser.add_argument("--scene", required=True, help="Scene id (e.g. 0007).")
    parser.add_argument("--images-root", default="test_images")
    parser.add_argument("--gps-root", default="test_gps")
    parser.add_argument("--references-root", default="test_references")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--resize-mode", default="stretch",
        choices=["stretch", "letterbox", "center_crop"],
    )
    parser.add_argument(
        "--map-size-meters", type=float, default=100.0,
        help="Satellite map crop size in metres around GPS center.",
    )
    parser.add_argument(
        "--map-size-px", type=int, default=896,
        help="Resize map crop to this square pixel size.",
    )
    # Model
    parser.add_argument(
        "--checkpoint-path", required=True,
        help="Path to SAT-RoMa .pth checkpoint.",
    )
    parser.add_argument(
        "--encoder-type", default="dinov3",
        choices=["dinov3", "dinov2"],
    )
    parser.add_argument(
        "--pretrain-model", default="vit_large_patch16_dinov3",
        help="timm model id for DINOv3 backbone (ignored for DINOv2).",
    )
    parser.add_argument(
        "--resolution", default="medium",
        choices=["low", "medium", "high"],
    )
    # Matching
    parser.add_argument(
        "--sample-num", type=int, default=10000,
        help="Number of matches to sample from the warp (for centroid fallback).",
    )
    parser.add_argument(
        "--match-tau", type=float, default=0.0,
        help="Certainty threshold for SatRomaMatcher sampling.",
    )
    # Output
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for matched_viz images and results JSON. "
             "Creates <output-dir>/matched_viz/ and <output-dir>/results.json.",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save per-sample results to JSON (overrides --output-dir default).",
    )
    parser.add_argument(
        "--max-matches-draw", type=int, default=200,
        help="Max match lines drawn per visualization image.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: cuda if available).",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ---- Create SatRomaMatcher (handles model loading, compile key
    #      remapping, encoder auto-detection, etc.) ----
    matcher = SatRomaMatcher(
        device=device,
        checkpoint_path=args.checkpoint_path,
        encoder_type=args.encoder_type,
        pretrain_model=args.pretrain_model,
        train_resolution=args.resolution,
        sample_num=args.sample_num,
        match_tau=args.match_tau,
        save_heatmaps=False,
        log_missing_gaussians=False,
        save_debug_plots=False,
    )
    encoder_type = args.encoder_type

    # ---- Build dataloader ----
    loader = create_test_dataloader(
        scene=args.scene,
        images_root=args.images_root,
        gps_root=args.gps_root,
        references_root=args.references_root,
        resize_hw=(args.image_height, args.image_width),
        resize_mode=args.resize_mode,
        sat_compat=True,
        map_crop_meters=args.map_size_meters,
        map_output_hw=(args.map_size_px, args.map_size_px),
        batch_size=1,
        num_workers=0,
    )
    print(f"Dataset: scene={args.scene}, {len(loader.dataset)} samples")

    # ---- Load TFW from reference TIFF ----
    # The dataset resolves the map TIFF path; grab it from the first sample.
    map_tiff_path = Path(loader.dataset._samples[0].map_path)
    tfw_path = _find_tfw(map_tiff_path)
    tfw = _read_tfw(tfw_path)
    print(f"TFW: {tfw_path}  (pixel_size={tfw.pixel_size_x:.4f}m)")

    # ---- Resolve output paths ----
    output_dir = Path(args.output_dir) if args.output_dir else None
    viz_dir: Path | None = None
    if output_dir is not None:
        viz_dir = output_dir / "matched_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations -> {viz_dir}")

    # ---- Run inference ----
    results = run_evaluation(
        matcher=matcher,
        loader=loader,
        device=device,
        sample_num=args.sample_num,
        tfw=tfw,
        viz_dir=viz_dir,
        max_matches_draw=args.max_matches_draw,
    )

    # ---- Print summary ----
    print_summary(results, scene=args.scene, encoder_type=encoder_type)

    # ---- Save JSON ----
    json_path = args.output_json
    if json_path is None and output_dir is not None:
        json_path = str(output_dir / "results.json")
    if json_path:
        out_path = Path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _stats(key: str) -> dict[str, float]:
            errs = np.array([r[key] for r in results])
            valid = errs[np.isfinite(errs)]
            if len(valid) == 0:
                return {"valid": 0}
            return {
                "valid": int(len(valid)),
                "mean_m": float(valid.mean()),
                "median_m": float(np.median(valid)),
                "std_m": float(valid.std()),
                "min_m": float(valid.min()),
                "max_m": float(valid.max()),
                "success_at_1m": float((valid < 1.0).mean()),
                "success_at_2m": float((valid < 2.0).mean()),
                "success_at_5m": float((valid < 5.0).mean()),
                "success_at_10m": float((valid < 10.0).mean()),
                "success_at_20m": float((valid < 20.0).mean()),
            }

        summary = {
            "scene": args.scene,
            "encoder_type": encoder_type,
            "checkpoint": args.checkpoint_path,
            "num_samples": len(results),
            "warp_center": _stats("warp_center_error_m"),
            "homography": _stats("homography_error_m"),
            "centroid": _stats("centroid_error_m"),
            "per_sample": results,
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
