#!/usr/bin/env python3
"""GPS time-sync / drift diagnostic for the up-count-track drone scenes.

Localizes drone frames inside the GPS-centered reference TIFF once (SAT-RoMa
4adxis71), then sweeps a time offset on the GPS side only to test whether the
residual error is a GPS<->frame time-sync / drift artifact rather than model error.
up-count's GPS is index-based (frame N <-> GPS second N), so a start offset and/or
a frame-rate drift are prime suspects. Shares sat_roma_gps_sync_core.py with samolot.
See docs spec 2026-06-26-gps-sync-edge-diagnostic-design.md (samolot repo).

Run with the sat_roma venv (has romatch) from this repo / worktree:
    /home/maciej/Github/sat_roma/.venv/bin/python sat_roma_gps_sync_upcount.py --scene 0001
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import sat_roma_gps_sync_core as core

log = logging.getLogger("gps_sync_uc")

DEFAULT_SAT_ROMA_REPO = Path("/home/maciej/Github/sat_roma")
DEFAULT_CKPT_REL = "workspace/checkpoints/train_roma_sat_4adxis71_latest.pth"

# GeoTIFF tag ids (mirror up-count/vis_first_frame_on_tiff + samolot datamodule).
_GEOKEY_TAG = 34735
_PROJECTED_CRS_GEOKEY = 3072


def _main_repo_root(start: Path) -> Path:
    try:
        common = subprocess.run(["git", "-C", str(start), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return Path(common).resolve().parent
    except Exception:
        return start.resolve()


def _select_indices(n_total: int, num_pairs: int) -> list[int]:
    if num_pairs >= n_total:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, num_pairs).round().astype(int).tolist()))


def _read_epsg(tiff_path: Path) -> int:
    """ProjectedCSTypeGeoKey (3072) from the GeoKeyDirectory; defaults to 2180."""
    from PIL import Image
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(tiff_path) as img:
            gk = img.tag_v2.get(_GEOKEY_TAG)
    finally:
        Image.MAX_IMAGE_PIXELS = old
    if not gk or len(gk) < 4:
        return 2180
    idx = 4
    for _ in range(int(gk[3])):
        if idx + 4 > len(gk):
            break
        key_id, loc, count, value = (int(gk[idx + k]) for k in range(4))
        if key_id == _PROJECTED_CRS_GEOKEY and loc == 0 and count == 1:
            return int(value)
        idx += 4
    return 2180


def _gps_interp_from_csv(gps_txt: Path):
    """test_gps/<scene>.txt: time_s, lat, lon, alt -> interpolator t -> (lat, lon)."""
    data = np.loadtxt(gps_txt, delimiter=",")
    ts, lat, lon = data[:, 0], data[:, 1], data[:, 2]
    order = np.argsort(ts)
    ts, lat, lon = ts[order], lat[order], lon[order]

    def interp(t):
        return float(np.interp(t, ts, lat)), float(np.interp(t, ts, lon))
    return interp


def _axes_georef(map_axes_x, map_axes_y):
    """Build px<->world from the crop's world-coordinate axes (linspace per output px)."""
    ax = np.asarray(map_axes_x, dtype=np.float64)
    ay = np.asarray(map_axes_y, dtype=np.float64)
    w, h = len(ax), len(ay)
    x0, sx = ax[0], (ax[-1] - ax[0]) / max(1, w - 1)
    y0, sy = ay[0], (ay[-1] - ay[0]) / max(1, h - 1)

    def px_to_world(u, v):
        return (x0 + u * sx, y0 + v * sy)

    def world_to_px(x, y):
        return ((x - x0) / sx, (y - y0) / sy)
    return px_to_world, world_to_px, (w, h)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="0001")
    ap.add_argument("--sat-roma-repo", type=Path, default=DEFAULT_SAT_ROMA_REPO)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--num-pairs", type=int, default=100)
    # The reference _crop.tiff is only ~308 m across, so the crop must fit inside it
    # (a larger crop pads with black). 200 m is a balanced default; the ideal GSD-match
    # (~4x the ~99 m drone footprint) would need ~400 m, which the reference can't supply.
    ap.add_argument("--crop-meters", type=float, default=200.0,
                    help="Reference (im_B) edge length in metres. Larger widens the "
                         "detectable time-offset range (true point must stay in-crop).")
    ap.add_argument("--dt-range", nargs=3, type=float, default=[-8.0, 8.0, 0.25],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--drift-slope", nargs=3, type=float, default=[-0.05, 0.05, 0.005],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--block-len", type=int, default=10)
    ap.add_argument("--num-viz", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resize-mode", default="safe_square",
                    choices=["stretch", "letterbox", "center_crop", "safe_square"],
                    help="im_A preprocessing; safe_square = samolot-style centered safe-square "
                         "crop (r<=frac) then resize. DJI Mini 2 video is rectilinear (k~0).")
    ap.add_argument("--safe-radius-frac", type=float, default=0.69,
                    help="safe radius (fraction of half-diagonal) for --resize-mode safe_square; "
                         "0.69 = square corners sit on the r=0.69 circle for a 16:9 frame "
                         "(DJI Mini 2 is rectilinear, so this is framing, not distortion)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    main_root = _main_repo_root(here)
    sat_repo = args.sat_roma_repo.resolve()
    for p in (str(sat_repo), str(main_root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from sat_test_up_count_track.torch_dataloader import SyncedGpsImageDataset
    from pyproj import Transformer
    from romatch.eval.multimodel_homography import resolve_multimodel_cfg

    ckpt = args.ckpt or (sat_repo / DEFAULT_CKPT_REL)
    images_root = main_root / "test_images"
    gps_root = main_root / "test_gps"
    references_root = main_root / "test_references"
    out_dir = Path(args.out) if args.out else (main_root / "output" / "gps_sync" / args.scene)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_tiff = next(references_root.glob(f"{args.scene}_*.tif*"), None)
    if ref_tiff is None:
        raise FileNotFoundError(f"no reference TIFF for scene {args.scene} in {references_root}")
    epsg = _read_epsg(ref_tiff)
    gps_interp = _gps_interp_from_csv(gps_root / f"{args.scene}.txt")
    to_world_tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def gps_world(t):
        lat, lon = gps_interp(t)
        x, y = to_world_tf.transform(lon, lat)
        return (float(x), float(y))

    import cv2
    cv2.setRNGSeed(0)
    device = torch.device(args.device)
    log.info("loading %s on %s | scene %s | ref %s (EPSG:%d)",
             Path(ckpt).name, device, args.scene, ref_tiff.name, epsg)
    model, _ = core.load_best_model(Path(ckpt), device, sat_repo, normalize_input=True)
    cfg = resolve_multimodel_cfg()

    ds = SyncedGpsImageDataset(
        scene=args.scene, images_root=str(images_root), gps_root=str(gps_root),
        references_root=str(references_root), resize_hw=(224, 224), map_scale=4,
        resize_mode=args.resize_mode, safe_radius_frac=args.safe_radius_frac,
        sat_compat=True, map_crop_meters=args.crop_meters, map_output_hw=(896, 896))
    indices = _select_indices(len(ds), args.num_pairs)
    log.info("scene frames=%d -> %d pairs (crop %.0f m)", len(ds), len(indices), args.crop_meters)

    pred_world, ts, world_to_px, crop_wh = [], [], [], []
    labels, pred_uv_l, H_l, imA_l, imB_l = [], [], [], [], []
    for bi in range(0, len(indices), args.batch_size):
        chunk = indices[bi:bi + args.batch_size]
        samples = [ds[i] for i in chunk]
        im_A = torch.stack([s["im_A"] for s in samples])
        im_B = torch.stack([s["im_B"] for s in samples])
        res = core.localize_centers(model, im_A, im_B, cfg, device)
        for b, s in enumerate(samples):
            p2w, w2p, wh = _axes_georef(s["map_axes_x"].numpy(), s["map_axes_y"].numpy())
            ts.append(float(s["gps_time"]))
            world_to_px.append(w2p)
            crop_wh.append(wh)
            labels.append(s["im_A_label"])
            r = res[b]
            if r is None:
                pred_world.append([np.nan, np.nan])
                pred_uv_l.append(None)
                H_l.append(None)
            else:
                pw = p2w(*r["uv"])
                pred_world.append([pw[0], pw[1]])
                pred_uv_l.append(r["uv"])
                H_l.append(r["H"])
            imA_l.append(im_A[b].cpu().numpy())
            imB_l.append(im_B[b].cpu().numpy())
        log.info("  localized %d/%d", min(bi + args.batch_size, len(indices)), len(indices))

    pred_world = np.array(pred_world, dtype=np.float64)
    ts = np.array(ts, dtype=np.float64)

    lo, hi, step = args.dt_range
    offsets = np.arange(lo, hi + 1e-9, step)
    curve, best = core.sweep_constant(pred_world, ts, gps_world, world_to_px, crop_wh, offsets)
    best_dt = best["dt"] if best else 0.0
    blo, bhi, bstep = args.drift_slope
    a, b_slope, resid, nval = core.fit_drift(
        pred_world, ts, gps_world, world_to_px, crop_wh, offsets, np.arange(blo, bhi + 1e-9, bstep))
    err0, valid0 = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, 0.0)
    e0 = err0[valid0]
    resid_ci = core.block_bootstrap_ci(resid, np.median, block_len=args.block_len) if len(resid) else (float("nan"),) * 2

    summary = {
        "scene": args.scene, "checkpoint": Path(ckpt).name, "crop_meters": args.crop_meters,
        "n_pairs": int(len(ts)), "n_localized": int(valid0.sum()),
        "error_at_dt0_median": float(np.median(e0)) if len(e0) else float("nan"),
        "best_dt_s": float(best_dt),
        "error_at_best_dt_median": float(best["median"]) if best else float("nan"),
        "drift_a_s": float(a), "drift_b_s_per_s": float(b_slope),
        "drift_residual_median": float(np.median(resid)) if len(resid) else float("nan"),
        "drift_residual_ci95": list(resid_ci),
        "block_len": int(args.block_len), "effective_n": round(nval / max(1, args.block_len), 1),
        "verdict": core.verdict(float(np.median(e0)) if len(e0) else float("nan"), best_dt,
                                best["median"] if best else float("nan"),
                                float(np.median(resid)) if len(resid) else float("nan"), b_slope),
    }

    _write_csv(out_dir / "gps_sync.csv", labels, ts, pred_world, gps_world, world_to_px,
               crop_wh, best_dt, a, b_slope)
    (out_dir / "gps_sync_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "gps_sync_curve.png").unlink(missing_ok=True)
    core.render_curve(curve, best_dt, summary["drift_residual_median"], out_dir / "gps_sync_curve.png")
    _render_viz(args.num_viz, labels, imA_l, imB_l, pred_uv_l, H_l, ts, pred_world,
                gps_world, world_to_px, crop_wh, best_dt, out_dir)
    _write_report(out_dir, summary)

    log.info("\n=== GPS sync diagnostic (up-count scene %s) ===", args.scene)
    log.info("error @dt=0: %.2f m  ->  @best dt=%.2fs: %.2f m",
             summary["error_at_dt0_median"], best_dt, summary["error_at_best_dt_median"])
    log.info("drift: offset(t)=%.2f%+.4f*t  residual median=%.2f m [%.2f, %.2f]",
             a, b_slope, summary["drift_residual_median"], *resid_ci)
    log.info("verdict: %s", summary["verdict"])
    log.info("wrote %s", out_dir)


def _write_csv(path, labels, ts, pred_world, gps_world, world_to_px, crop_wh, best_dt, a, b):
    e0, _ = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, 0.0)
    eb, _ = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, best_dt)
    ed, _ = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, a + b * ts)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "source_time", "pred_x", "pred_y", "err_dt0_m", "err_bestdt_m", "err_drift_m"])
        for i in range(len(ts)):
            w.writerow([labels[i], ts[i], pred_world[i][0], pred_world[i][1], e0[i], eb[i], ed[i]])


def _render_viz(num_viz, labels, imA_l, imB_l, pred_uv_l, H_l, ts, pred_world,
                gps_world, world_to_px, crop_wh, best_dt, out_dir):
    valid = [i for i in range(len(ts)) if pred_uv_l[i] is not None]
    if not valid or num_viz <= 0:
        return
    err0, _ = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, 0.0)
    errb, _ = core.errors_at(pred_world, ts, gps_world, world_to_px, crop_wh, best_dt)
    valid.sort(key=lambda i: errb[i] if np.isfinite(errb[i]) else 1e9)
    if len(valid) > num_viz:
        sel = np.linspace(0, len(valid) - 1, num_viz).round().astype(int)
        valid = [valid[i] for i in sel]
    items, imA_s, imB_s = [], [], []
    for i in valid:
        gt0 = world_to_px[i](*gps_world(ts[i]))
        gtb = world_to_px[i](*gps_world(ts[i] + best_dt))
        corners = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], dtype=np.float64)
        quad = core._project(H_l[i], corners).tolist() if H_l[i] is not None else None
        items.append({"label": labels[i], "pred_uv": list(pred_uv_l[i]),
                      "gt0_uv": [float(gt0[0]), float(gt0[1])],
                      "gtbest_uv": [float(gtb[0]), float(gtb[1])],
                      "err0": float(err0[i]) if np.isfinite(err0[i]) else -1.0,
                      "errbest": float(errb[i]) if np.isfinite(errb[i]) else -1.0,
                      "quad": quad})
        imA_s.append(imA_l[i])
        imB_s.append(imB_l[i])
    n = core.render_panels(items, imA_s, imB_s, out_dir / "viz")
    log.info("wrote %d viz panels", n)


def _write_report(out_dir, s):
    lo, hi = s["drift_residual_ci95"]
    txt = f"""# GPS time-sync / drift diagnostic — up-count scene {s['scene']}

- **Checkpoint:** `{s['checkpoint']}`  | **Reference edge (im_B):** {s['crop_meters']:.0f} m
- **Pairs:** {s['n_pairs']} ({s['n_localized']} localized); single trajectory, effective n ~ {s['effective_n']}

## Result

| quantity | value |
|---|---|
| median error @ Δt=0 | {s['error_at_dt0_median']:.2f} m |
| best constant offset Δt* | {s['best_dt_s']:.2f} s |
| median error @ Δt* | {s['error_at_best_dt_median']:.2f} m |
| drift fit offset(t) | {s['drift_a_s']:.2f} {s['drift_b_s_per_s']:+.4f}·t (s) |
| residual median (after drift) | {s['drift_residual_median']:.2f} m [95% CI {lo:.2f}, {hi:.2f}] |

**Verdict:** {s['verdict']}

## Notes

up-count GPS is index-based (frame N ↔ GPS second N), so a start offset and/or a
frame-rate drift are plausible. Localization runs once (crop centered on Δt=0 GPS);
the time offset is swept on the GPS side only. A sharp drop from @Δt=0 to @Δt* ⇒ a
GPS↔frame time lag; a nonzero drift slope `b` ⇒ a frame-rate/clock-rate mismatch; the
residual after the best drift fit is the irreducible model/sensor error. Increase
`--crop-meters` to detect larger offsets. See `gps_sync_curve.png` and `viz/`.
"""
    (out_dir / "report.md").write_text(txt)


if __name__ == "__main__":
    main()
