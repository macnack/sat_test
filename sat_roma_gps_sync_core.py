#!/usr/bin/env python3
"""Project-agnostic core for the SAT-RoMa GPS time-sync / drift diagnostic.

Shared by the samolot and up-count-track adapters (identical copy in each repo,
mirroring how both repos carry their own sat_compat dataloader).

The model predicts the query (im_A) center in WORLD coordinates (projected metres),
which do not depend on the ground-truth GPS. So we localize ONCE (crop centered on
the Δt=0 GPS) and then vary a time offset only on the GT side:
    GT(Δt) = gps_world(source_time + Δt)
A sharp error dip at Δt≠0 ⇒ GPS↔frame time-sync; a nonzero best slope b in
offset(t)=a+b·t ⇒ frame-rate drift. The residual at the best (a,b) is the
irreducible model/sensor error.

Pure functions (errors_at / sweep_constant / fit_drift / block_bootstrap_ci) work in
world coordinates with plain callables, so they are unit-tested without a GPU/model.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

TRAIN_REF_PX = 560.0  # 4adxis71 reference baseline (im_A 224 * map ratio 4 / ... ); see sat_roma_demo


# --------------------------------------------------------------------------- #
# Model loading (bypasses romatch.ablation package init -> h5py/pyvips ABI clash)
# --------------------------------------------------------------------------- #
def _load_state_dict_helpers(sat_repo: Path):
    p = sat_repo / "romatch" / "ablation" / "matchers" / "state_dict.py"
    spec = importlib.util.spec_from_file_location("_sat_state_dict", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_best_model(ckpt: Path, device, sat_repo: Path, normalize_input: bool = True):
    """Build SatRoMaModel + load checkpoint without importing romatch.ablation."""
    import torch
    from experiments.train_roma_sat_model import SatRoMaModel
    sd = _load_state_dict_helpers(sat_repo)

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    state, _ = sd.extract_model_state(raw)
    state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    encoder_type = sd.infer_sat_encoder_type_from_state_dict(state) or "dinov2"
    proj_dim = sd.infer_proj_dim_from_state_dict(state) or 512
    cls_to_coord_res = sd.infer_cls_to_coord_res_from_state_dict(state)

    model = SatRoMaModel(
        resolution="medium", pretrain_model="auto", encoder_type=encoder_type,
        proj_dim=proj_dim, cls_to_coord_res=cls_to_coord_res, compile_backbone=False,
        normalize_input=normalize_input, attenuate_cert=False,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, (cls_to_coord_res or 14)


# --------------------------------------------------------------------------- #
# Localization (forward -> gm_cls -> predicted im_B pixel of the im_A center)
# --------------------------------------------------------------------------- #
def _project(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1))], axis=1)
    p = (H @ homo.T).T
    return p[:, :2] / p[:, 2:3]


def localize_centers(model, im_A_b, im_B_b, cfg, device):
    """Return per-sample dict {uv, H, n_corr} (predicted im_B pixel of the im_A
    center + pixel homography) or None when the homography could not be estimated."""
    import torch
    from romatch.eval.multimodel_homography import estimate_multimodel_homography
    from ransac_multimodel.transforms import convert_to_pixel_homography

    gaussian_cfg, optimize_cfg = cfg
    im_A = im_A_b.to(device)
    im_B = im_B_b.to(device)
    h_a, w_a = im_A.shape[-2:]
    h_b, w_b = im_B.shape[-2:]
    scale_factor = float(((int(h_a) * int(w_a)) / (TRAIN_REF_PX * TRAIN_REF_PX)) ** 0.5)

    with model.exposed_intermediates(), torch.no_grad():
        out = model.forward({"im_A": im_A, "im_B": im_B},
                            batched=False, scale_factor=scale_factor)
    gm_cls_b = out.get(16, {}).get("gm_cls") if isinstance(out, dict) else None
    if gm_cls_b is None:
        return [None] * im_A.shape[0]

    results = []
    for b in range(gm_cls_b.shape[0]):
        gm_cls = gm_cls_b[b].detach().cpu().float()
        mm = estimate_multimodel_homography(
            gm_cls, gaussian_cfg=gaussian_cfg, optimize_cfg=optimize_cfg, refine=False)
        n_corr = int(mm.pts_A.shape[0]) if mm.pts_A is not None else 0
        if mm.H_final is None:
            results.append(None)
            continue
        C = int(gm_cls.shape[0])
        H_pix = convert_to_pixel_homography(
            np.asarray(mm.H_final, dtype=np.float64),
            in_patch_dim=int(gm_cls.shape[-1]), out_patch_dim=int(round(C ** 0.5)),
            crop_res=(int(h_a), int(w_a)), map_res=(int(h_b), int(w_b)))
        if not (isinstance(H_pix, np.ndarray) and H_pix.shape == (3, 3) and np.isfinite(H_pix).all()):
            results.append(None)
            continue
        uv = _project(H_pix.astype(np.float64),
                      np.array([[float(w_a) / 2.0, float(h_a) / 2.0]]))[0]
        results.append({"uv": (float(uv[0]), float(uv[1])), "H": H_pix.astype(np.float64),
                        "n_corr": n_corr})
    return results


# --------------------------------------------------------------------------- #
# Pure diagnostic math (world coords; GPU-free, unit-tested)
# --------------------------------------------------------------------------- #
def errors_at(pred_world: np.ndarray, t: np.ndarray, gps_world: Callable[[float], tuple],
              world_to_px: list, crop_wh: list, dt) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame error (m) and in-crop validity mask at time offset `dt`.

    pred_world: (N,2) predicted world points (NaN rows = failed localization).
    gps_world(time) -> (x, y) world; GT(Δt) = gps_world(t + dt).
    world_to_px[i](x, y) -> (u, v); crop_wh[i] = (W, H). A frame is invalid when its
    shifted GT falls outside the im_B crop (the model could not have matched it there).
    dt: scalar or (N,) array (the latter for drift offset(t)=a+b·t).
    """
    n = len(t)
    dt_arr = np.full(n, float(dt)) if np.isscalar(dt) else np.asarray(dt, dtype=np.float64)
    errs = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(pred_world[i]).all():
            continue
        gx, gy = gps_world(float(t[i]) + float(dt_arr[i]))
        u, v = world_to_px[i](gx, gy)
        w, h = crop_wh[i]
        if not (0.0 <= u < w and 0.0 <= v < h):
            continue
        errs[i] = math.hypot(pred_world[i][0] - gx, pred_world[i][1] - gy)
        valid[i] = True
    return errs, valid


def _n_localized(pred_world: np.ndarray) -> int:
    return int(np.isfinite(pred_world).all(axis=1).sum())


def sweep_constant(pred_world, t, gps_world, world_to_px, crop_wh, offsets, min_valid_frac=0.5):
    """Median error vs constant offset. Returns (curve, best) where curve is a list of
    {dt, median, n_valid, n_oor}. `best` is the min-median entry among offsets that keep
    >= min_valid_frac of localized frames in-crop (so the offset can't 'win' by pushing
    most frames out of the reference and scoring the lucky remainder)."""
    n = len(t)
    need = max(1, int(round(min_valid_frac * _n_localized(pred_world))))
    curve = []
    for dt in offsets:
        errs, valid = errors_at(pred_world, t, gps_world, world_to_px, crop_wh, dt)
        e = errs[valid]
        curve.append({"dt": float(dt),
                      "median": float(np.median(e)) if len(e) else float("nan"),
                      "n_valid": int(valid.sum()),
                      "n_oor": int(n - valid.sum())})
    cand = [c for c in curve if c["n_valid"] >= need and np.isfinite(c["median"])]
    best = min(cand, key=lambda c: c["median"]) if cand else None
    return curve, best


def fit_drift(pred_world, t, gps_world, world_to_px, crop_wh, a_grid, b_grid, min_valid_frac=0.5):
    """Grid-search offset(t)=a+b·t minimizing median error, subject to keeping
    >= min_valid_frac of localized frames in-crop. Returns (a, b, residual_errs, n_valid)."""
    t = np.asarray(t, dtype=np.float64)
    need = max(1, int(round(min_valid_frac * _n_localized(pred_world))))
    best = None
    for a in a_grid:
        for b in b_grid:
            errs, valid = errors_at(pred_world, t, gps_world, world_to_px, crop_wh, a + b * t)
            if int(valid.sum()) < need:
                continue
            e = errs[valid]
            med = float(np.median(e))
            if best is None or med < best[0]:
                best = (med, float(a), float(b), e, int(valid.sum()))
    if best is None:
        return float("nan"), float("nan"), np.array([]), 0
    _, a, b, e, nval = best
    return a, b, e, nval


def verdict(e0: float, best_dt: float, e_best: float, resid: float, b_slope: float) -> str:
    """Honest, quantitative one-liner. Reports what each stage actually buys rather
    than a binary 'detected', and names the dominant cause of the residual."""
    if not np.isfinite(e0) or e0 <= 0:
        return "INCONCLUSIVE: no successful localizations"
    imp_dt = (e0 - e_best) / e0
    imp_drift = (e0 - resid) / e0
    parts = []
    if abs(best_dt) > 0.25 and imp_dt > 0.05:
        parts.append(f"constant offset Δt≈{best_dt:.1f}s cuts error {100*imp_dt:.0f}% "
                     f"({e0:.1f}→{e_best:.1f} m)")
    else:
        parts.append("no constant time offset improves the fit")
    if abs(b_slope) > 0.005 and (imp_drift - imp_dt) > 0.05:
        parts.append(f"linear drift (b={b_slope:+.4f} s/s) helps further → {resid:.1f} m")
    dominant = "largely a GPS time-sync artifact" if imp_drift > 0.5 \
        else "dominated by model/sensor/domain error"
    parts.append(f"residual ~{resid:.1f} m ⇒ {dominant}")
    return "; ".join(parts)


def block_bootstrap_ci(values: np.ndarray, stat, block_len: int = 10,
                       n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI via circular moving-block bootstrap (single autocorrelated
    trajectory). block_len=1 reduces to the iid bootstrap."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        boots[i] = stat(values[idx[:n]])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# --------------------------------------------------------------------------- #
# Viz (clean subprocess: pyvips poisons matplotlib freetype in-process)
# --------------------------------------------------------------------------- #
_PANEL_SRC = r'''
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = np.load(sys.argv[1]); meta = json.loads(sys.argv[2]); viz_dir = sys.argv[3]
im_A, im_B = npz["im_A"], npz["im_B"]
m = meta["items"]
n = 0
for k, it in enumerate(m):
    a = np.transpose(im_A[k], (1, 2, 0)).clip(0, 1)
    b = np.transpose(im_B[k], (1, 2, 0)).clip(0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(a); ax[0].set_title("im_A (query): %s" % it["label"]); ax[0].axis("off")
    ax[1].imshow(b); ax[1].axis("off")
    ax[1].set_title("err @dt=0: %.1f m   @best: %.1f m" % (it["err0"], it["errbest"]))
    if it.get("quad"):
        q = np.array(it["quad"] + [it["quad"][0]])
        ax[1].plot(q[:, 0], q[:, 1], "y-", lw=2, label="predicted footprint")
    pu, pv = it["pred_uv"]; ax[1].plot(pu, pv, "rx", ms=16, mew=3, label="predicted")
    g0u, g0v = it["gt0_uv"]; ax[1].plot(g0u, g0v, "g+", ms=18, mew=3, label="GT (dt=0)")
    gbu, gbv = it["gtbest_uv"]; ax[1].plot(gbu, gbv, "c*", ms=14, label="GT (best dt)")
    ax[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("%s/pair_%02d_err%.0fm.png" % (viz_dir, k, it["errbest"]), dpi=120)
    plt.close(fig); n += 1
print(n)
'''


def render_panels(items: list[dict], im_A_stack, im_B_stack, viz_dir: Path) -> int:
    """items[k]: {label, pred_uv, gt0_uv, gtbest_uv, err0, errbest, quad?}."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    for old in viz_dir.glob("pair_*.png"):
        old.unlink()
    npz_path = viz_dir / "_panels.npz"
    np.savez_compressed(npz_path, im_A=np.stack(im_A_stack), im_B=np.stack(im_B_stack))
    meta = json.dumps({"items": items})
    out = subprocess.run([sys.executable, "-c", _PANEL_SRC, str(npz_path), meta, str(viz_dir)],
                         capture_output=True, text=True, timeout=600)
    npz_path.unlink(missing_ok=True)
    if out.returncode != 0:
        return 0
    return int(out.stdout.strip() or 0)


_CURVE_SRC = r'''
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.loads(sys.argv[1]); out = sys.argv[2]
dts = [c["dt"] for c in d["curve"]]; med = [c["median"] for c in d["curve"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(dts, med, "-o", ms=3)
ax.axvline(d["best_dt"], color="r", ls="--", label="best dt = %.2f s" % d["best_dt"])
ax.axhline(d["residual_median"], color="g", ls=":", label="drift residual = %.1f m" % d["residual_median"])
ax.set_xlabel("GPS time offset dt (s)"); ax.set_ylabel("median localization error (m)")
ax.set_title("Error vs GPS time offset"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
'''


def render_curve(curve: list[dict], best_dt: float, residual_median: float, out_path: Path):
    payload = json.dumps({"curve": curve, "best_dt": float(best_dt),
                          "residual_median": float(residual_median)})
    subprocess.run([sys.executable, "-c", _CURVE_SRC, payload, str(out_path)],
                   capture_output=True, text=True, timeout=120)
