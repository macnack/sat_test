import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sat_roma_gps_sync_core as core


def _make_scene(true_offset, n=50, speed=20.0, crop_m=400.0, size=896):
    """Aircraft moving east at `speed` m/s; predictions correspond to GT shifted by
    `true_offset` (scalar or per-frame array). Each crop is centered on the Δt=0 GPS."""
    t = np.arange(n, dtype=np.float64)

    def gps_world(time):
        return (speed * float(time), 0.0)

    off = np.full(n, true_offset) if np.isscalar(true_offset) else np.asarray(true_offset)
    pred_world = np.array([gps_world(t[i] + off[i]) for i in range(n)], dtype=np.float64)
    gsd = crop_m / size
    centers = [gps_world(t[i]) for i in range(n)]

    def mk_w2p(cx, cy):
        return lambda x, y, cx=cx, cy=cy: ((x - cx) / gsd + size / 2.0,
                                           (y - cy) / gsd + size / 2.0)

    world_to_px = [mk_w2p(cx, cy) for (cx, cy) in centers]
    crop_wh = [(size, size)] * n
    return pred_world, t, gps_world, world_to_px, crop_wh


def test_sweep_recovers_constant_offset():
    pred, t, gw, w2p, cw = _make_scene(true_offset=1.5, crop_m=400.0)
    curve, best = core.sweep_constant(pred, t, gw, w2p, cw, np.arange(-3, 3.01, 0.25))
    assert best is not None
    assert abs(best["dt"] - 1.5) < 0.26  # recovers injected offset to grid resolution
    assert best["median"] < 1e-6         # error vanishes at the true offset


def test_sweep_flags_out_of_crop():
    # Tight 80 m crop (half = 40 m): at speed 20 m/s, |dt| > 2 s leaves the crop.
    pred, t, gw, w2p, cw = _make_scene(true_offset=0.0, crop_m=80.0)
    curve, best = core.sweep_constant(pred, t, gw, w2p, cw, np.arange(-5, 5.01, 1.0))
    by_dt = {round(c["dt"], 1): c for c in curve}
    assert by_dt[0.0]["n_oor"] == 0          # centered GT always in crop
    assert by_dt[5.0]["n_valid"] == 0        # 100 m east -> all out of crop
    assert abs(best["dt"]) < 1e-9            # best is the true zero offset


def test_fit_drift_recovers_a_b():
    n = 50
    t = np.arange(n, dtype=np.float64)
    a0, b0 = 0.5, 0.05
    pred, _, gw, w2p, cw = _make_scene(true_offset=a0 + b0 * t, n=n, crop_m=400.0)
    a, b, resid, nval = core.fit_drift(
        pred, t, gw, w2p, cw,
        a_grid=np.arange(-1.0, 1.01, 0.1), b_grid=np.arange(-0.1, 0.101, 0.01))
    assert abs(a - a0) < 0.11 and abs(b - b0) < 0.011
    assert nval == n and float(np.median(resid)) < 1e-6


def test_errors_at_marks_failed_localization_invalid():
    pred, t, gw, w2p, cw = _make_scene(true_offset=0.0)
    pred[3] = [np.nan, np.nan]  # failed homography
    errs, valid = core.errors_at(pred, t, gw, w2p, cw, 0.0)
    assert not valid[3] and np.isnan(errs[3])
    assert valid.sum() == len(t) - 1


def test_block_bootstrap_ci_brackets_and_handles_empty():
    vals = np.arange(1.0, 101.0)
    lo, hi = core.block_bootstrap_ci(vals, np.median, block_len=10, n_boot=2000, seed=0)
    assert lo < np.median(vals) < hi
    lo2, hi2 = core.block_bootstrap_ci(np.array([]), np.median)
    assert np.isnan(lo2) and np.isnan(hi2)


def test_project_translation():
    T = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], dtype=float)
    pts = np.array([[0.0, 0.0], [10.0, 5.0]])
    assert np.allclose(core._project(T, pts), pts + np.array([3.0, -2.0]))
