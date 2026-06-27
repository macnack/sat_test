"""Geometry of the samolot-style safe-square crop (resize_mode='safe_square')."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sat_test_up_count_track.torch_dataloader import _safe_square_crop_box


def test_safe_square_crop_box_4k_height_limited():
    # DJI Mini 2 4K 3840x2160, r<=0.85: height-limited square 2160 at (840, 0).
    # r is non-binding here (square corners reach only ~0.69).
    assert _safe_square_crop_box(3840, 2160, 0.85) == (840, 0, 2160, 2160)


def test_safe_square_crop_box_tight_frac_binds():
    # A small frac makes the r<=frac circle the binding constraint, not the image.
    side = int(round(0.3 * math.hypot(1920, 1080) * math.sqrt(2.0)))
    x0, y0 = (3840 - side) // 2, (2160 - side) // 2
    assert _safe_square_crop_box(3840, 2160, 0.3) == (x0, y0, side, side)


def test_safe_square_crop_box_dji_069():
    # DJI default frac=0.69: square corners land on r=0.69 (just r-binding) -> 2150 sq.
    assert _safe_square_crop_box(3840, 2160, 0.69) == (845, 5, 2150, 2150)


if __name__ == "__main__":
    test_safe_square_crop_box_4k_height_limited()
    test_safe_square_crop_box_tight_frac_binds()
    test_safe_square_crop_box_dji_069()
    print("ok")
