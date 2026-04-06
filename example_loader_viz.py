from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from sat_test_up_count_track import create_test_dataloader


def _to_hwc_image(batch_image: Any):
    # batch_image: [1, 3, H, W] float32 in [0,1]
    return batch_image[0].permute(1, 2, 0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dataloader output (batch=1) and change frame at a fixed interval."
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene id, e.g. 0007 (uses test_images/test_gps with georeferenced TIFF map folder).",
    )
    parser.add_argument("--sequences-root", default="test_images", help="Root directory with sequence folders.")
    parser.add_argument("--gps-root", default="test_gps", help="Root directory with scene GPS txt files.")
    parser.add_argument(
        "--references-root",
        default="test_references",
        help="Root directory with reference tiffs.",
    )
    parser.add_argument(
        "--map-size-meters",
        type=float,
        default=26.0,
        help="When scene mode is used, crop this map area size in meters around GPS center.",
    )
    parser.add_argument(
        "--map-size-px",
        type=int,
        default=896,
        help="When scene mode is used, resize map crop to this square pixel size.",
    )
    parser.add_argument("--interval-sec", type=float, default=0.1, help="Seconds between frame changes.")
    parser.add_argument("--height", type=int, default=224, help="Target image height.")
    parser.add_argument("--width", type=int, default=224, help="Target image width.")
    parser.add_argument(
        "--letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep aspect ratio with padding (default: enabled). Use --no-letterbox for stretch resize.",
    )
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"--height and --width must be positive, got {args.height}x{args.width}")
    if args.map_size_meters <= 0:
        raise ValueError(f"--map-size-meters must be positive, got {args.map_size_meters}")
    if args.map_size_px <= 0:
        raise ValueError(f"--map-size-px must be positive, got {args.map_size_px}")

    resize_mode = "letterbox" if args.letterbox else "stretch"
    if args.scene:
        loader = create_test_dataloader(
            scene=args.scene,
            images_root=args.sequences_root,
            gps_root=args.gps_root,
            references_root=args.references_root,
            batch_size=1,
            num_workers=0,
            resize_hw=(args.height, args.width),
            resize_mode=resize_mode,
            sat_compat=True,
            map_crop_meters=args.map_size_meters,
            map_output_hw=(args.map_size_px, args.map_size_px),
        )
    else:
        loader = create_test_dataloader(
            sequences_root=args.sequences_root,
            batch_size=1,
            num_workers=0,
            resize_hw=(args.height, args.width),
            resize_mode=resize_mode,
        )

    iterator = iter(loader)

    fig, ax = plt.subplots(figsize=(6, 6))
    assert isinstance(ax, Axes)
    ax.set_axis_off()

    first = next(iterator)
    img = _to_hwc_image(first["image"])
    artist = ax.imshow(img)
    assert isinstance(artist, AxesImage)

    def _title_from_batch(batch: dict[str, Any]) -> str:
        seq = batch["sequence"][0]
        name = batch["image_name"][0]
        gps = batch["gps"][0].tolist()
        proc_wh = batch["processed_size_wh"][0]
        pad = batch["pad_ltrb"][0]
        if "map" in batch:
            map_wh = batch["map_size_wh"][0]
            return (
                f"scene={seq} | image={name} | mode={resize_mode} | out={proc_wh[0]}x{proc_wh[1]} | "
                f"map={map_wh[0]}x{map_wh[1]} | pad={pad} | "
                f"lat={gps[0]:.6f}, lon={gps[1]:.6f}, alt={gps[2]:.2f}"
            )
        return (
            f"seq={seq} | image={name} | mode={resize_mode} | out={proc_wh[0]}x{proc_wh[1]} | "
            f"pad={pad} | lat={gps[0]:.6f}, lon={gps[1]:.6f}, alt={gps[2]:.2f}"
        )

    ax.set_title(_title_from_batch(first), fontsize=9)

    def _update(_frame_idx: int) -> None:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        artist.set_data(_to_hwc_image(batch["image"]))
        ax.set_title(_title_from_batch(batch), fontsize=9)
        return None

    _anim = FuncAnimation(fig, _update, interval=max(1, int(args.interval_sec * 1000)), blit=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
