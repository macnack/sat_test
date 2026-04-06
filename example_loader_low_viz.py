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
        description="Visualize loader_low (224x224 letterbox, batch=1) and change frame every 0.1 second."
    )
    parser.add_argument("--scene", default=None, help="Scene id, e.g. 0007 (uses test_images/test_gps/test_references).")
    parser.add_argument("--sequences-root", default="test_images", help="Root directory with sequence folders.")
    parser.add_argument("--gps-root", default="test_gps", help="Root directory with scene GPS txt files.")
    parser.add_argument("--references-root", default="test_references", help="Root directory with reference tiffs.")
    parser.add_argument("--interval-sec", type=float, default=0.1, help="Seconds between frame changes.")
    args = parser.parse_args()

    if args.scene:
        loader_low = create_test_dataloader(
            scene=args.scene,
            images_root=args.sequences_root,
            gps_root=args.gps_root,
            references_root=args.references_root,
            map_scale=4,
            batch_size=1,
            num_workers=0,
            resize_hw=(224, 224),
            resize_mode="letterbox",
        )
    else:
        loader_low = create_test_dataloader(
            sequences_root=args.sequences_root,
            batch_size=1,
            num_workers=0,
            resize_hw=(224, 224),
            resize_mode="letterbox",
        )

    iterator = iter(loader_low)

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
        if "map" in batch:
            map_shape = tuple(batch["map"].shape[-2:])
            return (
                f"scene={seq} | image={name} | map_hw={map_shape} | "
                f"lat={gps[0]:.6f}, lon={gps[1]:.6f}, alt={gps[2]:.2f}"
            )
        return f"seq={seq} | image={name} | lat={gps[0]:.6f}, lon={gps[1]:.6f}, alt={gps[2]:.2f}"

    ax.set_title(_title_from_batch(first), fontsize=10)

    def _update(_frame_idx: int) -> None:
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader_low)
            batch = next(iterator)

        artist.set_data(_to_hwc_image(batch["image"]))
        ax.set_title(_title_from_batch(batch), fontsize=10)
        return None

    _anim = FuncAnimation(fig, _update, interval=max(1, int(args.interval_sec * 1000)), blit=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
