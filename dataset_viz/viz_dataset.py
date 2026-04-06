from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_test_up_count_track import create_test_dataloader


def _to_hwc_float(batch_tensor):
    return batch_tensor[0].permute(1, 2, 0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dataset samples and save PNG previews (image + im_B map crop)."
    )
    parser.add_argument("--scene", type=str, default="0007")
    parser.add_argument("--images-root", type=str, default="test_images")
    parser.add_argument("--gps-root", type=str, default="test_gps")
    parser.add_argument(
        "--references-root",
        type=str,
        default="test_references",
    )
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=("stretch", "letterbox", "center_crop"),
        default="center_crop",
        help="Resize mode for im_A. Use 'center_crop' for geometry-correct output without black borders.",
    )
    parser.add_argument("--map-size-meters", type=float, default=100.0)
    parser.add_argument("--map-size-px", type=int, default=896)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_viz/output"))
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("--image-height and --image-width must be > 0")
    if args.map_size_meters <= 0:
        raise ValueError("--map-size-meters must be > 0")
    if args.map_size_px <= 0:
        raise ValueError("--map-size-px must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    saved = 0
    for i, batch in enumerate(loader):
        if i >= args.max_samples:
            break
        image = _to_hwc_float(batch["im_A"])
        map_img = _to_hwc_float(batch["im_B"])
        lat, lon, alt = batch["gps"][0].tolist()
        seq = batch["scene"][0]
        image_name = batch["image_name"][0]
        map_name = batch["im_B_label"][0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
        axes[0].imshow(image)
        axes[0].set_title(f"im_A: {image_name} ({args.resize_mode})")
        axes[0].axis("off")

        axes[1].imshow(map_img)
        axes[1].set_title(f"im_B: {map_name} ({args.map_size_meters:.1f}m -> {args.map_size_px}px)")
        axes[1].axis("off")

        fig.suptitle(
            f"scene={seq} | idx={int(batch['index'][0])} | "
            f"lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}",
            fontsize=10,
        )
        fig.tight_layout()

        out_path = args.output_dir / f"sample_{i:04d}_{seq}_{Path(image_name).stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} preview image(s) to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
