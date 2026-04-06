from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sat_test_up_count_track import create_test_dataloader


MODEL_PIXEL_SCALE_TAG = 33550
MODEL_TIEPOINT_TAG = 33922


def load_full_map_with_extent(map_path: str) -> tuple[np.ndarray, list[float]]:
    img = Image.open(map_path)
    tags = img.tag_v2
    if MODEL_PIXEL_SCALE_TAG not in tags or MODEL_TIEPOINT_TAG not in tags:
        raise ValueError(f"Missing GeoTIFF tags in map file: {map_path}")

    scale = tags[MODEL_PIXEL_SCALE_TAG]
    tiepoints = tags[MODEL_TIEPOINT_TAG]
    if len(scale) < 2 or len(tiepoints) < 6:
        raise ValueError(f"Invalid GeoTIFF scale/tiepoint tags in: {map_path}")

    scale_x = float(scale[0])
    scale_y = float(scale[1])
    tie_i = float(tiepoints[0])
    tie_j = float(tiepoints[1])
    tie_x = float(tiepoints[3])
    tie_y = float(tiepoints[4])

    width, height = img.size
    x0 = tie_x + (0.0 - tie_i) * scale_x
    x1 = tie_x + (float(width - 1) - tie_i) * scale_x
    y0 = tie_y - (0.0 - tie_j) * scale_y
    y1 = tie_y - (float(height - 1) - tie_j) * scale_y
    extent = [x0, x1, y1, y0]
    rgb = img.convert("RGB")
    return np.asarray(rgb), extent


def main() -> None:
    loader = create_test_dataloader(
        scene="0007",
        images_root="test_images",
        gps_root="test_gps",
        references_root="test_references",
        resize_hw=(448, 448),  # defined image resolution
        resize_mode="letterbox",
        sat_compat=True,
        map_crop_meters=26.0,
        map_output_hw=(896, 896),
        batch_size=1,
    )
    loader_iter = iter(loader)
    first_batch = next(loader_iter, None)
    if first_batch is None:
        return

    running = [True]

    def on_key(event) -> None:
        if event.key == "q":
            running[0] = False

    image = first_batch["image"][0].permute(1, 2, 0).cpu().numpy()
    map_x = first_batch["map_axes_x"][0].cpu().numpy()
    map_y = first_batch["map_axes_y"][0].cpu().numpy()
    center_x = 0.5 * float(map_x[0] + map_x[-1])
    center_y = 0.5 * float(map_y[0] + map_y[-1])
    overlay_extent = [float(map_x[0]), float(map_x[-1]), float(map_y[-1]), float(map_y[0])]
    full_map_img, full_map_extent = load_full_map_with_extent(first_batch["map_path"][0])
    
    fig_image, (ax_image, ax_full_map) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        num="Camera + Full Map",
    )
    fig_map, ax_map = plt.subplots(1, 1, figsize=(14, 11), num="Map Overlay")
    fig_image.canvas.mpl_connect("key_press_event", on_key)
    fig_map.canvas.mpl_connect("key_press_event", on_key)
    im1 = ax_image.imshow(image)
    image_center_marker = ax_image.scatter(
        [0.5 * (image.shape[1] - 1)],
        [0.5 * (image.shape[0] - 1)],
        c="lime",
        s=80,
        marker="x",
        linewidths=2.0,
    )
    ax_full_map.imshow(full_map_img, extent=full_map_extent)
    full_map_marker = ax_full_map.scatter(
        [center_x],
        [center_y],
        c="red",
        s=140,
        marker="x",
        linewidths=2.5,
    )
    ax_map.imshow(full_map_img, extent=full_map_extent)
    im_overlay = ax_map.imshow(image, extent=overlay_extent, alpha=0.45)
    gps_marker = ax_map.scatter([center_x], [center_y], c="yellow", s=35, marker="x", linewidths=1.5)
    lat, lon, alt = first_batch["gps"][0].tolist()
    scene = first_batch["scene"][0]
    image_name = first_batch["image_name"][0]
    ax_image.set_title(f"Scene {scene} frame {image_name}")
    ax_image.axis("off")
    ax_full_map.set_title("Full-resolution reference map")
    ax_full_map.set_xlabel("Map X")
    ax_full_map.set_ylabel("Map Y")
    ax_map.set_title("Full map + image overlay (center localized by GPS)")
    ax_map.set_xlabel("Map X")
    ax_map.set_ylabel("Map Y")
    fig_image.suptitle(f"Scene {scene} frame {image_name}")
    fig_map.suptitle(f"GPS lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}")
    fig_image.tight_layout()
    fig_map.tight_layout()
    plt.ion()
    plt.show(block=False)
    plt.pause(0.001)

    for batch in loader_iter:
        if not running[0]:
            break
        
        image = batch["image"][0].permute(1, 2, 0).cpu().numpy()
        lat, lon, alt = batch["gps"][0].tolist()
        scene = batch["scene"][0]
        image_name = batch["image_name"][0]
        map_x = batch["map_axes_x"][0].cpu().numpy()
        map_y = batch["map_axes_y"][0].cpu().numpy()
        center_x = 0.5 * float(map_x[0] + map_x[-1])
        center_y = 0.5 * float(map_y[0] + map_y[-1])
        overlay_extent = [float(map_x[0]), float(map_x[-1]), float(map_y[-1]), float(map_y[0])]

        im1.set_data(image)
        image_center_marker.set_offsets(
            [[0.5 * (image.shape[1] - 1), 0.5 * (image.shape[0] - 1)]]
        )
        full_map_marker.set_offsets([[center_x, center_y]])
        im_overlay.set_data(image)
        im_overlay.set_extent(overlay_extent)
        gps_marker.set_offsets([[center_x, center_y]])
        ax_image.set_title(f"Scene {scene} frame {image_name}")
        fig_image.suptitle(f"Scene {scene} frame {image_name}")
        fig_map.suptitle(f"GPS lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}")
        fig_image.canvas.draw_idle()
        fig_map.canvas.draw_idle()
        plt.pause(0.001)

    plt.ioff()
    plt.close(fig_image)
    plt.close(fig_map)


if __name__ == "__main__":
    main()
