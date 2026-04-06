from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from PIL import Image
from pyproj import CRS, Transformer
from torch.utils.data import DataLoader, Dataset

MODEL_PIXEL_SCALE_TAG = 33550
MODEL_TIEPOINT_TAG = 33922
GEO_KEY_DIRECTORY_TAG = 34735
PROJECTED_CRS_GEOKEY = 3072


def _extract_epsg_from_geokeys(geokeys: tuple[int, ...] | list[int]) -> int | None:
    if len(geokeys) < 4:
        return None
    key_count = int(geokeys[3])
    idx = 4
    for _ in range(key_count):
        if idx + 4 > len(geokeys):
            break
        key_id = int(geokeys[idx])
        tiff_tag_location = int(geokeys[idx + 1])
        count = int(geokeys[idx + 2])
        value_offset = int(geokeys[idx + 3])
        if key_id == PROJECTED_CRS_GEOKEY and tiff_tag_location == 0 and count == 1:
            return value_offset
        idx += 4
    return None


def _load_geotiff_tags(path: Path) -> dict[str, float | str]:
    # Some reference maps are very large (e.g. 30000x30000). We only read tags,
    # so temporarily disabling this safety limit is acceptable here.
    old_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            tags = img.tag_v2
            if MODEL_PIXEL_SCALE_TAG not in tags:
                raise ValueError(f"Missing TIFF tag {MODEL_PIXEL_SCALE_TAG} in {path}")
            if MODEL_TIEPOINT_TAG not in tags:
                raise ValueError(f"Missing TIFF tag {MODEL_TIEPOINT_TAG} in {path}")
            if GEO_KEY_DIRECTORY_TAG not in tags:
                raise ValueError(f"Missing TIFF tag {GEO_KEY_DIRECTORY_TAG} in {path}")

            scale = tags[MODEL_PIXEL_SCALE_TAG]
            tiepoints = tags[MODEL_TIEPOINT_TAG]
            geokeys = tags[GEO_KEY_DIRECTORY_TAG]
    finally:
        Image.MAX_IMAGE_PIXELS = old_max_pixels

    if len(scale) < 2 or len(tiepoints) < 6:
        raise ValueError(f"Invalid GeoTIFF scale/tiepoint tags in {path}")

    epsg = _extract_epsg_from_geokeys(geokeys)
    if epsg is None:
        raise ValueError(f"ProjectedCSTypeGeoKey (3072) not found in {path}")

    return {
        "scale_x": float(scale[0]),
        "scale_y": float(scale[1]),
        "tie_i": float(tiepoints[0]),
        "tie_j": float(tiepoints[1]),
        "tie_x": float(tiepoints[3]),
        "tie_y": float(tiepoints[4]),
        "crs": f"EPSG:{epsg}",
    }


def _project_to_tiff_pixel(
    lon: float,
    lat: float,
    scale_x: float,
    scale_y: float,
    tie_i: float,
    tie_j: float,
    tie_x: float,
    tie_y: float,
    transformer: Transformer,
) -> tuple[float, float]:
    x_map, y_map = transformer.transform(lon, lat)
    px = tie_i + (float(x_map) - tie_x) / scale_x
    py = tie_j + (tie_y - float(y_map)) / scale_y
    return float(px), float(py)


def _parse_gps_txt(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid GPS row at {path}:{line_no}: '{text}'")
            rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def _vips_import() -> Any:
    try:
        import pyvips
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "pyvips is required for scene mode. Install it first, e.g. `pip install pyvips` "
            "and ensure libvips is available on your system."
        ) from exc
    return pyvips


def _vips_to_rgb(image: Any, pyvips: Any) -> Any:
    if image.bands == 1:
        image = image.bandjoin([image, image])
    if image.bands >= 3:
        image = image.extract_band(0, n=3)
    if image.format != "uchar":
        image = image.cast("uchar")
    if image.interpretation != "srgb":
        image = image.copy(interpretation="srgb")
    return image


def _vips_black_rgb(width: int, height: int, pyvips: Any) -> Any:
    black = pyvips.Image.black(width, height)
    black = black.bandjoin([black, black]).cast("uchar")
    return black.copy(interpretation="srgb")


def _vips_to_numpy_hwc(image: Any) -> np.ndarray:
    arr = np.frombuffer(image.write_to_memory(), dtype=np.uint8)
    return arr.reshape(image.height, image.width, image.bands)


def _resize_with_vips(
    image: Any,
    resize_hw: tuple[int, int] | None,
    resize_mode: Literal["stretch", "letterbox", "center_crop"],
    pyvips: Any,
) -> tuple[Any, tuple[float, float], tuple[int, int, int, int]]:
    if resize_hw is None:
        return image, (1.0, 1.0), (0, 0, 0, 0)

    target_h, target_w = resize_hw
    src_w = image.width
    src_h = image.height
    if resize_mode == "stretch":
        sx = target_w / float(src_w)
        sy = target_h / float(src_h)
        resized = image.resize(sx, vscale=sy, kernel="lanczos3")
        return resized, (sx, sy), (0, 0, 0, 0)

    if resize_mode == "center_crop":
        scale = max(target_w / float(src_w), target_h / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize(new_w / float(src_w), vscale=new_h / float(src_h), kernel="lanczos3")
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        cropped = resized.crop(left, top, target_w, target_h)
        return cropped, (scale, scale), (0, 0, 0, 0)

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize(new_w / float(src_w), vscale=new_h / float(src_h), kernel="lanczos3")
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    canvas = _vips_black_rgb(target_w, target_h, pyvips=pyvips)
    return canvas.insert(resized, left, top), (scale, scale), (left, top, right, bottom)


def _extract_patch_with_padding(
    image: Any,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
    pyvips: Any,
) -> tuple[Any, int, int]:
    left = int(round(center_x - crop_w / 2.0))
    top = int(round(center_y - crop_h / 2.0))
    right = left + crop_w
    bottom = top + crop_h

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.width)
    pad_bottom = max(0, bottom - image.height)

    source = image
    if pad_left or pad_top or pad_right or pad_bottom:
        bg = [0] * image.bands
        source = image.embed(
            pad_left,
            pad_top,
            image.width + pad_left + pad_right,
            image.height + pad_top + pad_bottom,
            extend="background",
            background=bg,
        )

    crop = source.crop(left + pad_left, top + pad_top, crop_w, crop_h)
    return crop, left, top


def _resolve_scene_reference_path(scene: str, references_root: Path) -> Path:
    patterns = [
        f"{scene}_year_*_crop.tiff",
        f"{scene}_year_*_crop.tif",
        f"{scene}_*.tiff",
        f"{scene}_*.tif",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(references_root.glob(pattern)))
    uniq = sorted(set(matches))
    if not uniq:
        raise FileNotFoundError(
            f"No reference TIFF found for scene '{scene}' in {references_root}. "
            f"Expected e.g. {scene}_year_2024_crop.tiff"
        )
    return uniq[0]


def _extract_year_from_name(path: Path) -> int | None:
    for token in path.stem.replace("-", "_").split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _resolve_reference_path_flexible(scene: str, references_root: Path) -> Path:
    """Resolve map TIFF for scene mode, with fallback for generic map folders.

    Preferred:
    1) Scene-prefixed files: <scene>_year_*_crop.tiff (legacy behavior)
    2) map.tif[f]
    3) year_*.tif[f] (latest year)
    4) Any *.tif[f] (lexicographically last)
    """
    try:
        return _resolve_scene_reference_path(scene=scene, references_root=references_root)
    except FileNotFoundError:
        pass

    direct_map = [
        references_root / "map.tiff",
        references_root / "map.tif",
    ]
    for candidate in direct_map:
        if candidate.exists():
            return candidate

    year_candidates = sorted(references_root.glob("year_*.tif*"))
    if year_candidates:
        year_with_key = []
        for p in year_candidates:
            y = _extract_year_from_name(p)
            year_with_key.append((y if y is not None else -1, p))
        year_with_key.sort(key=lambda t: (t[0], t[1].name))
        return year_with_key[-1][1]

    all_tiffs = sorted(p for p in references_root.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if all_tiffs:
        return all_tiffs[-1]

    raise FileNotFoundError(
        f"No reference TIFF found in {references_root}. "
        "Expected either scene-prefixed TIFFs, map.tiff, year_*.tiff, or any .tif/.tiff file."
    )


@dataclass(frozen=True)
class _SampleRef:
    sequence: str
    scene: str
    image_name: str
    image_path: Path
    gps_time: float
    latitude: float
    longitude: float
    altitude: float
    map_path: Path | None = None


class SyncedGpsImageDataset(Dataset[dict[str, Any]]):
    """Dataset for synchronized GPS + image metadata in test/inference mode.

    Expected metadata format is the same as generated by `videos_to_frames.py`,
    for example:
      test_images/<sequence>/gps_metadata.json
    """

    def __init__(
        self,
        scene: str | None = None,
        images_root: str | Path | None = None,
        gps_root: str | Path | None = None,
        references_root: str | Path | None = None,
        map_scale: int = 4,
        metadata_files: list[str | Path] | None = None,
        sequences_root: str | Path | None = None,
        metadata_name: str = "gps_metadata.json",
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gps_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        resize_hw: tuple[int, int] | None = None,
        resize_mode: Literal["stretch", "letterbox", "center_crop"] = "stretch",
        sat_compat: bool = False,
        roma_patch_size: int | None = None,
        map_crop_meters: float | None = None,
        map_output_hw: tuple[int, int] | None = None,
        sample_interval_ms: float | None = None,
    ) -> None:
        scene_mode = scene is not None
        if map_scale < 1:
            raise ValueError(f"map_scale must be >= 1, got {map_scale}")
        if scene_mode and (metadata_files is not None or sequences_root is not None):
            raise ValueError("scene mode cannot be combined with metadata_files/sequences_root mode.")
        if not scene_mode and metadata_files is None and sequences_root is None:
            raise ValueError("Provide one of: scene or metadata_files or sequences_root.")
        if metadata_files is not None and sequences_root is not None:
            raise ValueError("Use metadata_files or sequences_root, not both.")
        if resize_hw is not None and (resize_hw[0] <= 0 or resize_hw[1] <= 0):
            raise ValueError(f"resize_hw must be positive (H, W), got {resize_hw}")
        if resize_mode not in {"stretch", "letterbox", "center_crop"}:
            raise ValueError(
                f"resize_mode must be 'stretch', 'letterbox', or 'center_crop', got {resize_mode}"
            )
        if scene_mode and resize_hw is None:
            raise ValueError("scene mode requires resize_hw so image/map shapes are deterministic.")
        if sat_compat and not scene_mode:
            raise ValueError("sat_compat requires scene mode because both image and map tensors are needed.")
        if map_crop_meters is not None and map_crop_meters <= 0:
            raise ValueError(f"map_crop_meters must be > 0, got {map_crop_meters}")
        if map_output_hw is not None and (map_output_hw[0] <= 0 or map_output_hw[1] <= 0):
            raise ValueError(f"map_output_hw must be positive (H, W), got {map_output_hw}")
        if sample_interval_ms is not None and sample_interval_ms <= 0:
            raise ValueError(f"sample_interval_ms must be > 0, got {sample_interval_ms}")
        if roma_patch_size is not None:
            if roma_patch_size <= 0:
                raise ValueError(f"roma_patch_size must be > 0, got {roma_patch_size}")
            if resize_hw is None:
                raise ValueError("roma_patch_size validation requires resize_hw to be set.")
            h, w = resize_hw
            p = int(roma_patch_size)
            map_h = h * int(map_scale)
            map_w = w * int(map_scale)
            if (h % p) != 0 or (w % p) != 0:
                raise ValueError(f"resize_hw={resize_hw} must be divisible by roma_patch_size={p}")
            if (map_h % p) != 0 or (map_w % p) != 0:
                raise ValueError(
                    f"(resize_hw * map_scale)=({map_h}, {map_w}) must be divisible by roma_patch_size={p}"
                )

        self.image_transform = image_transform
        self.gps_transform = gps_transform
        self.resize_hw = resize_hw
        self.resize_mode = resize_mode
        self.sat_compat = bool(sat_compat)
        self.roma_patch_size = int(roma_patch_size) if roma_patch_size is not None else None
        self.map_scale = int(map_scale)
        self.map_crop_meters = float(map_crop_meters) if map_crop_meters is not None else None
        self.map_output_hw = tuple(map_output_hw) if map_output_hw is not None else None
        self.sample_interval_ms = float(sample_interval_ms) if sample_interval_ms is not None else None
        self._scene = scene
        self._map_georef: dict[str, float | str] | None = None
        self._map_vips: Any | None = None
        self._map_transformer: Transformer | None = None
        self._pyvips: Any | None = None
        if scene_mode:
            self._pyvips = _vips_import()
            self._samples = self._build_scene_index(
                scene=scene,  # type: ignore[arg-type]
                images_root=Path(images_root or "test_images"),
                gps_root=Path(gps_root or "test_gps"),
                references_root=Path(references_root or "test_references"),
            )
        else:
            self._samples = self._build_index(
                metadata_files=metadata_files,
                sequences_root=sequences_root,
                metadata_name=metadata_name,
            )
        if self.sample_interval_ms is not None:
            self._samples = self._sample_by_time(self._samples, interval_ms=self.sample_interval_ms)
        if not self._samples:
            raise ValueError("No samples found for provided metadata input.")

    def _sample_by_time(self, samples: list[_SampleRef], interval_ms: float) -> list[_SampleRef]:
        interval_s = float(interval_ms) / 1000.0
        grouped: dict[tuple[str, str], list[_SampleRef]] = {}
        for sample in samples:
            key = (sample.sequence, sample.scene)
            grouped.setdefault(key, []).append(sample)

        out: list[_SampleRef] = []
        for _, group_samples in grouped.items():
            group_samples = sorted(group_samples, key=lambda s: float(s.gps_time))
            if len(group_samples) <= 1:
                out.extend(group_samples)
                continue

            times = [float(s.gps_time) for s in group_samples]
            picked = [0]
            current_idx = 0
            while True:
                target_t = times[current_idx] + interval_s
                right_idx = bisect_left(times, target_t, lo=current_idx + 1)
                if right_idx >= len(group_samples):
                    break
                left_idx = right_idx - 1
                if left_idx <= current_idx:
                    chosen_idx = right_idx
                else:
                    left_dt = abs(times[left_idx] - target_t)
                    right_dt = abs(times[right_idx] - target_t)
                    chosen_idx = left_idx if left_dt <= right_dt else right_idx
                    if chosen_idx <= current_idx:
                        chosen_idx = right_idx
                if chosen_idx <= current_idx:
                    break
                picked.append(chosen_idx)
                current_idx = chosen_idx

            out.extend(group_samples[i] for i in picked)
        return out

    def _build_scene_index(
        self,
        scene: str,
        images_root: Path,
        gps_root: Path,
        references_root: Path,
    ) -> list[_SampleRef]:
        scene_dir = images_root / scene
        gps_path = gps_root / f"{scene}.txt"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene images directory does not exist: {scene_dir}")
        if not gps_path.exists():
            raise FileNotFoundError(f"GPS txt does not exist: {gps_path}")

        map_path = _resolve_reference_path_flexible(scene=scene, references_root=references_root)
        map_georef = _load_geotiff_tags(map_path)
        self._map_georef = map_georef
        self._map_transformer = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_user_input(str(map_georef["crs"])), always_xy=True
        )
        assert self._pyvips is not None
        self._map_vips = _vips_to_rgb(
            self._pyvips.Image.new_from_file(str(map_path), access="random"),
            pyvips=self._pyvips,
        )

        gps_rows = _parse_gps_txt(gps_path)
        image_paths = sorted(
            p
            for p in scene_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        )
        if not image_paths:
            raise ValueError(f"No image files found in {scene_dir}")

        samples: list[_SampleRef] = []
        for image_path in image_paths:
            stem = image_path.stem
            if not stem.isdigit():
                continue
            image_index = int(stem)
            if image_index < 0 or image_index >= len(gps_rows):
                continue
            gps_time, lat, lon, alt = gps_rows[image_index]
            samples.append(
                _SampleRef(
                    sequence=scene,
                    scene=scene,
                    image_name=image_path.name,
                    image_path=image_path,
                    gps_time=float(gps_time),
                    latitude=float(lat),
                    longitude=float(lon),
                    altitude=float(alt),
                    map_path=map_path,
                )
            )

        if not samples:
            raise ValueError(
                f"No compatible image/gps pairs for scene '{scene}'. "
                "Expected image names like 000123.jpg and GPS rows indexed by that id."
            )
        return samples

    def _build_index(
        self,
        metadata_files: list[str | Path] | None,
        sequences_root: str | Path | None,
        metadata_name: str,
    ) -> list[_SampleRef]:
        meta_paths: list[Path]
        if metadata_files is not None:
            meta_paths = [Path(p) for p in metadata_files]
        else:
            root = Path(sequences_root)  # type: ignore[arg-type]
            if not root.exists():
                raise FileNotFoundError(f"sequences_root does not exist: {root}")
            meta_paths = sorted(root.glob(f"*/{metadata_name}"))

        samples: list[_SampleRef] = []
        for meta_path in meta_paths:
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            sequence = str(payload.get("sequence", meta_path.parent.name))
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Metadata {meta_path} has invalid or missing 'items' list.")

            for item in items:
                image_name = str(item["image"])
                image_path = meta_path.parent / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image referenced in metadata: {image_path}")

                samples.append(
                    _SampleRef(
                        sequence=sequence,
                        scene=sequence,
                        image_name=image_name,
                        image_path=image_path,
                        gps_time=float(item["gps_time"]),
                        latitude=float(item["latitude"]),
                        longitude=float(item["longitude"]),
                        altitude=float(item["altitude"]),
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _resize_letterbox(
        self, image: Image.Image, target_hw: tuple[int, int]
    ) -> tuple[Image.Image, tuple[float, float], tuple[int, int, int, int]]:
        target_h, target_w = target_hw
        src_w, src_h = image.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
        canvas.paste(resized, (left, top))
        return canvas, (float(scale), float(scale)), (left, top, right, bottom)

    def _resize_center_crop(
        self, image: Image.Image, target_hw: tuple[int, int]
    ) -> tuple[Image.Image, tuple[float, float], tuple[int, int, int, int]]:
        target_h, target_w = target_hw
        src_w, src_h = image.size
        scale = max(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h
        cropped = resized.crop((left, top, right, bottom))
        return cropped, (float(scale), float(scale)), (0, 0, 0, 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._samples[idx]
        if self._scene is None:
            image = Image.open(row.image_path).convert("RGB")
            orig_w, orig_h = image.size
            resize_scale_xy = (1.0, 1.0)
            pad_ltrb = (0, 0, 0, 0)
            if self.resize_hw is not None:
                if self.resize_mode == "stretch":
                    image = image.resize((self.resize_hw[1], self.resize_hw[0]), Image.BILINEAR)
                    resize_scale_xy = (
                        self.resize_hw[1] / float(orig_w),
                        self.resize_hw[0] / float(orig_h),
                    )
                elif self.resize_mode == "center_crop":
                    image, resize_scale_xy, pad_ltrb = self._resize_center_crop(image, self.resize_hw)
                else:
                    image, resize_scale_xy, pad_ltrb = self._resize_letterbox(image, self.resize_hw)

            image_arr = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous()

            gps_tensor = torch.tensor(
                [row.latitude, row.longitude, row.altitude],
                dtype=torch.float32,
            )
            gps_time_tensor = torch.tensor(row.gps_time, dtype=torch.float32)

            if self.image_transform is not None:
                image_tensor = self.image_transform(image_tensor)
            if self.gps_transform is not None:
                gps_tensor = self.gps_transform(gps_tensor)

            return {
                "image": image_tensor,
                "gps": gps_tensor,
                "gps_time": gps_time_tensor,
                "sequence": row.sequence,
                "scene": row.scene,
                "image_name": row.image_name,
                "image_path": str(row.image_path),
                "index": idx,
                "orig_size_wh": (orig_w, orig_h),
                "processed_size_wh": image.size,
                "resize_mode": self.resize_mode,
                "resize_scale_xy": resize_scale_xy,
                "pad_ltrb": pad_ltrb,
            }

        assert self._pyvips is not None
        assert self._map_georef is not None
        assert self._map_vips is not None
        assert self._map_transformer is not None

        image_vips = _vips_to_rgb(
            self._pyvips.Image.new_from_file(str(row.image_path), access="sequential"),
            pyvips=self._pyvips,
        )
        orig_w, orig_h = image_vips.width, image_vips.height
        resize_scale_xy = (1.0, 1.0)
        pad_ltrb = (0, 0, 0, 0)
        image_vips, resize_scale_xy, pad_ltrb = _resize_with_vips(
            image=image_vips,
            resize_hw=self.resize_hw,
            resize_mode=self.resize_mode,
            pyvips=self._pyvips,
        )
        image_arr = _vips_to_numpy_hwc(image_vips).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous()

        gps_tensor = torch.tensor(
            [row.latitude, row.longitude, row.altitude],
            dtype=torch.float32,
        )
        gps_time_tensor = torch.tensor(row.gps_time, dtype=torch.float32)

        out_h, out_w = image_arr.shape[0], image_arr.shape[1]
        if self.map_crop_meters is not None:
            scale_x = float(self._map_georef["scale_x"])
            scale_y = float(self._map_georef["scale_y"])
            crop_w_px = max(1, int(round(self.map_crop_meters / scale_x)))
            crop_h_px = max(1, int(round(self.map_crop_meters / scale_y)))
            if self.map_output_hw is not None:
                map_h, map_w = self.map_output_hw
            else:
                map_h, map_w = crop_h_px, crop_w_px
        else:
            crop_h_px = out_h * self.map_scale
            crop_w_px = out_w * self.map_scale
            map_h = crop_h_px
            map_w = crop_w_px

        px, py = _project_to_tiff_pixel(
            lon=row.longitude,
            lat=row.latitude,
            scale_x=float(self._map_georef["scale_x"]),
            scale_y=float(self._map_georef["scale_y"]),
            tie_i=float(self._map_georef["tie_i"]),
            tie_j=float(self._map_georef["tie_j"]),
            tie_x=float(self._map_georef["tie_x"]),
            tie_y=float(self._map_georef["tie_y"]),
            transformer=self._map_transformer,
        )
        map_patch_vips, map_left_px, map_top_px = _extract_patch_with_padding(
            image=self._map_vips,
            center_x=px,
            center_y=py,
            crop_w=crop_w_px,
            crop_h=crop_h_px,
            pyvips=self._pyvips,
        )
        if map_patch_vips.width != map_w or map_patch_vips.height != map_h:
            sx = map_w / float(map_patch_vips.width)
            sy = map_h / float(map_patch_vips.height)
            map_patch_vips = map_patch_vips.resize(sx, vscale=sy, kernel="lanczos3")
        map_arr = _vips_to_numpy_hwc(map_patch_vips).astype(np.float32) / 255.0
        map_tensor = torch.from_numpy(map_arr).permute(2, 0, 1).contiguous()

        image_axes_x = torch.arange(out_w, dtype=torch.float32)
        image_axes_y = torch.arange(out_h, dtype=torch.float32)
        scale_x = float(self._map_georef["scale_x"])
        scale_y = float(self._map_georef["scale_y"])
        tie_i = float(self._map_georef["tie_i"])
        tie_j = float(self._map_georef["tie_j"])
        tie_x = float(self._map_georef["tie_x"])
        tie_y = float(self._map_georef["tie_y"])
        map_x0 = tie_x + (float(map_left_px) - tie_i) * scale_x
        map_y0 = tie_y - (float(map_top_px) - tie_j) * scale_y
        map_x1 = map_x0 + (crop_w_px - 1) * scale_x
        map_y1 = map_y0 - (crop_h_px - 1) * scale_y
        map_axes_x = torch.tensor(np.linspace(map_x0, map_x1, map_w, dtype=np.float32), dtype=torch.float32)
        map_axes_y = torch.tensor(np.linspace(map_y0, map_y1, map_h, dtype=np.float32), dtype=torch.float32)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image_tensor)
        if self.gps_transform is not None:
            gps_tensor = self.gps_transform(gps_tensor)

        out = {
            "image": image_tensor,
            "map": map_tensor,
            "gps": gps_tensor,
            "gps_time": gps_time_tensor,
            "sequence": row.sequence,
            "scene": row.scene,
            "image_name": row.image_name,
            "image_path": str(row.image_path),
            "map_path": str(row.map_path) if row.map_path is not None else None,
            "index": idx,
            "orig_size_wh": (orig_w, orig_h),
            "processed_size_wh": (out_w, out_h),
            "map_size_wh": (map_w, map_h),
            "resize_mode": self.resize_mode,
            "resize_scale_xy": resize_scale_xy,
            "pad_ltrb": pad_ltrb,
            "image_axes_x": image_axes_x,
            "image_axes_y": image_axes_y,
            "map_axes_x": map_axes_x,
            "map_axes_y": map_axes_y,
            "map_crop_origin_xy": (map_left_px, map_top_px),
            "map_crop_size_wh": (crop_w_px, crop_h_px),
        }
        if self.sat_compat:
            out["im_A"] = image_tensor
            out["im_B"] = map_tensor
            out["im_A_label"] = row.image_name
            out["im_B_label"] = Path(row.map_path).name if row.map_path is not None else "unknown_map"
            # In this dataset variant we do not have synthetic GT warp labels.
            out["gt_warp"] = None
            out["mask"] = None
            out["homography"] = None
            out["map_to_img_ratio"] = int(self.map_scale)
            out["has_gt_warp"] = False
            out["roma_patch_size"] = self.roma_patch_size
        return out


def test_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for test/inference mode preserving metadata fields."""
    if not batch:
        raise ValueError("Empty batch in test_collate_fn.")
    out: dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "gps": torch.stack([b["gps"] for b in batch], dim=0),
        "gps_time": torch.stack([b["gps_time"] for b in batch], dim=0),
        "sequence": [b["sequence"] for b in batch],
        "scene": [b.get("scene", b["sequence"]) for b in batch],
        "image_name": [b["image_name"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "orig_size_wh": [b["orig_size_wh"] for b in batch],
        "processed_size_wh": [b["processed_size_wh"] for b in batch],
        "resize_mode": [b["resize_mode"] for b in batch],
        "resize_scale_xy": torch.tensor([b["resize_scale_xy"] for b in batch], dtype=torch.float32),
        "pad_ltrb": [b["pad_ltrb"] for b in batch],
    }
    if all("map" in b for b in batch):
        out["map"] = torch.stack([b["map"] for b in batch], dim=0)
        out["map_path"] = [b["map_path"] for b in batch]
        out["map_size_wh"] = [b["map_size_wh"] for b in batch]
        out["image_axes_x"] = torch.stack([b["image_axes_x"] for b in batch], dim=0)
        out["image_axes_y"] = torch.stack([b["image_axes_y"] for b in batch], dim=0)
        out["map_axes_x"] = torch.stack([b["map_axes_x"] for b in batch], dim=0)
        out["map_axes_y"] = torch.stack([b["map_axes_y"] for b in batch], dim=0)
        if all("map_crop_origin_xy" in b for b in batch):
            out["map_crop_origin_xy"] = [b["map_crop_origin_xy"] for b in batch]
            out["map_crop_size_wh"] = [b["map_crop_size_wh"] for b in batch]
    if all("im_A" in b and "im_B" in b for b in batch):
        out["im_A"] = torch.stack([b["im_A"] for b in batch], dim=0)
        out["im_B"] = torch.stack([b["im_B"] for b in batch], dim=0)
        out["im_A_label"] = [b["im_A_label"] for b in batch]
        out["im_B_label"] = [b["im_B_label"] for b in batch]
        out["map_to_img_ratio"] = torch.tensor([b["map_to_img_ratio"] for b in batch], dtype=torch.int64)
        out["has_gt_warp"] = torch.tensor([bool(b.get("has_gt_warp", False)) for b in batch], dtype=torch.bool)
        gt_warp_vals = [b.get("gt_warp") for b in batch]
        mask_vals = [b.get("mask") for b in batch]
        homography_vals = [b.get("homography") for b in batch]
        out["gt_warp"] = (
            torch.stack(gt_warp_vals, dim=0)
            if all(torch.is_tensor(v) for v in gt_warp_vals)
            else gt_warp_vals
        )
        out["mask"] = (
            torch.stack(mask_vals, dim=0)
            if all(torch.is_tensor(v) for v in mask_vals)
            else mask_vals
        )
        out["homography"] = (
            torch.stack(homography_vals, dim=0)
            if all(torch.is_tensor(v) for v in homography_vals)
            else homography_vals
        )
        if all("roma_patch_size" in b for b in batch):
            out["roma_patch_size"] = [b["roma_patch_size"] for b in batch]
    return out


def create_test_dataloader(
    scene: str | None = None,
    images_root: str | Path | None = None,
    gps_root: str | Path | None = None,
    references_root: str | Path | None = None,
    map_scale: int = 4,
    metadata_files: list[str | Path] | None = None,
    sequences_root: str | Path | None = None,
    metadata_name: str = "gps_metadata.json",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    gps_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    resize_hw: tuple[int, int] | None = None,
    resize_mode: Literal["stretch", "letterbox", "center_crop"] = "stretch",
    sat_compat: bool = False,
    roma_patch_size: int | None = None,
    map_crop_meters: float | None = None,
    map_output_hw: tuple[int, int] | None = None,
    sample_interval_ms: float | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader suitable for inference/test mode (no shuffling).

    sample_interval_ms:
        Optional temporal downsampling interval. If set (e.g. 200.0), keeps one
        frame roughly every N milliseconds by choosing the nearest available
        sample by gps_time and dropping intermediate frames.
    """
    dataset = SyncedGpsImageDataset(
        scene=scene,
        images_root=images_root,
        gps_root=gps_root,
        references_root=references_root,
        map_scale=map_scale,
        metadata_files=metadata_files,
        sequences_root=sequences_root,
        metadata_name=metadata_name,
        image_transform=image_transform,
        gps_transform=gps_transform,
        resize_hw=resize_hw,
        resize_mode=resize_mode,
        sat_compat=sat_compat,
        roma_patch_size=roma_patch_size,
        map_crop_meters=map_crop_meters,
        map_output_hw=map_output_hw,
        sample_interval_ms=sample_interval_ms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=test_collate_fn,
    )
