# sat_test

Drone video frame extraction, GPS synchronization, and georeferenced visualization tools.

## Setup

### Using `uv` (recommended)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Or auto-activate with `direnv` (project includes `.envrc`):

```bash
direnv allow
```

For scene mode and GeoTIFF visualization (requires pyvips):

```bash
uv pip install --python .venv/bin/python pyvips pyvips-binary
```

`ffmpeg` must be in PATH for frame extraction.

### Using `just`

```bash
just setup-uv        # create venv + install deps
just setup-project    # install as editable package (needed for sat_test_up_count_track imports)
```

---

## Scripts Overview

### 1. Frame Extraction Pipeline

These scripts form the core data pipeline: video → frames → GPS metadata → trajectory plots.

| Script | Purpose |
|---|---|
| `videos_to_frames.py` | Extract frames from `.mp4` files with ffmpeg, sync GPS metadata |
| `vis_gps_on_osm.py` | Plot GPS trajectories on OpenStreetMap tiles |
| `run_all.py` | Run both scripts above in sequence |

```bash
# Extract frames (parallel)
python videos_to_frames.py --input test_videos --output test_images --jobs 4

# Overwrite existing frames
python videos_to_frames.py --input test_videos --output test_images --jobs 4 --overwrite

# Skip GPS sync
python videos_to_frames.py --input test_videos --output test_images --no-gps-sync

# Plot GPS trajectories
python vis_gps_on_osm.py --input test_gps --output test_trajectories --stats-output test_trajectories_stats

# Run full pipeline
python run_all.py --overwrite-frames --jobs 4
```

Or with `just`:

```bash
just frames -- --jobs 4 --overwrite
just traj
just all -- --jobs 4 --overwrite-frames
just clean-cache
```

**Data flow:**

```
test_videos/<id>.mp4  ──► test_images/<id>/*.jpg + gps_metadata.json
test_gps/<id>.txt     ──► test_trajectories/<id>.png + test_trajectories_stats/<id>.txt
```

GPS sync is index-based: frame N maps to GPS row N in `test_gps/<id>.txt`.
GPS file format: CSV with columns `time,lat,lon,alt` (one row per frame).

### 2. GeoTIFF Visualization & Registration

These scripts overlay drone frames onto georeferenced TIFF satellite maps. They all require a reference TIFF and GPS metadata from step 1.

| Script | Purpose |
|---|---|
| `vis_first_frame_on_tiff.py` | Overlay a single frame on a TIFF map (batch) |
| `vis_first_frame_on_tiff_interactive.py` | Same, but with interactive sliders for rotate/translate/scale |
| `stitch_frames_on_tiff_map.py` | Stitch sampled frames along a GPS trajectory on a TIFF (batch) |
| `stitch_frames_on_tiff_map_interactive.py` | Same, with interactive alignment sliders |
| `p2p_register_drone_to_tiff.py` | Interactive point-to-point registration (pick matching points on drone image and TIFF) |

```bash
# Overlay first frame on TIFF
python vis_first_frame_on_tiff.py \
  --tiff test_references/0001_year_2024_crop.tiff \
  --metadata test_images/0001/gps_metadata.json \
  --images-dir test_images/0001

# Stitch trajectory frames on TIFF
python stitch_frames_on_tiff_map.py \
  --tiff test_references/0001_year_2024_crop.tiff \
  --metadata test_images/0001/gps_metadata.json \
  --images-dir test_images/0001

# Interactive versions open a matplotlib window with sliders
python vis_first_frame_on_tiff_interactive.py --tiff ... --metadata ...
python stitch_frames_on_tiff_map_interactive.py --tiff ... --metadata ...
python p2p_register_drone_to_tiff.py --tiff ... --metadata ...
```

**Import dependencies:** `vis_first_frame_on_tiff.py` is also a shared utility module — `stitch_frames_on_tiff_map.py` and `p2p_register_drone_to_tiff.py` import GeoTIFF helpers from it.

### 3. PyTorch Dataloader (`sat_test_up_count_track`)

A PyTorch package for loading synchronized GPS + image data. Install as editable package first:

```bash
uv pip install --python .venv/bin/python -e .
```

**Basic usage:**

```python
from sat_test_up_count_track import create_test_dataloader

loader = create_test_dataloader(
    sequences_root="test_images",
    batch_size=8,
    num_workers=2,
    resize_hw=(256, 256),
    resize_mode="letterbox",
)

for batch in loader:
    images = batch["image"]       # [B, 3, H, W], float32 in [0, 1]
    gps = batch["gps"]            # [B, 3] -> [lat, lon, alt]
    gps_time = batch["gps_time"]  # [B]
    seq = batch["sequence"]       # list[str]
    names = batch["image_name"]   # list[str]
```

**Scene mode** (loads drone image + satellite map patch side by side):

```python
loader = create_test_dataloader(
    scene="0007",
    images_root="test_images",
    gps_root="test_gps",
    references_root="test_references",
    resize_hw=(224, 224),
    map_scale=4,
    resize_mode="letterbox",
    batch_size=1,
)

batch = next(iter(loader))
batch["image"]       # [B, 3, H, W] — drone frame
batch["map"]         # [B, 3, map_H, map_W] — satellite map patch
batch["map_axes_x"]  # projected X coordinates in GeoTIFF CRS
batch["map_axes_y"]  # projected Y coordinates in GeoTIFF CRS
```

Scene mode requires pyvips and reference TIFFs in `test_references/` named `<scene>_year_*_crop.tiff`.

**Example scripts:**

| Script | What it shows |
|---|---|
| `example_loader_viz.py` | Animated frame viewer with configurable resolution |
| `example_loader_low_viz.py` | Same but preset to 224x224 letterbox |
| `example_loader_scene_imshow.py` | Scene mode: drone image + map patch side by side |

---

## Environment Variables

Set in `.envrc` and mirrored in `justfile`:

| Variable | Default |
|---|---|
| `VIDEOS_INPUT` | `test_videos` |
| `IMAGES_OUTPUT` | `test_images` |
| `GPS_INPUT` | `test_gps` |
| `TRAJ_OUTPUT` | `test_trajectories` |
| `TRAJ_STATS_OUTPUT` | `test_trajectories_stats` |
