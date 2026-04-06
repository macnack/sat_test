# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Using `uv` (recommended):
```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Or install as an editable package (needed to use `sat_test_up_count_track` as a package):
```bash
uv pip install --python .venv/bin/python -e .
```

For scene mode support (pyvips):
```bash
uv pip install --python .venv/bin/python pyvips pyvips-binary
```

`ffmpeg` must be available in PATH for frame extraction.

`direnv` can auto-activate the venv and set default paths via `.envrc`.

## Commands

```bash
# Extract frames from videos + sync GPS metadata
python videos_to_frames.py --input test_videos --output test_images --jobs 4

# Visualize GPS trajectories on OSM tiles
python vis_gps_on_osm.py --input test_gps --output test_trajectories --stats-output test_trajectories_stats

# Run full pipeline (frames → GPS sync → trajectory plots)
python run_all.py --overwrite-frames

# Using just (reads env vars from .envrc / env)
just frames -- --jobs 4 --overwrite
just traj
just all -- --jobs 4 --overwrite-frames
just clean-cache
```

There are no automated tests in this repo.

## Architecture

### Data Flow

```
test_videos/<id>.mp4
    └─[videos_to_frames.py + ffmpeg]→ test_images/<id>/*.jpg
                                            + gps_metadata.json
test_gps/<id>.txt
    └─[vis_gps_on_osm.py]→ test_trajectories/<id>.png
                          + test_trajectories_stats/<id>.txt
```

GPS sync is **index-based**: frame `N` corresponds to GPS row `N` in `<id>.txt`. The GPS file format is CSV with columns `time,lat,lon,alt` (one row per frame).

### `gps_metadata.json` Schema

Written by `videos_to_frames.py` into each `test_images/<id>/` folder. Contains a top-level `items` list where each entry has `image`, `image_index`, `gps_index`, `gps_time`, `latitude`, `longitude`, `altitude`. This file is the contract between the pipeline and the dataloader.

### PyTorch Package (`sat_test_up_count_track`)

`SyncedGpsImageDataset` has two modes:

- **Standard mode** (`sequences_root` or `metadata_files`): reads `gps_metadata.json` files, loads images with PIL. No pyvips needed.
- **Scene mode** (`scene=`): additionally crops a GeoTIFF map patch centered on each GPS position. Requires pyvips. Reference TIFFs must be in `test_references/` and named `<scene>_year_*_crop.tiff`.

`create_test_dataloader()` is the main entry point — it wraps `SyncedGpsImageDataset` with `DataLoader` (no shuffle, uses `test_collate_fn`).

Batch keys:
- Always: `image` [B,3,H,W], `gps` [B,3], `gps_time` [B], `sequence`, `image_name`
- Scene mode only: `map` [B,3,map_H,map_W], `map_axes_x`, `map_axes_y` (projected coordinates in the GeoTIFF CRS)

GeoTIFF parsing reads TIFF tags directly via PIL (no gdal/rasterio): pixel scale (tag 33550), tiepoint (33922), and GeoKey directory (34735) to extract the EPSG code. Coordinates are re-projected using `pyproj`.

### Top-level Scripts

| Script | Purpose |
|---|---|
| `videos_to_frames.py` | ffmpeg frame extraction + GPS metadata sync |
| `vis_gps_on_osm.py` | GPS trajectory plots on OSM tile background |
| `run_all.py` | Orchestrates the two scripts in sequence |
| `vis_first_frame_on_tiff.py` | Visualize first frame projected onto a GeoTIFF |
| `vis_first_frame_on_tiff_interactive.py` | Interactive rotate/translate a frame over a GeoTIFF |
| `stitch_frames_on_tiff_map.py` | Stitch multiple frames onto a GeoTIFF map |
| `stitch_frames_on_tiff_map_interactive.py` | Interactive manual alignment for stitched overlays |
| `p2p_register_drone_to_tiff.py` | Interactive point-to-point registration of drone image to TIFF |

`vis_first_frame_on_tiff.py` is the shared utility module — `stitch_frames_on_tiff_map.py` and `p2p_register_drone_to_tiff.py` import helpers from it (`_load_geotiff`, `_project_to_tiff_pixel`, `_resize_with_pillow`, etc.). Editing those shared functions affects all three scripts.

### Environment Variables / Defaults

Configured via `.envrc` and mirrored in `justfile`:

| Variable | Default |
|---|---|
| `VIDEOS_INPUT` | `test_videos` |
| `IMAGES_OUTPUT` | `test_images` |
| `GPS_INPUT` | `test_gps` |
| `TRAJ_OUTPUT` | `test_trajectories` |
| `TRAJ_STATS_OUTPUT` | `test_trajectories_stats` |
