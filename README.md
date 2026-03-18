# sat_test

## Virtual Environment Setup

### Option A: `uv` (recommended)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create venv and install dependencies:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Activate manually:

```bash
source .venv/bin/activate
```

Or auto-activate with `direnv` (project already includes `.envrc`):

```bash
direnv allow
```

Directory defaults are also set in `.envrc`:
`VIDEOS_INPUT`, `IMAGES_OUTPUT`, `GPS_INPUT`, `TRAJ_OUTPUT`, `TRAJ_STATS_OUTPUT`.

### Option B: standard Python `venv`

```bash
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

### Using `just`

```bash
just setup-uv
just all
```

Pass optional flags to frame extraction:

```bash
just frames -- --jobs 4 --overwrite
```

### Manual: where to put `jobs` and `overwrite`

`just` passes script flags only after `--`.

Examples:

```bash
# frames only (parallel + overwrite)
just frames -- --jobs 4 --overwrite

# frames only (parallel, no overwrite)
just frames -- --jobs 4

# full pipeline (parallel frames + overwrite in step 1)
just all -- --jobs 4 --overwrite-frames
```

Extract frames from videos:

```bash
python videos_to_frames.py --input test_videos --output test_images
```

Run extraction across multiple videos in parallel:

```bash
python videos_to_frames.py --input test_videos --output test_images --jobs 4
```

By default this also syncs GPS from `test_gps/<id>.txt` and writes:
`test_images/<id>/gps_metadata.json`.

Extract frames and sync GPS poses to image metadata:

```bash
python videos_to_frames.py \
  --input test_videos \
  --output test_images \
  --gps-input test_gps \
  --metadata-name gps_metadata.json
```

This writes `gps_metadata.json` inside each sequence directory, for example:
`test_images/0001/gps_metadata.json`.

Disable GPS sync if needed:

```bash
python videos_to_frames.py --no-gps-sync
```

Generate trajectory plots and per-track stats `.txt` files:

```bash
.venv/bin/python vis_gps_on_osm.py \
  --input test_gps \
  --output test_trajectories \
  --stats-output test_trajectories_stats
```

Run full pipeline (frames + GPS metadata + trajectory plots + stats):

```bash
.venv/bin/python run_all.py --overwrite-frames
```

Using `direnv` + `just`:

```bash
direnv allow
just setup-uv
just all
```
