set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

videos_input := env_var_or_default("VIDEOS_INPUT", "test_videos")
images_output := env_var_or_default("IMAGES_OUTPUT", "test_images")
gps_input := env_var_or_default("GPS_INPUT", "test_gps")
traj_output := env_var_or_default("TRAJ_OUTPUT", "test_trajectories")
traj_stats_output := env_var_or_default("TRAJ_STATS_OUTPUT", "test_trajectories_stats")

default:
  @just --list

setup:
  uv venv --clear .venv
  uv pip install --python .venv/bin/python -r requirements.txt

setup-project:
  uv venv --clear .venv
  uv pip install --python .venv/bin/python -e .

setup-uv:
  @just setup

setup-uv-project:
  @just setup-project

frames *args:
  python videos_to_frames.py --input {{videos_input}} --output {{images_output}} --gps-input {{gps_input}} {{args}}

traj *args:
  python vis_gps_on_osm.py --input {{gps_input}} --output {{traj_output}} --stats-output {{traj_stats_output}} {{args}}

all *args:
  python run_all.py --videos-input {{videos_input}} --images-output {{images_output}} --gps-input {{gps_input}} --traj-output {{traj_output}} --stats-output {{traj_stats_output}} {{args}}

clean-cache:
  rm -rf {{traj_output}}/.tile_cache
