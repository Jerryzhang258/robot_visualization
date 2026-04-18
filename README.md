# Robot Visualization

Visualization tool for VR dual-arm teleoperation data.

Two data formats are supported:
- **LeRobot v2.1 parquet** — preferred path via the Rerun viewer (GPU-accelerated, interactive timeline scrubbing, multi-episode browsing)
- **Zarr (`.zarr.zip`)** — legacy PyRender/OpenCV pipeline (`viz_3d_enhanced.py`)

## Install

```bash
git clone https://github.com/Jerryzhang258/robot_visualization.git
cd robot_visualization
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `rerun-sdk>=0.16`, `pyrender`, `trimesh`, `scipy`, `opencv-python`, `pyarrow`.

## Usage

### LeRobot dataset (recommended)

```bash
# All episodes
python src/viz_rerun.py path/to/lerobot_dataset

# Single episode
python src/viz_rerun.py path/to/dataset -e 3

# Multiple non-contiguous
python src/viz_rerun.py path/to/dataset -e 0 3 7

# Range (inclusive)
python src/viz_rerun.py path/to/dataset -e 0-10

# Mixed
python src/viz_rerun.py path/to/dataset -e 0-5 8 12-15

# Save to .rrd (no viewer)
python src/viz_rerun.py path/to/dataset --save out.rrd
rerun out.rrd
```

Once the Rerun viewer window opens, drag the timeline to scrub freely across the entire loaded range.

Other entry points:

- `src/viz_raw_0407.py` — LeRobot **v3** chunked parquet layout (`data/chunk-*/file-*.parquet`)
- `src/visualize_lerobot_data.py` — load via HuggingFace `datasets` (works around a known bug in the native LeRobot loader), supports integrating pose from action deltas and exporting MP4

### Zarr (legacy pipeline)

```bash
# Interactive
python src/viz_3d_enhanced.py data/your_data.zarr.zip

# Record video
python src/viz_3d_enhanced.py data/your_data.zarr.zip -r --record_episode 1 --output_video demo.mp4
```

Controls (zarr pipeline only):

- `A / D` — previous / next frame
- `W / S` — switch episode
- `P` — toggle autoplay
- `1-5` — playback speed (0.25x / 0.5x / 1x / 2x / 5x)
- `Q` — quit

## Data formats

### LeRobot v2.1

Directory layout:

```
dataset/
├── meta/
│   ├── info.json
│   └── episodes.jsonl
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        └── ...
```

Columns per parquet row:

| Column | Description |
|--------|-------------|
| `observation.state` | 20-D (`0-5` left arm pos + rotvec, `6` left gripper width, `7-12` right arm pos + rotvec, `13` right gripper width, `14-19` right relative to left) |
| `observation.images.camera0 / 1` | wrist-mounted fisheye cameras (224×224 RGB) |
| `observation.images.tactile_left_0 / right_0 / left_1 / right_1` | four tactile image streams |
| `actions` | 20-D action vector |

### Zarr

`.zarr.zip` archive. Keys:

- `robot0/1_eef_pos` — end-effector position
- `robot0/1_gripper_width` — gripper width
- `robot0/1_visual` — RGB camera images
- `robot0/1_left_tactile / right_tactile` — tactile images

## 3D assets

STL files in `src/meshes/`:

- `finger.STL` — gripper finger
- `夹爪.STL` — gripper body (mirrored for L/R)
- `Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl` / `...Right.stl` — VR controllers
