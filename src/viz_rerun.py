#!/usr/bin/env python3
"""
Rerun-based robot teleoperation data visualization (LeRobot v2.1 format).

Replaces the PyRender/OpenCV pipeline with Rerun's interactive viewer.
Features:
  - 3D world view with EEF poses, trajectories, gripper and controller meshes
  - Camera and tactile image panels
  - Timeseries plots for EEF position and gripper width
  - Interactive timeline scrubbing across all episodes

Usage (run from any directory):
    python /path/to/src/viz_rerun.py /path/to/lerobot_dataset
    python /path/to/src/viz_rerun.py /path/to/dataset --episode 0
    python /path/to/src/viz_rerun.py /path/to/dataset --save output.rrd
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr
import rerun.blueprint as rrb

# ── project imports ──────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

# ── constants ─────────────────────────────────────────────────────────────────
ROBOT_IDS = [0, 1]
ROBOT_COLORS = {0: [220, 60, 60], 1: [60, 220, 60]}  # red / green


# ── mesh loading ──────────────────────────────────────────────────────────────
def _load_trimesh(path, scale=1.0, extra_transform=None):
    """Load an STL file via trimesh, returning vertex/face/normal arrays."""
    import trimesh
    mesh = trimesh.load(path)
    if scale != 1.0:
        mesh.apply_scale(scale)
    if extra_transform is not None:
        mesh.apply_transform(extra_transform)
    return {
        "vertices": mesh.vertices.astype(np.float32),
        "faces":    mesh.faces.astype(np.int32),
        "normals":  (mesh.vertex_normals.astype(np.float32)
                     if hasattr(mesh, "vertex_normals") else None),
    }


def load_static_meshes():
    """
    Load gripper and controller STL files once.
    Returns a dict with keys:
      gripper_left, gripper_right, controller_left, controller_right
    """
    meshes = {}

    # ── gripper ───────────────────────────────────────────────────────────────
    gripper_path = os.path.join(_SRC, "meshes", "夹爪.STL")
    if os.path.exists(gripper_path):
        try:
            import trimesh
            base = trimesh.load(gripper_path)
            base.apply_scale(0.001)                                   # mm → m
            center = (base.bounds[0] + base.bounds[1]) / 2
            base.apply_translation(-center)
            rot = np.eye(4)
            rot[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            base.apply_transform(rot)

            # left finger
            meshes["gripper_left"] = {
                "vertices": base.vertices.astype(np.float32),
                "faces":    base.faces.astype(np.int32),
                "normals":  base.vertex_normals.astype(np.float32),
            }

            # right finger  – mirror across Y axis
            right = base.copy()
            mirror = np.eye(4); mirror[1, 1] = -1
            right.apply_transform(mirror)
            meshes["gripper_right"] = {
                "vertices": right.vertices.astype(np.float32),
                "faces":    right.faces.astype(np.int32),
                "normals":  right.vertex_normals.astype(np.float32),
            }
        except Exception as e:
            print(f"Warning: could not load gripper STL: {e}")

    # ── controllers ───────────────────────────────────────────────────────────
    for side, fname in [
        ("left",  "Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl"),
        ("right", "Oculus_Meta_Quest_Touch_Plus_Controller_Right.stl"),
    ]:
        fpath = os.path.join(_SRC, "meshes", fname)
        if os.path.exists(fpath):
            try:
                meshes[f"controller_{side}"] = _load_trimesh(fpath, scale=0.0015)
            except Exception as e:
                print(f"Warning: could not load controller mesh ({side}): {e}")

    return meshes


# ── static scene logging ──────────────────────────────────────────────────────
def log_static_geometry(meshes):
    """Log world frame, axes, and mesh geometries that never change shape."""

    # world coordinate convention: Z-up, right-hand
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for r in ROBOT_IDS:
        # EEF coordinate axes (3 arrows in local EEF frame, inherit EEF transform)
        rr.log(
            f"world/robot{r}/eef/axes",
            rr.Arrows3D(
                vectors=np.eye(3, dtype=np.float32) * 0.05,
                origins=np.zeros((3, 3), dtype=np.float32),
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )

        # Controller mesh – geometry is static; only its parent EEF transform moves
        ctrl_key = f"controller_{'left' if r == 1 else 'right'}"
        if ctrl_key in meshes:
            m = meshes[ctrl_key]
            n = len(m["vertices"])
            ctrl_rot_quat = Rotation.from_euler("y", 90, degrees=True).as_quat()  # xyzw
            rr.log(
                f"world/robot{r}/eef/controller",
                rr.Transform3D(
                    translation=[0, 0, 0.05],
                    quaternion=rr.Quaternion(xyzw=ctrl_rot_quat),
                ),
                static=True,
            )
            rr.log(
                f"world/robot{r}/eef/controller/mesh",
                rr.Mesh3D(
                    vertex_positions=m["vertices"],
                    triangle_indices=m["faces"],
                    vertex_normals=m.get("normals"),
                    vertex_colors=np.full((n, 3), [100, 100, 200], dtype=np.uint8),
                ),
                static=True,
            )

        # Gripper finger meshes – shape is constant, only transforms change per frame
        for side in ("left", "right"):
            key = f"gripper_{side}"
            if key in meshes:
                m = meshes[key]
                n = len(m["vertices"])
                rr.log(
                    f"world/robot{r}/eef/gripper/{side}/mesh",
                    rr.Mesh3D(
                        vertex_positions=m["vertices"],
                        triangle_indices=m["faces"],
                        vertex_normals=m.get("normals"),
                        vertex_colors=np.full((n, 3), [180, 180, 180], dtype=np.uint8),
                    ),
                    static=True,
                )

            # Tactile sensor marker (small disc at gripper tip)
            sensor_color = [0, 220, 0] if side == "left" else [220, 0, 0]
            rr.log(
                f"world/robot{r}/eef/gripper/{side}/sensor",
                rr.Points3D(positions=[[0, 0, 0]], colors=[sensor_color], radii=[0.012]),
                static=True,
            )


# ── LeRobot parquet loader ────────────────────────────────────────────────────
def _pose6_to_mat(pose6):
    """6-dim pose (pos + rotvec) → 4×4 transform matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = Rotation.from_rotvec(pose6[3:6]).as_matrix()
    return T


def _extract_image_bytes(img_dict):
    """抽出 LeRobot parquet image cell 里的原始 JPEG/PNG bytes,不解码。"""
    raw = img_dict.get("bytes") if isinstance(img_dict, dict) else img_dict
    if raw is None:
        return None
    # pyarrow 可能返回 pa.Binary / np.bytes_ 等类型,统一转成 bytes
    if not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    return raw


def _detect_media_type(raw):
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    if raw.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    return None


_HAS_ENCODED_IMAGE = hasattr(rr, "EncodedImage")
_ENCODED_IMAGE_FAILED = False  # 首次失败后永久回退到 cv2 解码路径


def _decode_image_bytes(img_dict):
    """回退路径:老版 rerun 没 EncodedImage 时才用,cv2 解码成 (H,W,3)。"""
    import cv2
    raw = _extract_image_bytes(img_dict)
    if raw is None:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_lerobot_episode(dataset_path, episode_idx):
    """
    Load one episode from a LeRobot v2.1 parquet dataset.

    Returns data dict compatible with log_frame():
      data['robot0']['poses']    (N, 4, 4) np.float32
      data['robot0']['gripper']  (N,)       np.float32
      data['robot0']['visual']   list of HxWx3 uint8
      ...
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    chunk = episode_idx // 1000
    parquet_path = os.path.join(
        dataset_path,
        f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet",
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    # ── vectorized state → (N, 4, 4) 位姿 ───────────────────────────────────
    states = np.stack(df["observation.state"].values).astype(np.float32)
    N = len(states)

    robot0_mats = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    robot0_mats[:, :3, :3] = Rotation.from_rotvec(states[:, 3:6]).as_matrix()
    robot0_mats[:, :3, 3] = states[:, 0:3]

    rel_mats = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    rel_mats[:, :3, :3] = Rotation.from_rotvec(states[:, 17:20]).as_matrix()
    rel_mats[:, :3, 3] = states[:, 14:17]
    robot1_mats = robot0_mats @ np.linalg.inv(rel_mats)

    data = {
        "robot0": {
            "poses":   robot0_mats,
            "gripper": states[:, 6].astype(np.float32),
            "visual": [], "left_tactile": [], "right_tactile": [],
            "left_pc": [], "right_pc": [],
        },
        "robot1": {
            "poses":   robot1_mats,
            "gripper": states[:, 13].astype(np.float32),
            "visual": [], "left_tactile": [], "right_tactile": [],
            "left_pc": [], "right_pc": [],
        },
    }

    # ── 图像:走 EncodedImage 就只抽 bytes;否则并行 cv2 解码 ─────────────
    img_cols = {
        "observation.images.camera0":         ("robot0", "visual"),
        "observation.images.camera1":         ("robot1", "visual"),
        "observation.images.tactile_left_0":  ("robot0", "left_tactile"),
        "observation.images.tactile_right_0": ("robot0", "right_tactile"),
        "observation.images.tactile_left_1":  ("robot1", "left_tactile"),
        "observation.images.tactile_right_1": ("robot1", "right_tactile"),
    }
    available = [(c, k) for c, k in img_cols.items() if c in df.columns]
    if _HAS_ENCODED_IMAGE:
        for col, (robot_key, sensor_key) in available:
            data[robot_key][sensor_key] = [_extract_image_bytes(r) for r in df[col].tolist()]
    else:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for col, (robot_key, sensor_key) in available:
                imgs = list(ex.map(_decode_image_bytes, df[col].tolist()))
                data[robot_key][sensor_key] = [im for im in imgs if im is not None]

    return data


# ── per-episode batch logging ─────────────────────────────────────────────────
def _image_log(cam_path, raw_or_arr):
    """图像 log — 有 EncodedImage 就直接发 bytes,否则回退到 cv2 解码成数组。"""
    global _ENCODED_IMAGE_FAILED
    if raw_or_arr is None:
        return

    if _HAS_ENCODED_IMAGE and not _ENCODED_IMAGE_FAILED \
            and isinstance(raw_or_arr, (bytes, bytearray, memoryview)):
        raw = bytes(raw_or_arr) if not isinstance(raw_or_arr, bytes) else raw_or_arr
        media_type = _detect_media_type(raw)
        try:
            if media_type:
                rr.log(cam_path, rr.EncodedImage(contents=raw, media_type=media_type))
            else:
                rr.log(cam_path, rr.EncodedImage(contents=raw))
            return
        except Exception as e:
            print(f"[viz_rerun] EncodedImage 失败,回退到 cv2 解码: {e}")
            _ENCODED_IMAGE_FAILED = True

    # 回退:cv2 解码成 ndarray 再 log
    if isinstance(raw_or_arr, (bytes, bytearray, memoryview)):
        arr = _decode_image_bytes(raw_or_arr)
        if arr is not None:
            rr.log(cam_path, rr.Image(arr))
    else:
        rr.log(cam_path, rr.Image(raw_or_arr))


def log_episode(data, global_start):
    """Log a whole episode in bulk. 尽量用 send_columns,图像走 EncodedImage。"""
    max_frames = len(data["robot0"]["poses"])
    frame_seq = np.arange(global_start, global_start + max_frames, dtype=np.int64)

    for r in ROBOT_IDS:
        prefix = f"robot{r}"
        poses_arr = data[prefix]["poses"]                                # (N,4,4)
        positions = poses_arr[:, :3, 3].astype(np.float32)               # (N,3)
        quats = Rotation.from_matrix(poses_arr[:, :3, :3]).as_quat().astype(np.float32)
        grip = data[prefix]["gripper"]
        offsets = np.maximum(grip * 0.5, 0.03).astype(np.float32)

        # ── 整条轨迹:本 ep 首帧一次性 log ───────────────────────────────────
        rr.set_time("frame", sequence=int(frame_seq[0]))
        rr.log(
            f"world/robot{r}/trajectory",
            rr.LineStrips3D([positions], colors=[ROBOT_COLORS[r]], radii=[0.003]),
        )

        # ── 批量 send_columns:标量时序 + Transform3D ───────────────────────
        try:
            tc = rr.TimeColumn("frame", sequence=frame_seq)
            # 标量
            for axis_idx, axis in enumerate(("x", "y", "z")):
                rr.send_columns(
                    f"timeseries/robot{r}/eef_{axis}", indexes=[tc],
                    columns=rr.Scalars.columns(scalars=positions[:, axis_idx]),
                )
            rr.send_columns(
                f"timeseries/robot{r}/gripper_width", indexes=[tc],
                columns=rr.Scalars.columns(scalars=grip),
            )
            # EEF 位姿(整条批量)
            rr.send_columns(
                f"world/robot{r}/eef", indexes=[tc],
                columns=rr.Transform3D.columns(translation=positions, quaternion=quats),
            )
            # 手指左右偏移(整条批量)
            left_t = np.column_stack([
                np.full(max_frames, 0.02, dtype=np.float32), -offsets,
                np.full(max_frames, -0.04, dtype=np.float32),
            ])
            right_t = left_t.copy()
            right_t[:, 1] = offsets
            rr.send_columns(
                f"world/robot{r}/eef/gripper/left", indexes=[tc],
                columns=rr.Transform3D.columns(translation=left_t),
            )
            rr.send_columns(
                f"world/robot{r}/eef/gripper/right", indexes=[tc],
                columns=rr.Transform3D.columns(translation=right_t),
            )
            batched = True
        except (AttributeError, TypeError):
            batched = False

        # ── 每帧仍需逐帧 log 的:图像 (+ 老 SDK fallback)───────────────────
        cam_paths = {
            "visual":        f"cameras/robot{r}/visual",
            "left_tactile":  f"cameras/robot{r}/left_tactile",
            "right_tactile": f"cameras/robot{r}/right_tactile",
        }
        for i in range(max_frames):
            rr.set_time("frame", sequence=int(frame_seq[i]))
            if not batched:
                rr.log(
                    f"world/robot{r}/eef",
                    rr.Transform3D(
                        translation=positions[i],
                        quaternion=rr.Quaternion(xyzw=quats[i]),
                    ),
                )
                rr.log(f"world/robot{r}/eef/gripper/left",
                       rr.Transform3D(translation=[0.02, -offsets[i], -0.04]))
                rr.log(f"world/robot{r}/eef/gripper/right",
                       rr.Transform3D(translation=[0.02,  offsets[i], -0.04]))
                rr.log(f"timeseries/robot{r}/eef_x", rr.Scalars(float(positions[i, 0])))
                rr.log(f"timeseries/robot{r}/eef_y", rr.Scalars(float(positions[i, 1])))
                rr.log(f"timeseries/robot{r}/eef_z", rr.Scalars(float(positions[i, 2])))
                rr.log(f"timeseries/robot{r}/gripper_width", rr.Scalars(float(grip[i])))
            for sensor, cam_path in cam_paths.items():
                imgs = data[prefix].get(sensor, [])
                if imgs and i < len(imgs):
                    _image_log(cam_path, imgs[i])


# ── blueprint ─────────────────────────────────────────────────────────────────
def make_blueprint():
    """Configure the Rerun viewer layout."""
    return rrb.Blueprint(
        rrb.Horizontal(
            # left column: 3D world + timeseries
            rrb.Vertical(
                rrb.Spatial3DView(name="3D World", origin="world"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Robot 0", origin="timeseries/robot0"),
                    rrb.TimeSeriesView(name="Robot 1", origin="timeseries/robot1"),
                ),
                row_shares=[3, 2],
            ),
            # right column: camera feeds + gripper timeseries
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R0 Visual",  origin="cameras/robot0/visual"),
                    rrb.Spatial2DView(name="R0 L-Tact",  origin="cameras/robot0/left_tactile"),
                    rrb.Spatial2DView(name="R0 R-Tact",  origin="cameras/robot0/right_tactile"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R1 Visual",  origin="cameras/robot1/visual"),
                    rrb.Spatial2DView(name="R1 L-Tact",  origin="cameras/robot1/left_tactile"),
                    rrb.Spatial2DView(name="R1 R-Tact",  origin="cameras/robot1/right_tactile"),
                ),
                rrb.TimeSeriesView(name="Gripper Widths", origin="timeseries"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


# ── episode selection helper ──────────────────────────────────────────────────
def _parse_episodes(spec, n_total):
    """
    Parse --episode argument into a sorted list of episode indices.

    spec=None              → all episodes
    spec=["3"]             → [3]
    spec=["0","3","7"]     → [0, 3, 7]
    spec=["0-10"]          → [0,1,...,10]
    spec=["0-5","8","12-15"] → [0,1,2,3,4,5,8,12,13,14,15]
    """
    if spec is None:
        return list(range(n_total))

    indices = set()
    for token in spec:
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range: '{token}' (use start-end)")
            start, end = int(parts[0]), int(parts[1])
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))

    result = sorted(i for i in indices if 0 <= i < n_total)
    out_of_range = sorted(i for i in indices if not (0 <= i < n_total))
    if out_of_range:
        print(f"Warning: episode indices out of range (0–{n_total-1}), skipping: {out_of_range}")
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize LeRobot teleoperation data with Rerun.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/viz_rerun.py /path/to/dataset\n"
            "  python src/viz_rerun.py /path/to/dataset -e 3\n"
            "  python src/viz_rerun.py /path/to/dataset -e 0 3 7\n"
            "  python src/viz_rerun.py /path/to/dataset -e 0-10\n"
            "  python src/viz_rerun.py /path/to/dataset -e 0-5 8 12-15\n"
            "  python src/viz_rerun.py /path/to/dataset --save out.rrd\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="LeRobot dataset directory (must contain meta/info.json)",
    )
    parser.add_argument(
        "--episode", "-e", nargs="+", default=None,
        metavar="EP",
        help=(
            "Episodes to visualize. Accepts:\n"
            "  single:  -e 3\n"
            "  multiple: -e 0 3 7\n"
            "  range:   -e 0-10\n"
            "  mixed:   -e 0-5 8 12-15\n"
            "(default: all episodes)"
        ),
    )
    parser.add_argument(
        "--save", "-s", type=str, default=None,
        help="Save recording to .rrd file instead of spawning the viewer",
    )
    args = parser.parse_args()

    # resolve to absolute path so the script works from any CWD
    dataset_path = os.path.abspath(args.dataset)

    if not os.path.exists(dataset_path):
        print(f"Error: path not found: {dataset_path}")
        sys.exit(1)

    if not os.path.exists(os.path.join(dataset_path, "meta", "info.json")):
        print("Error: not a LeRobot dataset (missing meta/info.json)")
        sys.exit(1)

    # ── initialise Rerun ──────────────────────────────────────────────────────
    blueprint = make_blueprint()
    rr.init("robot_visualization", spawn=(args.save is None))
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        rr.send_blueprint(blueprint)

    # ── static geometry ───────────────────────────────────────────────────────
    print("Loading meshes...")
    meshes = load_static_meshes()
    log_static_geometry(meshes)
    print(f"  gripper: {'✓' if 'gripper_left' in meshes else '✗ (STL not found)'}  "
          f"controllers: {'✓' if 'controller_left' in meshes else '✗ (STL not found)'}")

    # ── episode data ──────────────────────────────────────────────────────────
    import json
    with open(os.path.join(dataset_path, "meta", "info.json")) as f:
        meta = json.load(f)
    n_episodes = meta["total_episodes"]
    print(f"LeRobot dataset: {meta['total_frames']:,} frames | {n_episodes} episodes")

    ep_indices = _parse_episodes(args.episode, n_episodes)

    global_frame = 0
    for ep_idx in ep_indices:
        rr.set_time("episode", sequence=ep_idx)
        print(f"  ep {ep_idx:3d}:", end="", flush=True)
        try:
            data = load_lerobot_episode(dataset_path, ep_idx)
        except FileNotFoundError as e:
            print(f"  skip (not found: {e})")
            continue
        max_frames = len(data["robot0"]["poses"])
        print(f" {max_frames} frames", end="", flush=True)
        log_episode(data, global_frame)
        global_frame += max_frames
        print("  ✓")

    print(f"\nDone — {global_frame} frames logged.")
    if args.save:
        print(f"Saved to: {args.save}")
    else:
        print("Rerun viewer opened. Use the timeline slider to scrub frames.")


if __name__ == "__main__":
    main()
