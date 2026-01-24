import argparse
import json
import os

import numpy as np
import trimesh

from calibrate_finger import _compute_center_pose, _thick_axis_mesh, _center_pose_for_mesh


def _scale_transform(T, scale):
    out = np.array(T, dtype=np.float64)
    out[:3, 3] = out[:3, 3] * scale
    return out


def _load_calibration(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tip_axis = data.get("tip_axis", "z")
    if "transform_finger_left_to_no_finger" in data:
        t_left = np.array(data["transform_finger_left_to_no_finger"], dtype=np.float64)
        t_right = np.array(data["transform_finger_right_to_no_finger"], dtype=np.float64)
    else:
        t_nf_to_assem = np.array(data["transform_no_finger_to_assem"], dtype=np.float64)
        t_left_assem = np.array(data["transform_finger_left"], dtype=np.float64)
        t_right_assem = np.array(data["transform_finger_right"], dtype=np.float64)
        t_nf_to_assem_inv = np.linalg.inv(t_nf_to_assem)
        t_left = t_nf_to_assem_inv @ t_left_assem
        t_right = t_nf_to_assem_inv @ t_right_assem

    t_center = _compute_center_pose(t_left, t_right, tip_axis=tip_axis)
    return data, t_left, t_right, t_center


def _colorize(mesh, rgba):
    colored = mesh.copy()
    rgba_arr = (np.array(rgba) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba_arr, (len(colored.vertices), 1))
    return colored


def main():
    parser = argparse.ArgumentParser(description="Verify finger calibration by reassembling meshes.")
    parser.add_argument("--json", default=os.path.join("src", "finger_calibrate", "output", "calibration_result.json"))
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to meshes and translation components.")
    args = parser.parse_args()

    data, t_left, t_right, t_center = _load_calibration(args.json)

    no_finger_path = data["no_finger_path"]
    finger_path = data["finger_path"]

    no_finger = trimesh.load(no_finger_path)
    finger = trimesh.load(finger_path)

    if args.scale != 1.0:
        no_finger.apply_scale(args.scale)
        finger.apply_scale(args.scale)
        t_left = _scale_transform(t_left, args.scale)
        t_right = _scale_transform(t_right, args.scale)
        t_center = _scale_transform(t_center, args.scale)

    left_mesh = finger.copy()
    left_mesh.apply_transform(t_left)
    right_mesh = finger.copy()
    right_mesh.apply_transform(t_right)

    scene = trimesh.Scene()
    scene.add_geometry(_colorize(no_finger, [0.7, 0.7, 0.7, 0.9]), node_name="no_finger")
    scene.add_geometry(_colorize(left_mesh, [0.9, 0.4, 0.4, 0.9]), node_name="finger_left")
    scene.add_geometry(_colorize(right_mesh, [0.4, 0.9, 0.4, 0.9]), node_name="finger_right")

    axis_scale = float(max(no_finger.extents)) * 0.2 if hasattr(no_finger, "extents") else 0.1
    axis_scale = max(axis_scale, 0.02)
    axis_mesh = _thick_axis_mesh(scale=axis_scale)
    axis_poses = [
        _center_pose_for_mesh(no_finger),
        _center_pose_for_mesh(finger, base_tf=t_left),
        _center_pose_for_mesh(finger, base_tf=t_right),
        t_center,
    ]
    for i, pose in enumerate(axis_poses):
        scene.add_geometry(axis_mesh.copy(), transform=pose, node_name=f"axis_{i}")

    scene.show()


if __name__ == "__main__":
    main()
