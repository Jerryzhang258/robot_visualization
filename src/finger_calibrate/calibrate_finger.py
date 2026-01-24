import argparse
import json
import os

import numpy as np
import trimesh


def _sample_points(mesh, count):
    return mesh.sample(count)


def _run_icp(source_pts, target_pts, initial=None, max_iterations=50):
    result = trimesh.registration.icp(
        source_pts,
        target_pts,
        initial=initial,
        max_iterations=max_iterations,
    )
    if isinstance(result, tuple):
        matrix = result[0]
    else:
        matrix = result
    return matrix


def _closest_distances(mesh, points):
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def _split_points(points, axis="y", value=0.0):
    axis_index = "xyz".index(axis)
    left = points[points[:, axis_index] < value]
    right = points[points[:, axis_index] > value]
    return left, right


def _colorize(mesh, rgba):
    colored = mesh.copy()
    rgba_arr = (np.array(rgba) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba_arr, (len(colored.vertices), 1))
    return colored


def _build_scene(base_mesh, assem_mesh, left_mesh, right_mesh):
    scene = trimesh.Scene()
    scene.add_geometry(_colorize(base_mesh, [0.7, 0.7, 0.7, 0.9]), node_name="base")
    scene.add_geometry(_colorize(assem_mesh, [0.2, 0.6, 0.9, 0.35]), node_name="assem")
    scene.add_geometry(_colorize(left_mesh, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left")
    scene.add_geometry(_colorize(right_mesh, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right")
    return scene


def _transform_from_params(tx, ty, tz, rx, ry, rz):
    rot = trimesh.transformations.euler_matrix(
        np.deg2rad(rx),
        np.deg2rad(ry),
        np.deg2rad(rz),
        "sxyz",
    )
    rot[:3, 3] = [tx, ty, tz]
    return rot


def _manual_adjust_loop(no_finger_aligned, assem, finger, t_left, t_right):
    adjust_left = np.eye(4)
    adjust_right = np.eye(4)

    help_text = (
        "Manual adjust commands:\n"
        "  l tx ty tz rx ry rz  (adjust left finger)\n"
        "  r tx ty tz rx ry rz  (adjust right finger)\n"
        "  b tx ty tz rx ry rz  (adjust both fingers)\n"
        "  a                   (accept and save)\n"
        "  q                   (quit without saving)\n"
        "  h                   (help)\n"
        "Units: meters for translation, degrees for rotation.\n"
    )
    print(help_text)

    while True:
        left_mesh = finger.copy()
        left_mesh.apply_transform(t_left @ adjust_left)
        right_mesh = finger.copy()
        right_mesh.apply_transform(t_right @ adjust_right)

        scene = _build_scene(no_finger_aligned, assem, left_mesh, right_mesh)
        scene.show()

        cmd = input("adjust> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "h":
            print(help_text)
            continue
        if cmd.lower() == "a":
            return adjust_left, adjust_right
        if cmd.lower() == "q":
            return None, None

        parts = cmd.split()
        if len(parts) != 7 or parts[0].lower() not in {"l", "r", "b"}:
            print("Invalid command. Use 'h' for help.")
            continue
        try:
            tx, ty, tz, rx, ry, rz = map(float, parts[1:])
        except ValueError:
            print("Invalid numeric values.")
            continue
        inc = _transform_from_params(tx, ty, tz, rx, ry, rz)
        if parts[0].lower() == "l":
            adjust_left = adjust_left @ inc
        elif parts[0].lower() == "r":
            adjust_right = adjust_right @ inc
        else:
            adjust_left = adjust_left @ inc
            adjust_right = adjust_right @ inc


def calibrate(no_finger_path, finger_path, assem_path, out_dir,
              samples=50000, icp_iters=80, residual_thresh=0.5,
              split_axis="y", split_value=0.0, manual=False):
    no_finger = trimesh.load(no_finger_path)
    finger = trimesh.load(finger_path)
    assem = trimesh.load(assem_path)

    nf_pts = _sample_points(no_finger, samples)
    assem_pts = _sample_points(assem, samples)

    t_nf_to_assem = _run_icp(nf_pts, assem_pts, max_iterations=icp_iters)
    no_finger_aligned = no_finger.copy()
    no_finger_aligned.apply_transform(t_nf_to_assem)

    dist = _closest_distances(no_finger_aligned, assem_pts)
    residual_pts = assem_pts[dist > residual_thresh]
    if residual_pts.size == 0:
        raise RuntimeError("Residual points empty; lower --residual-thresh or increase --samples")

    left_pts, right_pts = _split_points(residual_pts, axis=split_axis, value=split_value)
    if left_pts.size == 0 or right_pts.size == 0:
        left_pts = residual_pts
        right_pts = residual_pts

    finger_pts = _sample_points(finger, samples)
    t_finger_left = _run_icp(finger_pts, left_pts, max_iterations=icp_iters)
    t_finger_right = _run_icp(finger_pts, right_pts, max_iterations=icp_iters)

    if manual:
        adjust_left, adjust_right = _manual_adjust_loop(
            no_finger_aligned,
            assem,
            finger,
            t_finger_left,
            t_finger_right,
        )
        if adjust_left is None:
            print("Manual adjustment canceled. No output saved.")
            return None
        t_finger_left = t_finger_left @ adjust_left
        t_finger_right = t_finger_right @ adjust_right

    os.makedirs(out_dir, exist_ok=True)

    left_mesh = finger.copy()
    left_mesh.apply_transform(t_finger_left)
    left_path = os.path.join(out_dir, "finger_left_aligned.stl")
    left_mesh.export(left_path)

    right_mesh = finger.copy()
    right_mesh.apply_transform(t_finger_right)
    right_path = os.path.join(out_dir, "finger_right_aligned.stl")
    right_mesh.export(right_path)

    nf_out = os.path.join(out_dir, "left_no_finger_aligned.stl")
    no_finger_aligned.export(nf_out)

    result = {
        "no_finger_path": no_finger_path,
        "finger_path": finger_path,
        "assem_path": assem_path,
        "samples": samples,
        "icp_iters": icp_iters,
        "residual_thresh": residual_thresh,
        "split_axis": split_axis,
        "split_value": split_value,
        "transform_no_finger_to_assem": t_nf_to_assem.tolist(),
        "transform_finger_left": t_finger_left.tolist(),
        "transform_finger_right": t_finger_right.tolist(),
        "outputs": {
            "left_no_finger_aligned": nf_out,
            "finger_left_aligned": left_path,
            "finger_right_aligned": right_path,
        },
    }

    json_path = os.path.join(out_dir, "calibration_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate finger meshes against an assembled reference.")
    parser.add_argument("--no-finger", default="src/meshes/left_no_finger.STL")
    parser.add_argument("--finger", default="src/meshes/finger.STL")
    parser.add_argument("--assem", default="src/meshes/left_assem.STL")
    parser.add_argument("--out", default="src/finger_calibrate/output")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--icp-iters", type=int, default=80)
    parser.add_argument("--residual-thresh", type=float, default=0.5)
    parser.add_argument("--split-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--split-value", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true", help="Show viewer and allow manual adjustments before saving.")
    args = parser.parse_args()

    calibrate(
        args.no_finger,
        args.finger,
        args.assem,
        args.out,
        samples=args.samples,
        icp_iters=args.icp_iters,
        residual_thresh=args.residual_thresh,
        split_axis=args.split_axis,
        split_value=args.split_value,
        manual=args.manual,
    )


if __name__ == "__main__":
    main()
