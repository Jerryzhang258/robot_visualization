import argparse
import json
import os
import datetime

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree
import pyglet
from trimesh.viewer import SceneViewer
from pyglet.window import key as pyglet_key


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


def _principal_axes(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return np.eye(3), np.zeros(3)
    center = pts.mean(axis=0)
    centered = pts - center
    u, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes, center


def _rigid_transform(T):
    out = np.array(T, dtype=np.float64)
    u, _, vt = np.linalg.svd(out[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out[:3, :3] = r
    return out


def _mean_nn_distance(src_pts, target_pts):
    if len(src_pts) == 0 or len(target_pts) == 0:
        return float("inf")
    tree = cKDTree(target_pts)
    dist, _ = tree.query(src_pts, k=1)
    return float(np.mean(dist))


def _initial_align(source_pts, target_pts):
    src_axes, src_center = _principal_axes(source_pts)
    tgt_axes, tgt_center = _principal_axes(target_pts)
    rot = tgt_axes @ src_axes.T
    if np.linalg.det(rot) < 0:
        rot[:, -1] *= -1
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = tgt_center - rot @ src_center
    return tf


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


def _build_scene(base_mesh, assem_mesh, left_mesh, right_mesh, axis_poses=None, axis_scale=0.1):
    scene = trimesh.Scene()
    scene.add_geometry(_colorize(base_mesh, [0.7, 0.7, 0.7, 0.9]), node_name="base")
    scene.add_geometry(_colorize(assem_mesh, [0.2, 0.6, 0.9, 0.35]), node_name="assem")
    scene.add_geometry(_colorize(left_mesh, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left")
    scene.add_geometry(_colorize(right_mesh, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right")
    if axis_poses:
        axis_mesh = _thick_axis_mesh(scale=axis_scale)
        for i, pose in enumerate(axis_poses):
            scene.add_geometry(axis_mesh.copy(), transform=pose, node_name=f"axis_{i}")
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


def _average_transform(a, b):
    t = 0.5 * (a[:3, 3] + b[:3, 3])

    def _as_rotation(m):
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return r

    r_a = _as_rotation(a[:3, :3])
    r_b = _as_rotation(b[:3, :3])
    rotations = Rotation.from_matrix([r_a, r_b])
    slerp = Slerp([0, 1], rotations)
    r = slerp(0.5).as_matrix()
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _tip_axis_vector(T, tip_axis):
    axis_map = {
        "x": (0, 1.0),
        "-x": (0, -1.0),
        "y": (1, 1.0),
        "-y": (1, -1.0),
        "z": (2, 1.0),
        "-z": (2, -1.0),
    }
    idx, sign = axis_map.get(tip_axis, (2, 1.0))
    return T[:3, idx] * sign


def _compute_center_pose_from_residuals(t_left, left_pts, right_pts, tip_axis="z"):
    left_pos = left_pts.mean(axis=0) if left_pts.size else t_left[:3, 3]
    right_pos = right_pts.mean(axis=0) if right_pts.size else t_left[:3, 3]
    delta = left_pos - right_pos
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-6:
        return t_left
    x_axis = delta / delta_norm

    z_axis = _tip_axis_vector(t_left, tip_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return t_left
    z_axis = z_axis / z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return t_left
    y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    if np.dot(x_axis, delta) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    center = np.eye(4)
    center[:3, 0] = x_axis
    center[:3, 1] = y_axis
    center[:3, 2] = z_axis
    center[:3, 3] = 0.5 * (left_pos + right_pos)
    return center


def _thick_axis_mesh(scale=0.1):
    radius = max(scale * 0.03, 0.002)
    length = scale

    def _axis_cyl(color, axis):
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        rgba = (np.array(color) * 255).astype(np.uint8)
        cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
        if axis == "x":
            tf = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
            tf[:3, 3] = [length / 2.0, 0, 0]
        elif axis == "y":
            tf = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
            tf[:3, 3] = [0, length / 2.0, 0]
        else:
            tf = np.eye(4)
            tf[:3, 3] = [0, 0, length / 2.0]
        cyl.apply_transform(tf)
        return cyl

    x = _axis_cyl([1.0, 0.0, 0.0, 1.0], "x")
    y = _axis_cyl([0.0, 1.0, 0.0, 1.0], "y")
    z = _axis_cyl([0.0, 0.0, 1.0, 1.0], "z")
    return trimesh.util.concatenate([x, y, z])


def _center_pose_for_mesh(mesh, base_tf=None):
    centroid = mesh.centroid
    pose = np.eye(4)
    if base_tf is not None:
        pose[:3, :3] = base_tf[:3, :3]
        pos = base_tf @ np.array([centroid[0], centroid[1], centroid[2], 1.0])
        pose[:3, 3] = pos[:3]
    else:
        pose[:3, 3] = centroid
    return pose


def _manual_adjust_loop(no_finger_aligned, assem, finger, t_left, t_right):
    adjust_left = np.eye(4)
    adjust_right = np.eye(4)
    if hasattr(assem, "extents"):
        axis_scale = float(max(assem.extents)) * 0.2
    else:
        axis_scale = 0.1
    axis_scale = max(axis_scale, 0.02)

    help_text = (
        "Manual adjust commands:\n"
        "  (Viewer) press 'h' in the window to see SceneViewer controls.\n"
        "  l tx ty tz rx ry rz  (adjust left finger)\n"
        "  r tx ty tz rx ry rz  (adjust right finger)\n"
        "  b tx ty tz rx ry rz  (adjust both fingers)\n"
        "  a                   (accept and save)\n"
        "  q                   (quit without saving)\n"
        "  h/?/help            (this help)\n"
        "Units: meters for translation, degrees for rotation.\n"
    )
    print(help_text)

    while True:
        left_tf = t_left @ adjust_left
        right_tf = t_right @ adjust_right
        center_tf = _average_transform(left_tf, right_tf)

        left_mesh = finger.copy()
        left_mesh.apply_transform(left_tf)
        right_mesh = finger.copy()
        right_mesh.apply_transform(right_tf)

        axis_poses = [
            np.eye(4),
            left_tf,
            right_tf,
            center_tf,
        ]
        axis_poses = [
            _center_pose_for_mesh(no_finger_aligned),
            _center_pose_for_mesh(assem),
            _center_pose_for_mesh(finger, base_tf=left_tf),
            _center_pose_for_mesh(finger, base_tf=right_tf),
            center_tf,
        ]
        scene = _build_scene(
            no_finger_aligned,
            assem,
            left_mesh,
            right_mesh,
            axis_poses=axis_poses,
            axis_scale=axis_scale,
        )
        scene.show()

        cmd = input("adjust> ").strip()
        if not cmd:
            continue
        if cmd.lower() in {"h", "help", "?"}:
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


def _align_local_axis(T, axis_name, target_dir):
    axis_map = {
        "x": (0, 1.0),
        "-x": (0, -1.0),
        "y": (1, 1.0),
        "-y": (1, -1.0),
        "z": (2, 1.0),
        "-z": (2, -1.0),
    }
    idx, sign = axis_map.get(axis_name, (0, 1.0))
    axis_vec = T[:3, idx] * sign
    if np.linalg.norm(axis_vec) < 1e-9:
        return T
    target = target_dir / (np.linalg.norm(target_dir) + 1e-9)
    rot = trimesh.geometry.align_vectors(axis_vec, target)
    T[:3, :3] = rot[:3, :3] @ T[:3, :3]
    return T


def _align_axis_in_world(center_pose, left_rel, axis_index, target_dir):
    left_tf = center_pose @ left_rel
    axis_vec = left_tf[:3, axis_index]
    if np.linalg.norm(axis_vec) < 1e-9:
        return left_rel
    target = target_dir / (np.linalg.norm(target_dir) + 1e-9)
    rot = trimesh.geometry.align_vectors(axis_vec, target)
    left_tf = rot @ left_tf
    return np.linalg.inv(center_pose) @ left_tf


class _NudgeViewer(SceneViewer):
    def __init__(self, scene, state):
        self.state = state
        super().__init__(scene, start_loop=False, smooth=False)
        self.accepted = False
        self._refresh()

    def on_key_press(self, symbol, modifiers):
        if symbol in (pyglet_key.Q, pyglet_key.ESCAPE):
            self.accepted = False
            self.close()
            pyglet.app.exit()
            return
        if symbol == pyglet_key.ENTER:
            self.accepted = True
            self.close()
            pyglet.app.exit()
            return
        if symbol == pyglet_key.M:
            self.state["target"] = "finger" if self.state["target"] == "center" else "center"
            print(f"Target: {self.state['target']}")
            return
        if symbol == pyglet_key.V:
            print(
                f"Steps: translation={self.state['t_step']:.4f}m "
                f"rotation={self.state['r_step']:.1f}deg target={self.state['target']}"
            )
            return
        if symbol == pyglet_key.P:
            _save_nudge_snapshot(self.state["save_dir"], self.state["center_pose"], self.state["left_rel"])
            print("Saved nudge snapshot.")
            return
        if symbol in (pyglet_key.X, pyglet_key.Y, pyglet_key.Z):
            axis_map = {pyglet_key.X: 0, pyglet_key.Y: 1, pyglet_key.Z: 2}
            axis_index = axis_map[symbol]
            toggle_key = f"align_toggle_{axis_index}"
            self.state[toggle_key] = not self.state.get(toggle_key, False)
            sign = 1.0 if self.state[toggle_key] else -1.0
            target = self.state["center_pose"][:3, axis_index] * sign
            self.state["left_rel"] = _align_axis_in_world(
                self.state["center_pose"],
                self.state["left_rel"],
                axis_index,
                target,
            )
            self._refresh()
            return
        if symbol == pyglet_key.BRACKETLEFT:
            self.state["t_step"] = max(self.state["t_step"] * 0.5, 1e-5)
            return
        if symbol == pyglet_key.BRACKETRIGHT:
            self.state["t_step"] = min(self.state["t_step"] * 2.0, 0.1)
            return
        if symbol == pyglet_key.SEMICOLON:
            self.state["r_step"] = max(self.state["r_step"] * 0.5, 0.1)
            return
        if symbol == pyglet_key.APOSTROPHE:
            self.state["r_step"] = min(self.state["r_step"] * 2.0, 45.0)
            return

        tx = ty = tz = rx = ry = rz = 0.0
        if symbol == pyglet_key.W:
            ty = self.state["t_step"]
        elif symbol == pyglet_key.S:
            ty = -self.state["t_step"]
        elif symbol == pyglet_key.A:
            tx = -self.state["t_step"]
        elif symbol == pyglet_key.D:
            tx = self.state["t_step"]
        elif symbol == pyglet_key.R:
            tz = self.state["t_step"]
        elif symbol == pyglet_key.F:
            tz = -self.state["t_step"]
        elif symbol == pyglet_key.I:
            rx = self.state["r_step"]
        elif symbol == pyglet_key.K:
            rx = -self.state["r_step"]
        elif symbol == pyglet_key.J:
            ry = self.state["r_step"]
        elif symbol == pyglet_key.L:
            ry = -self.state["r_step"]
        elif symbol == pyglet_key.U:
            rz = self.state["r_step"]
        elif symbol == pyglet_key.O:
            rz = -self.state["r_step"]
        else:
            return

        inc = _transform_from_params(tx, ty, tz, rx, ry, rz)
        if self.state["target"] == "center":
            self.state["center_pose"] = self.state["center_pose"] @ inc
        else:
            self.state["left_rel"] = self.state["left_rel"] @ inc
        self._refresh()

    def _refresh(self):
        center_pose = self.state["center_pose"]
        left_rel = self.state["left_rel"]
        left_tf = center_pose @ left_rel
        mirror = np.eye(4)
        mirror[0, 0] = -1.0
        right_tf = center_pose @ mirror @ left_rel

        self.scene.graph.update("finger_left", matrix=left_tf)
        self.scene.graph.update("finger_right", matrix=right_tf)
        self.scene.graph.update("axis_left", matrix=_center_pose_for_mesh(self.state["finger"], base_tf=left_tf))
        self.scene.graph.update("axis_right", matrix=_center_pose_for_mesh(self.state["finger"], base_tf=right_tf))
        self.scene.graph.update("axis_center", matrix=center_pose)
        self._redraw = True
        self.dispatch_event("on_draw")
        self.flip()


def _save_nudge_snapshot(save_dir, center_pose, left_rel):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"nudge_snapshot_{ts}.json")
    payload = {
        "center_pose": center_pose.tolist(),
        "left_rel": left_rel.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

def _nudge_initial_loop(no_finger_aligned, assem, finger, center_pose, left_rel, inward_axis="x",
                        save_dir="src/finger_calibrate/output", tip_axis="z"):
    if hasattr(assem, "extents"):
        axis_scale = float(max(assem.extents)) * 0.2
    else:
        axis_scale = 0.1
    axis_scale = max(axis_scale, 0.02)

    help_text = (
        "Nudge controls (focus viewer window):\n"
        "  m            toggle target (center/finger)\n"
        "  w/s          +Y/-Y   a/d          -X/+X\n"
        "  r/f          +Z/-Z   i/k          +Rx/-Rx\n"
        "  j/l          +Ry/-Ry u/o          +Rz/-Rz\n"
        "  [ / ]        translation step down/up\n"
        "  ; / '        rotation step down/up\n"
        "  v            view current steps\n"
        "  x/y/z        align finger tip axis to center X/Y/Z (toggles +/-)\n"
        "  p            save current poses snapshot\n"
        "  enter        accept and start optimization\n"
        "  q/esc        quit without saving\n"
    )
    print(help_text)

    inward_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    left_rel = _align_local_axis(left_rel, inward_axis, inward_dir)

    left_tf = center_pose @ left_rel
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    right_tf = center_pose @ mirror @ left_rel

    scene = trimesh.Scene()
    scene.add_geometry(_colorize(no_finger_aligned, [0.7, 0.7, 0.7, 0.9]), node_name="base")
    scene.add_geometry(_colorize(assem, [0.2, 0.6, 0.9, 0.35]), node_name="assem")
    scene.add_geometry(_colorize(finger, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left", transform=left_tf)
    scene.add_geometry(_colorize(finger, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right", transform=right_tf)
    axis_mesh = _thick_axis_mesh(scale=axis_scale)
    scene.add_geometry(axis_mesh.copy(), node_name="axis_no_finger", transform=_center_pose_for_mesh(no_finger_aligned))
    scene.add_geometry(axis_mesh.copy(), node_name="axis_assem", transform=_center_pose_for_mesh(assem))
    scene.add_geometry(axis_mesh.copy(), node_name="axis_left", transform=_center_pose_for_mesh(finger, base_tf=left_tf))
    scene.add_geometry(axis_mesh.copy(), node_name="axis_right", transform=_center_pose_for_mesh(finger, base_tf=right_tf))
    scene.add_geometry(axis_mesh.copy(), node_name="axis_center", transform=center_pose)

    t_step = max(axis_scale * 0.05, 0.001)
    r_step = 5.0

    state = {
        "center_pose": center_pose,
        "left_rel": left_rel,
        "target": "center",
        "t_step": t_step,
        "r_step": r_step,
        "finger": finger,
        "save_dir": save_dir,
        "tip_axis": tip_axis,
    }
    viewer = _NudgeViewer(scene, state)
    pyglet.app.run()
    if not viewer.accepted:
        return None, None
    return state["center_pose"], state["left_rel"]


def calibrate(no_finger_path, finger_path, assem_path, out_dir,
              samples=50000, icp_iters=1500, residual_thresh=0.1,
              split_axis="y", split_value=0.0, manual=False,
              tip_axis="z", symmetric=True, nudge=False,
              inward_axis="x", keep_best=True):
    no_finger = trimesh.load(no_finger_path)
    finger = trimesh.load(finger_path)
    assem = trimesh.load(assem_path)

    nf_pts = _sample_points(no_finger, samples)
    assem_pts = _sample_points(assem, samples)

    t_nf_to_assem = _rigid_transform(_run_icp(nf_pts, assem_pts, max_iterations=icp_iters))
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
    init_left = _initial_align(finger_pts, left_pts)
    center_pose_assem = _compute_center_pose_from_residuals(
        init_left,
        left_pts,
        right_pts,
        tip_axis=tip_axis,
    )
    left_rel = np.linalg.inv(center_pose_assem) @ init_left
    if nudge:
        center_pose_assem, left_rel = _nudge_initial_loop(
            no_finger_aligned,
            assem,
            finger,
            center_pose_assem,
            left_rel,
            inward_axis=inward_axis,
            save_dir=out_dir,
            tip_axis=tip_axis,
        )
        if center_pose_assem is None:
            print("Initial nudging canceled. No output saved.")
            return None
    initial_left = center_pose_assem @ left_rel
    mirror = np.eye(4)
    mirror[0, 0] = -1.0  # mirror across center YZ plane
    initial_right = center_pose_assem @ (mirror @ left_rel)

    t_finger_left = _rigid_transform(_run_icp(
        finger_pts,
        left_pts,
        initial=initial_left,
        max_iterations=icp_iters,
    ))
    if symmetric:
        left_rel = np.linalg.inv(center_pose_assem) @ t_finger_left
        t_finger_right = center_pose_assem @ (mirror @ left_rel)
    else:
        init_right = _initial_align(finger_pts, right_pts)
        t_finger_right = _rigid_transform(_run_icp(
            finger_pts, right_pts, initial=init_right, max_iterations=icp_iters
        ))

    if keep_best:
        initial_left_dist = _mean_nn_distance(
            finger_pts @ initial_left[:3, :3].T + initial_left[:3, 3],
            left_pts,
        )
        opt_left_dist = _mean_nn_distance(
            finger_pts @ t_finger_left[:3, :3].T + t_finger_left[:3, 3],
            left_pts,
        )
        initial_right_dist = _mean_nn_distance(
            finger_pts @ initial_right[:3, :3].T + initial_right[:3, 3],
            right_pts,
        )
        opt_right_dist = _mean_nn_distance(
            finger_pts @ t_finger_right[:3, :3].T + t_finger_right[:3, 3],
            right_pts,
        )
        if opt_left_dist + opt_right_dist > initial_left_dist + initial_right_dist:
            print("Optimization increased residuals; keeping initial pose.")
            t_finger_left = initial_left
            t_finger_right = initial_right

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
        t_finger_left = _rigid_transform(t_finger_left @ adjust_left)
        t_finger_right = _rigid_transform(t_finger_right @ adjust_right)

    t_nf_to_assem_inv = np.linalg.inv(t_nf_to_assem)
    t_finger_left_to_no_finger = t_nf_to_assem_inv @ t_finger_left
    t_finger_right_to_no_finger = t_nf_to_assem_inv @ t_finger_right
    t_finger_base_to_no_finger = t_nf_to_assem_inv @ center_pose_assem

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
        "transform_finger_left_to_no_finger": t_finger_left_to_no_finger.tolist(),
        "transform_finger_right_to_no_finger": t_finger_right_to_no_finger.tolist(),
        "transform_finger_base_to_no_finger": t_finger_base_to_no_finger.tolist(),
        "tip_axis": tip_axis,
        "symmetric": symmetric,
        "nudge": nudge,
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
    parser.add_argument("--tip-axis", choices=["x", "-x", "y", "-y", "z", "-z"], default="z",
                        help="Which local axis of the finger mesh points toward the fingertip.")
    parser.add_argument("--inward-axis", choices=["x", "-x", "y", "-y", "z", "-z"], default="x",
                        help="Which local axis of the finger mesh points inward toward the center frame.")
    parser.add_argument("--no-symmetric", action="store_true",
                        help="Disable symmetric right finger solve and run ICP separately.")
    parser.add_argument("--manual", action="store_true", help="Show viewer and allow manual adjustments before saving.")
    parser.add_argument("--nudge", action="store_true",
                        help="Nudge initial center/finger poses before optimization.")
    parser.add_argument("--no-keep-best", action="store_true",
                        help="Allow ICP result even if residuals increase.")
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
        tip_axis=args.tip_axis,
        symmetric=(not args.no_symmetric),
        nudge=args.nudge,
        inward_axis=args.inward_axis,
        keep_best=(not args.no_keep_best),
    )


if __name__ == "__main__":
    main()
