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
    return mesh.sample(count).astype(np.float32)


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


def _mean_nn_distance(src_pts, target_pts, max_samples=5000):
    if len(src_pts) == 0 or len(target_pts) == 0:
        return float("inf")
    src = np.asarray(src_pts, dtype=np.float32)
    tgt = np.asarray(target_pts, dtype=np.float32)
    if src.shape[0] > max_samples:
        idx = np.random.choice(src.shape[0], max_samples, replace=False)
        src = src[idx]
    tree = cKDTree(tgt)
    dist, _ = tree.query(src, k=1)
    return float(np.mean(dist))




def _multi_start_icp(source_pts, target_pts, initial, max_iterations, verbose=True):
    """Run ICP from multiple starting orientations to avoid local minima.
    
    Tests 24 different rotation candidates covering all major orientations.
    """
    # Generate comprehensive rotation candidates
    candidates = []
    
    # Identity
    candidates.append(np.eye(3))
    
    # 90, 180, 270 degree rotations around each axis
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        for angle in [90, 180, 270]:
            rot = Rotation.from_euler(axis_name, angle, degrees=True).as_matrix()
            candidates.append(rot)
    
    # Diagonal rotations (useful for complex geometries)
    for angles in [[90, 90, 0], [90, 0, 90], [0, 90, 90], 
                   [180, 90, 0], [180, 0, 90], [0, 180, 90],
                   [90, 90, 90], [180, 180, 90]]:
        rot = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        candidates.append(rot)
    
    if verbose:
        print(f"  Testing {len(candidates)} rotation candidates...")
    
    best_tf = None
    best_score = float("inf")
    scores = []
    
    for i, rot in enumerate(candidates):
        init = np.array(initial, dtype=np.float64)
        init[:3, :3] = init[:3, :3] @ rot
        
        try:
            tf = _rigid_transform(_run_icp(source_pts, target_pts, initial=init, max_iterations=max_iterations))
            transformed_pts = source_pts @ tf[:3, :3].T + tf[:3, 3]
            score = _mean_nn_distance(transformed_pts, target_pts)
            scores.append(score)
            
            if score < best_score:
                best_score = score
                best_tf = tf
                if verbose and i % 5 == 0:
                    print(f"    Candidate {i+1}/{len(candidates)}: score={score:.6f} (best so far)")
        except Exception as e:
            if verbose:
                print(f"    Candidate {i+1} failed: {e}")
            continue
    
    if verbose:
        scores_sorted = sorted(scores)
        print(f"  Best score: {best_score:.6f}")
        print(f"  Top 5 scores: {[f'{s:.6f}' for s in scores_sorted[:5]]}")
        print(f"  Score range: {min(scores):.6f} to {max(scores):.6f}")
    
    return best_tf


def _mirror_points(points, center_pose):
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    tf = center_pose @ mirror @ np.linalg.inv(center_pose)
    pts = np.asarray(points, dtype=np.float32)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    mirrored = (tf @ homog.T).T[:, :3]
    return mirrored


def _initial_align(source_pts, target_pts, verbose=True):
    """Compute initial alignment using principal component analysis.
    
    Returns multiple candidate transforms to reduce dependency on a single initialization.
    """
    src_axes, src_center = _principal_axes(source_pts)
    tgt_axes, tgt_center = _principal_axes(target_pts)
    
    # Try all 8 sign combinations for axis alignment
    candidates = []
    
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                src_axes_mod = src_axes.copy()
                src_axes_mod[:, 0] *= sx
                src_axes_mod[:, 1] *= sy
                src_axes_mod[:, 2] *= sz
                
                # Ensure right-handed coordinate system
                if np.linalg.det(src_axes_mod) < 0:
                    src_axes_mod[:, -1] *= -1
                
                rot = tgt_axes @ src_axes_mod.T
                if np.linalg.det(rot) < 0:
                    rot[:, -1] *= -1
                
                tf = np.eye(4)
                tf[:3, :3] = rot
                tf[:3, 3] = tgt_center - rot @ src_center
                
                # Score this alignment
                transformed = source_pts @ tf[:3, :3].T + tf[:3, 3]
                score = _mean_nn_distance(transformed, target_pts, max_samples=1000)
                candidates.append((score, tf))
    
    # Sort by score and return the best
    candidates.sort(key=lambda x: x[0])
    
    if verbose:
        print(f"  Initial alignment scores (top 3): {[f'{c[0]:.6f}' for c in candidates[:3]]}")
    
    return candidates[0][1]


def _closest_distances(mesh, points):
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def _split_points(points, axis="y", value=0.0):
    axis_index = "xyz".index(axis)
    left = points[points[:, axis_index] < value]
    right = points[points[:, axis_index] > value]
    return left, right


def _cluster_finger_points(points, verbose=True):
    """Cluster finger-only points into two separate finger clusters.
    
    Uses a simple distance-based clustering approach to find two connected 
    point groups, then returns them as left and right fingers.
    
    Args:
        points: Finger-only points extracted from assembled mesh
        verbose: Print progress info
        
    Returns:
        (cluster1_pts, cluster2_pts): Two point arrays for the two fingers
        Raises RuntimeError if less than 2 clusters found
    """
    if len(points) < 10:
        raise RuntimeError("Not enough finger points to cluster")
    
    # Build a KD-tree for neighbor queries
    tree = cKDTree(points)
    
    # Estimate eps based on point density using k-nearest neighbor distance
    k = min(10, len(points) - 1)
    distances, _ = tree.query(points, k=k+1)  # +1 because first is self
    mean_knn_dist = np.mean(distances[:, 1:])  # Exclude self-distance
    
    # Use a connectivity threshold based on local point density
    eps = mean_knn_dist * 3.0
    
    if verbose:
        print(f"  Clustering with connectivity threshold eps={eps:.4f}")
    
    # Simple connected components clustering using flood fill
    n_points = len(points)
    labels = np.full(n_points, -1, dtype=np.int32)
    current_label = 0
    
    for attempt in range(5):
        # Reset labels for new attempt
        labels = np.full(n_points, -1, dtype=np.int32)
        current_label = 0
        
        for start_idx in range(n_points):
            if labels[start_idx] >= 0:
                continue
            
            # Flood fill from this point
            stack = [start_idx]
            labels[start_idx] = current_label
            
            while stack:
                idx = stack.pop()
                # Find all neighbors within eps
                neighbor_indices = tree.query_ball_point(points[idx], eps)
                
                for neighbor_idx in neighbor_indices:
                    if labels[neighbor_idx] < 0:
                        labels[neighbor_idx] = current_label
                        stack.append(neighbor_idx)
            
            current_label += 1
        
        n_clusters = current_label
        
        if verbose:
            cluster_counts = [np.sum(labels == i) for i in range(n_clusters)]
            print(f"    Attempt {attempt+1}: eps={eps:.4f}, found {n_clusters} clusters")
            if n_clusters > 0:
                print(f"    Cluster sizes: {sorted(cluster_counts, reverse=True)[:5]}")
        
        if n_clusters >= 2:
            break
        
        # Increase eps for next attempt
        eps *= 1.5
    
    if n_clusters < 2:
        raise RuntimeError(f"Could not find 2 finger clusters (found {n_clusters}). "
                          "Try adjusting --residual-thresh")
    
    # Get the two largest clusters
    cluster_sizes = []
    for label in range(n_clusters):
        mask = labels == label
        cluster_sizes.append((label, np.sum(mask)))
    
    # Sort by size descending
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Get the two largest
    label1, size1 = cluster_sizes[0]
    label2, size2 = cluster_sizes[1]
    
    cluster1_pts = points[labels == label1]
    cluster2_pts = points[labels == label2]
    
    # Determine which is "left" based on centroid position
    # Just use the axis with largest separation for initial assignment (user can swap later)
    centroid1 = cluster1_pts.mean(axis=0)
    centroid2 = cluster2_pts.mean(axis=0)
    
    # Return in consistent order based on the axis with largest separation
    diff = centroid1 - centroid2
    main_axis = np.argmax(np.abs(diff))
    
    if diff[main_axis] < 0:
        # cluster1 is on the negative side of main axis
        left_pts, right_pts = cluster1_pts, cluster2_pts
    else:
        left_pts, right_pts = cluster2_pts, cluster1_pts
    
    if verbose:
        print(f"  ✓ Clustered into {len(left_pts)} and {len(right_pts)} points")
        print(f"    Separation axis: {'xyz'[main_axis]}, distance: {np.abs(diff[main_axis]):.4f}")
    
    return left_pts.astype(np.float32), right_pts.astype(np.float32)


def _extract_finger_only_points(assem_pts, quest_mesh, residual_thresh, verbose=True):
    """Extract points from assem that are far from quest mesh.
    
    This identifies the finger regions by finding where the assembled mesh
    differs from the quest base, ensuring ICP targets only the actual
    finger geometry and not the base.
    
    Args:
        assem_pts: Points sampled from assembled mesh
        quest_mesh: The quest base mesh (already aligned to assem)
        residual_thresh: Distance threshold to identify finger regions
        verbose: Print progress info
        
    Returns:
        finger_only_pts: Points that are far from the quest mesh
    """
    dist = _closest_distances(quest_mesh, assem_pts)
    finger_only_pts = assem_pts[dist > residual_thresh]
    
    if verbose:
        print(f"  Extracted {len(finger_only_pts)} finger-only points "
              f"({100*len(finger_only_pts)/len(assem_pts):.1f}% of assem)")
    
    return finger_only_pts.astype(np.float32, copy=False)


def _colorize(mesh, rgba):
    colored = mesh.copy()
    rgba_arr = (np.array(rgba) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba_arr, (len(colored.vertices), 1))
    return colored


class _FingerSelectViewer(SceneViewer):
    """Interactive viewer for selecting which cluster is the left finger."""
    
    def __init__(self, scene, state):
        self.state = state
        super().__init__(scene, start_loop=False, smooth=False)
        self.accepted = False
        self._update_title()
    
    def _update_title(self):
        swapped = self.state.get("swapped", False)
        status = "(SWAPPED)" if swapped else "(original)"
        print(f"  Current: RED=Left, GREEN=Right {status}")
    
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
        if symbol == pyglet_key.SPACE:
            # Swap left and right
            self.state["swapped"] = not self.state.get("swapped", False)
            
            # Swap colors by rebuilding point clouds
            left_pts = self.state["left_pts"]
            right_pts = self.state["right_pts"]
            
            if self.state["swapped"]:
                # Show swapped: right becomes red (left), left becomes green (right)
                left_colors = np.tile([230, 80, 80, 255], (len(right_pts), 1)).astype(np.uint8)
                right_colors = np.tile([80, 230, 80, 255], (len(left_pts), 1)).astype(np.uint8)
                left_pc = trimesh.PointCloud(right_pts, colors=left_colors)
                right_pc = trimesh.PointCloud(left_pts, colors=right_colors)
            else:
                # Show original
                left_colors = np.tile([230, 80, 80, 255], (len(left_pts), 1)).astype(np.uint8)
                right_colors = np.tile([80, 230, 80, 255], (len(right_pts), 1)).astype(np.uint8)
                left_pc = trimesh.PointCloud(left_pts, colors=left_colors)
                right_pc = trimesh.PointCloud(right_pts, colors=right_colors)
            
            # Update scene - use graph.update for transforms, rebuild geometry
            try:
                self.scene.delete_geometry("left_points")
            except:
                pass
            try:
                self.scene.delete_geometry("right_points")
            except:
                pass
            self.scene.add_geometry(left_pc, node_name="left_points")
            self.scene.add_geometry(right_pc, node_name="right_points")
            
            self._update_title()
            
            # Force full scene redraw
            self.on_resize(self.width, self.height)
            self._update_vertex_list()
            return


def _select_left_finger(left_pts, right_pts, quest_aligned, assem):
    """Interactive viewer to let user select which cluster is the left finger.
    
    Displays two point clusters in different colors (red=assumed left, green=assumed right)
    and allows user to swap them with SPACE key.
    
    Args:
        left_pts: Points on one side of the split (initially assumed left)
        right_pts: Points on other side of the split (initially assumed right)
        quest_aligned: The aligned quest mesh for context
        assem: The assembled mesh for context
        
    Returns:
        (final_left_pts, final_right_pts): Points after user confirmation
        None if user cancels
    """
    print("\n  ═══════════════════════════════════════════════════════")
    print("  FINGER SELECTION - Choose which cluster is LEFT finger")
    print("  ═══════════════════════════════════════════════════════")
    print("  Controls:")
    print("    SPACE  - Swap left/right assignment")
    print("    ENTER  - Confirm selection")
    print("    Q/ESC  - Cancel calibration")
    print("  ═══════════════════════════════════════════════════════")
    
    # Create point clouds for visualization with explicit color arrays
    left_colors = np.tile([230, 80, 80, 255], (len(left_pts), 1)).astype(np.uint8)
    right_colors = np.tile([80, 230, 80, 255], (len(right_pts), 1)).astype(np.uint8)
    left_pc = trimesh.PointCloud(left_pts, colors=left_colors)
    right_pc = trimesh.PointCloud(right_pts, colors=right_colors)
    
    # Build scene
    scene = trimesh.Scene()
    scene.add_geometry(_colorize(quest_aligned, [0.5, 0.5, 0.5, 0.3]), node_name="quest_base")
    scene.add_geometry(_colorize(assem, [0.2, 0.5, 0.8, 0.15]), node_name="assem")
    scene.add_geometry(left_pc, node_name="left_points")
    scene.add_geometry(right_pc, node_name="right_points")
    
    state = {
        "swapped": False,
        "left_pts": left_pts,
        "right_pts": right_pts,
    }
    
    viewer = _FingerSelectViewer(scene, state)
    pyglet.app.run()
    
    if not viewer.accepted:
        return None, None
    
    # Return points based on final selection
    if state["swapped"]:
        print("  ✓ Selection confirmed (swapped)")
        return right_pts, left_pts
    else:
        print("  ✓ Selection confirmed (original)")
        return left_pts, right_pts


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


def _initial_center_from_finger_points(left_pts, right_pts, tip_axis="z"):
    """Compute initial center pose from the centroids of finger-only point clouds.
    
    Places the center frame at the midpoint between left and right finger centroids,
    with X-axis pointing from right to left, Z-axis along tip direction.
    """
    left_centroid = left_pts.mean(axis=0) if left_pts.size else np.zeros(3)
    right_centroid = right_pts.mean(axis=0) if right_pts.size else np.zeros(3)
    
    center_pos = 0.5 * (left_centroid + right_centroid)
    
    # X-axis points from right to left
    delta = left_centroid - right_centroid
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-6:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = delta / delta_norm
    
    # Z-axis along tip direction (default up)
    axis_map = {
        "x": np.array([1, 0, 0]),
        "-x": np.array([-1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "-y": np.array([0, -1, 0]),
        "z": np.array([0, 0, 1]),
        "-z": np.array([0, 0, -1]),
    }
    z_axis = axis_map.get(tip_axis, np.array([0, 0, 1]))
    
    # Orthogonalize
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        # x and z are parallel, pick arbitrary y
        y_axis = np.array([0, 1, 0]) if abs(x_axis[1]) < 0.9 else np.array([1, 0, 0])
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    
    center = np.eye(4)
    center[:3, 0] = x_axis
    center[:3, 1] = y_axis
    center[:3, 2] = z_axis
    center[:3, 3] = center_pos
    return center


def _optimize_center_pose(t_finger_left, finger_pts, right_pts, tip_axis="z", max_iters=50, verbose=True):
    """Optimize center pose so that mirrored right finger aligns with right_pts.
    
    Given a fixed left finger transform, find the center pose that when used to 
    mirror the left finger, produces a right finger that best matches right_pts.
    
    The center pose needs to be optimized for:
    - Position: midpoint between left and right fingers
    - Orientation: the mirror plane normal (X-axis) must correctly bisect the fingers
    """
    if verbose:
        print("  Optimizing center pose for symmetric alignment...")
    
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    
    # Get left finger position
    left_pos = t_finger_left[:3, 3]
    left_rot = t_finger_left[:3, :3]
    
    # Estimate right finger centroid from target points
    right_centroid = right_pts.mean(axis=0) if right_pts.size else left_pos
    
    # The center position should be the midpoint
    center_pos = 0.5 * (left_pos + right_centroid)
    
    # Direction from center to left finger (this is the X-axis of center frame)
    delta = left_pos - right_centroid
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-6:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = delta / delta_norm
    
    # Z-axis from tip_axis of left finger
    z_axis = _tip_axis_vector(t_finger_left, tip_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = z_axis / z_norm
    
    # Orthogonalize to build proper frame
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        # x and z parallel, pick arbitrary y
        y_axis = np.array([0, 1, 0]) if abs(x_axis[1]) < 0.9 else np.array([1, 0, 0])
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    
    # Ensure x points from center toward left
    if np.dot(x_axis, left_pos - center_pos) < 0:
        x_axis = -x_axis
        y_axis = -y_axis
    
    def build_center_pose(pos, x, y, z):
        pose = np.eye(4)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        pose[:3, 3] = pos
        return pose
    
    def compute_right_finger(center_pose):
        left_rel = np.linalg.inv(center_pose) @ t_finger_left
        return center_pose @ (mirror @ left_rel)
    
    def score_center(center_pose):
        right_tf = compute_right_finger(center_pose)
        transformed_right = finger_pts @ right_tf[:3, :3].T + right_tf[:3, 3]
        return _mean_nn_distance(transformed_right, right_pts, max_samples=2000)
    
    best_center = build_center_pose(center_pos, x_axis, y_axis, z_axis)
    best_score = score_center(best_center)
    
    if verbose:
        print(f"    Initial score: {best_score:.6f}")
    
    # Optimization 1: Search for best center position along x-axis (finger spacing)
    search_range = delta_norm * 0.5 if delta_norm > 1e-6 else 1.0
    for offset in np.linspace(-search_range, search_range, 21):
        test_pos = center_pos + offset * x_axis
        test_center = build_center_pose(test_pos, x_axis, y_axis, z_axis)
        score = score_center(test_center)
        if score < best_score:
            best_score = score
            best_center = test_center
    
    # Optimization 2: Search for rotation of center frame around Z-axis
    # This adjusts the mirror plane orientation
    best_pos = best_center[:3, 3].copy()
    for angle_deg in np.linspace(-45, 45, 31):
        angle_rad = np.deg2rad(angle_deg)
        rot_z = Rotation.from_rotvec(angle_rad * z_axis).as_matrix()
        new_x = rot_z @ x_axis
        new_y = rot_z @ y_axis
        test_center = build_center_pose(best_pos, new_x, new_y, z_axis)
        score = score_center(test_center)
        if score < best_score:
            best_score = score
            best_center = test_center
    
    # Optimization 3: Search for rotation around Y-axis (tilt of mirror plane)
    best_x = best_center[:3, 0].copy()
    best_y = best_center[:3, 1].copy()
    best_z = best_center[:3, 2].copy()
    for angle_deg in np.linspace(-30, 30, 21):
        angle_rad = np.deg2rad(angle_deg)
        rot_y = Rotation.from_rotvec(angle_rad * best_y).as_matrix()
        new_x = rot_y @ best_x
        new_z = rot_y @ best_z
        test_center = build_center_pose(best_pos, new_x, best_y, new_z)
        score = score_center(test_center)
        if score < best_score:
            best_score = score
            best_center = test_center
    
    # Optimization 4: Fine-tune position along all axes
    best_pos = best_center[:3, 3].copy()
    best_x = best_center[:3, 0].copy()
    best_y = best_center[:3, 1].copy()
    best_z = best_center[:3, 2].copy()
    fine_range = delta_norm * 0.1 if delta_norm > 1e-6 else 0.1
    
    for dx in np.linspace(-fine_range, fine_range, 11):
        for dy in np.linspace(-fine_range, fine_range, 11):
            test_pos = best_pos + dx * best_x + dy * best_y
            test_center = build_center_pose(test_pos, best_x, best_y, best_z)
            score = score_center(test_center)
            if score < best_score:
                best_score = score
                best_center = test_center
    
    # Optimization 5: Fine-tune rotation around Z
    best_pos = best_center[:3, 3].copy()
    best_x = best_center[:3, 0].copy()
    best_y = best_center[:3, 1].copy()
    best_z = best_center[:3, 2].copy()
    for angle_deg in np.linspace(-10, 10, 21):
        angle_rad = np.deg2rad(angle_deg)
        rot_z = Rotation.from_rotvec(angle_rad * best_z).as_matrix()
        new_x = rot_z @ best_x
        new_y = rot_z @ best_y
        test_center = build_center_pose(best_pos, new_x, new_y, best_z)
        score = score_center(test_center)
        if score < best_score:
            best_score = score
            best_center = test_center
    
    if verbose:
        print(f"    Optimized score: {best_score:.6f}")
    
    return best_center


def _optimize_symmetric_fingers(center_pose, left_rel, finger_pts, left_pts, right_pts, 
                                 tip_axis="z", icp_iters=50, verbose=True,
                                 show_stages=False, no_finger_aligned=None, assem=None, finger=None):
    """Jointly optimize center pose and left finger relative position.
    
    This optimizes both the center frame and the finger position together,
    using a combined score from both left and right finger alignment.
    The right finger is always the mirror of the left through the center.
    
    Key constraints:
    1. When adjusting center pose, left finger stays fixed in world coordinates
       (we adjust left_rel to compensate)
    2. Minimum distance constraint: fingers cannot get closer than their width
    
    This approach respects the user's nudge positioning and refines from there.
    
    Args:
        show_stages: If True, show visualization after each optimization stage
        no_finger_aligned, assem, finger: Meshes needed for stage visualization
    """
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    
    # Estimate finger width from the finger mesh points
    finger_bbox = finger_pts.max(axis=0) - finger_pts.min(axis=0)
    min_finger_width = min(finger_bbox)  # Smallest dimension is likely the width
    min_distance = min_finger_width * 1.2  # Minimum distance between finger and center
    
    if verbose:
        print(f"    Finger dimensions: {finger_bbox}, min separation: {min_distance:.4f}")
    
    def check_separation(center, l_rel):
        """Check if fingers have minimum separation from center."""
        # Distance from left finger to center
        left_dist = np.linalg.norm(l_rel[:3, 3])
        return left_dist >= min_distance
    
    def compute_score(center, l_rel):
        """Compute combined score for both fingers."""
        # Check minimum distance constraint
        if not check_separation(center, l_rel):
            return float('inf'), float('inf'), float('inf')
        
        left_tf = center @ l_rel
        right_tf = center @ (mirror @ l_rel)
        
        # Transform finger points
        left_transformed = finger_pts @ left_tf[:3, :3].T + left_tf[:3, 3]
        right_transformed = finger_pts @ right_tf[:3, :3].T + right_tf[:3, 3]
        
        # Score is average of both distances
        left_score = _mean_nn_distance(left_transformed, left_pts, max_samples=2000)
        right_score = _mean_nn_distance(right_transformed, right_pts, max_samples=2000)
        
        return (left_score + right_score) / 2, left_score, right_score
    
    def show_stage(stage_name, center, l_rel):
        """Display current state for debugging."""
        if not show_stages or finger is None:
            return
        left_tf = center @ l_rel
        right_tf = center @ (mirror @ l_rel)
        
        scene = trimesh.Scene()
        if no_finger_aligned is not None:
            scene.add_geometry(_colorize(no_finger_aligned, [0.5, 0.5, 0.5, 0.3]), node_name="base")
        if assem is not None:
            scene.add_geometry(_colorize(assem, [0.2, 0.5, 0.8, 0.15]), node_name="assem")
        scene.add_geometry(_colorize(finger, [0.9, 0.3, 0.3, 0.9]), node_name="finger_left", transform=left_tf)
        scene.add_geometry(_colorize(finger, [0.3, 0.9, 0.3, 0.9]), node_name="finger_right", transform=right_tf)
        
        # Add target points for reference
        left_colors = np.tile([255, 100, 100, 200], (len(left_pts), 1)).astype(np.uint8)
        right_colors = np.tile([100, 255, 100, 200], (len(right_pts), 1)).astype(np.uint8)
        scene.add_geometry(trimesh.PointCloud(left_pts, colors=left_colors), node_name="left_target")
        scene.add_geometry(trimesh.PointCloud(right_pts, colors=right_colors), node_name="right_target")
        
        print(f"\n  → Stage: {stage_name}")
        print(f"    Close viewer to continue...")
        scene.show()
    
    best_center = center_pose.copy()
    best_left_rel = left_rel.copy()
    best_score, best_left_s, best_right_s = compute_score(best_center, best_left_rel)
    
    # Store the world-frame left finger transform (this should stay fixed during center adjustments)
    left_tf_world = best_center @ best_left_rel
    
    if verbose:
        print(f"    Initial combined score: {best_score:.6f} (L:{best_left_s:.6f}, R:{best_right_s:.6f})")
    
    show_stage("Initial (before optimization)", best_center, best_left_rel)
    
    # Get current finger positions for scale reference
    right_tf = best_center @ (mirror @ best_left_rel)
    finger_distance = np.linalg.norm(left_tf_world[:3, 3] - right_tf[:3, 3])
    
    # ========================================================================
    # STAGE 1: Optimize left finger independently first (direct ICP refinement)
    # ========================================================================
    if verbose:
        print(f"\n    --- Stage 1: Refine left finger alignment ---")
    
    # Run ICP to refine left finger against left target points
    left_tf_refined = _run_icp(
        finger_pts, left_pts, 
        initial=left_tf_world, 
        max_iterations=min(50, icp_iters)
    )
    left_tf_refined = _rigid_transform(left_tf_refined)
    
    # Update left_rel based on refined world position
    best_left_rel = np.linalg.inv(best_center) @ left_tf_refined
    left_tf_world = best_center @ best_left_rel
    
    score_after_left_icp, ls, rs = compute_score(best_center, best_left_rel)
    if verbose:
        print(f"      After left ICP: {score_after_left_icp:.6f} (L:{ls:.6f}, R:{rs:.6f})")
    
    # Only keep if it improved
    if score_after_left_icp < best_score:
        best_score, best_left_s, best_right_s = score_after_left_icp, ls, rs
    else:
        # Revert
        best_left_rel = left_rel.copy()
        left_tf_world = best_center @ best_left_rel
        if verbose:
            print(f"      Reverted (score got worse)")
    
    show_stage("After Stage 1 (left ICP)", best_center, best_left_rel)
    
    # ========================================================================
    # STAGE 2: Optimize center pose to improve right finger (left stays fixed)
    # ========================================================================
    if verbose:
        print(f"\n    --- Stage 2: Optimize center for right finger ---")
    
    t_step = max(finger_distance * 0.05, 0.002)
    r_step = 5.0
    
    for sub_pass in range(2):
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            
            # Adjust center position
            for axis in range(3):
                for sign in [-1, 1]:
                    test_center = best_center.copy()
                    delta = np.zeros(3)
                    delta[axis] = sign * t_step
                    test_center[:3, 3] += delta
                    
                    # Keep left finger in place
                    test_left_rel = np.linalg.inv(test_center) @ left_tf_world
                    
                    score, ls, rs = compute_score(test_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_center = test_center
                        best_left_rel = test_left_rel
                        improved = True
            
            # Adjust center rotation
            for axis in range(3):
                for sign in [-1, 1]:
                    axis_vec = best_center[:3, axis]
                    rot = Rotation.from_rotvec(np.deg2rad(sign * r_step) * axis_vec).as_matrix()
                    
                    test_center = best_center.copy()
                    test_center[:3, :3] = rot @ best_center[:3, :3]
                    
                    test_left_rel = np.linalg.inv(test_center) @ left_tf_world
                    
                    score, ls, rs = compute_score(test_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_center = test_center
                        best_left_rel = test_left_rel
                        improved = True
        
        t_step *= 0.5
        r_step *= 0.5
    
    if verbose:
        print(f"      After center opt: {best_score:.6f} (L:{best_left_s:.6f}, R:{best_right_s:.6f})")
    
    show_stage("After Stage 2 (center optimization)", best_center, best_left_rel)
    
    # ========================================================================
    # STAGE 3: Joint optimization - balance both fingers together
    # ========================================================================
    if verbose:
        print(f"\n    --- Stage 3: Joint optimization (balance both) ---")
    
    t_step = max(finger_distance * 0.02, 0.001)
    r_step = 2.0
    
    for pass_idx in range(3):
        improved = True
        iterations = 0
        max_iters_per_pass = 100
        
        while improved and iterations < max_iters_per_pass:
            improved = False
            iterations += 1
            
            # === Optimize center pose (keeping left finger fixed in world) ===
            for axis in range(3):
                for sign in [-1, 1]:
                    test_center = best_center.copy()
                    delta = np.zeros(3)
                    delta[axis] = sign * t_step
                    test_center[:3, 3] += delta
                    
                    test_left_rel = np.linalg.inv(test_center) @ left_tf_world
                    
                    score, ls, rs = compute_score(test_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_center = test_center
                        best_left_rel = test_left_rel
                        improved = True
            
            # Try center rotation
            for axis in range(3):
                for sign in [-1, 1]:
                    axis_vec = best_center[:3, axis]
                    rot = Rotation.from_rotvec(np.deg2rad(sign * r_step) * axis_vec).as_matrix()
                    
                    test_center = best_center.copy()
                    test_center[:3, :3] = rot @ best_center[:3, :3]
                    
                    test_left_rel = np.linalg.inv(test_center) @ left_tf_world
                    
                    score, ls, rs = compute_score(test_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_center = test_center
                        best_left_rel = test_left_rel
                        improved = True
            
            # === Optimize left finger position (this DOES move left finger in world) ===
            for axis in range(3):
                for sign in [-1, 1]:
                    test_left_rel = best_left_rel.copy()
                    delta = np.zeros(3)
                    delta[axis] = sign * t_step
                    test_left_rel[:3, 3] += delta
                    
                    score, ls, rs = compute_score(best_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_left_rel = test_left_rel
                        left_tf_world = best_center @ best_left_rel
                        improved = True
            
            # Try left finger rotation
            for axis in range(3):
                for sign in [-1, 1]:
                    rot_vec = np.zeros(3)
                    rot_vec[axis] = np.deg2rad(sign * r_step)
                    rot = Rotation.from_rotvec(rot_vec).as_matrix()
                    
                    test_left_rel = best_left_rel.copy()
                    test_left_rel[:3, :3] = best_left_rel[:3, :3] @ rot
                    
                    score, ls, rs = compute_score(best_center, test_left_rel)
                    if score < best_score:
                        best_score, best_left_s, best_right_s = score, ls, rs
                        best_left_rel = test_left_rel
                        left_tf_world = best_center @ best_left_rel
                        improved = True
        
        if verbose:
            print(f"      Pass {pass_idx+1}: score={best_score:.6f} (L:{best_left_s:.6f}, R:{best_right_s:.6f}) iters={iterations}")
        
        t_step *= 0.5
        r_step *= 0.5
    
    show_stage("After Stage 3 (joint optimization)", best_center, best_left_rel)
    
    # ========================================================================
    # STAGE 4: Final ICP refinement for left finger only (if left is worse)
    # ========================================================================
    if best_left_s > best_right_s * 1.2:  # Left is significantly worse than right
        if verbose:
            print(f"\n    --- Stage 4: Extra left finger refinement (L score > R) ---")
        
        # Try small ICP adjustments for left finger
        left_tf_current = best_center @ best_left_rel
        left_tf_refined = _run_icp(
            finger_pts, left_pts,
            initial=left_tf_current,
            max_iterations=30
        )
        left_tf_refined = _rigid_transform(left_tf_refined)
        
        test_left_rel = np.linalg.inv(best_center) @ left_tf_refined
        score, ls, rs = compute_score(best_center, test_left_rel)
        
        if score < best_score:
            best_score, best_left_s, best_right_s = score, ls, rs
            best_left_rel = test_left_rel
            left_tf_world = best_center @ best_left_rel
            if verbose:
                print(f"      After extra left ICP: {best_score:.6f} (L:{best_left_s:.6f}, R:{best_right_s:.6f})")
        else:
            if verbose:
                print(f"      No improvement from extra left ICP")
    
    if verbose:
        final_left_dist = np.linalg.norm(best_left_rel[:3, 3])
        print(f"\n    Final combined score: {best_score:.6f} (L:{best_left_s:.6f}, R:{best_right_s:.6f})")
        print(f"    Final finger-to-center distance: {final_left_dist:.4f}")
    
    show_stage("Final result", best_center, best_left_rel)
    
    return best_center, best_left_rel


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


def _compute_center_pose(t_left, t_right, tip_axis="z"):
    """Compute center pose from two finger transforms (simpler version without residual points)."""
    left_pos = t_left[:3, 3]
    right_pos = t_right[:3, 3]
    delta = left_pos - right_pos
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-6:
        return t_left
    x_axis = delta / delta_norm

    # Average the tip axis vectors from both fingers
    z_axis = _tip_axis_vector(t_left, tip_axis) + _tip_axis_vector(t_right, tip_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1])
    else:
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
            mirror = np.eye(4)
            mirror[0, 0] = -1.0
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_path = _write_calibration_result(
                self.state["save_dir"],
                self.state["quest_path"],
                self.state["finger_path"],
                self.state["assem_path"],
                self.state["t_quest_to_assem"],
                self.state["center_pose"] @ self.state["left_rel"],
                self.state["center_pose"] @ (mirror @ self.state["left_rel"]),
                self.state["tip_axis"],
                self.state["symmetric"],
                True,
                self.state["samples"],
                self.state["icp_iters"],
                self.state["residual_thresh"],
                self.state["split_axis"],
                self.state["split_value"],
                self.state["quest_mesh"],
                self.state["finger_mesh"],
                f"nudge_{ts}",
                json_name=f"nudge_snapshot_{ts}.json",
            )
            print(f"Saved nudge snapshot: {snapshot_path}")
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


def _write_calibration_result(out_dir, quest_path, finger_path, assem_path,
                              t_quest_to_assem, t_finger_left, t_finger_right,
                              tip_axis, symmetric, nudge, samples, icp_iters,
                              residual_thresh, split_axis, split_value,
                              quest_mesh, finger_mesh, suffix,
                              json_name=None):
    os.makedirs(out_dir, exist_ok=True)
    t_quest_to_assem_inv = np.linalg.inv(t_quest_to_assem)
    
    # For backward compatibility, compute old format (quest = no_finger)
    t_finger_left_to_quest = t_quest_to_assem_inv @ t_finger_left
    t_finger_right_to_quest = t_quest_to_assem_inv @ t_finger_right
    
    # For visualization: quest is the reference frame from the data,
    # so transform_quest_to_gripper_base is identity
    t_quest_to_gripper_base = np.eye(4, dtype=np.float64)
    
    # Transform fingers directly to gripper_base (which equals quest frame)
    # So gripper_base_to_finger is the same as quest_to_finger
    t_gripper_base_to_left_finger = t_finger_left_to_quest.copy()
    t_gripper_base_to_right_finger = t_finger_right_to_quest.copy()

    quest_aligned = quest_mesh.copy()
    quest_aligned.apply_transform(t_quest_to_assem)
    quest_out = os.path.join(out_dir, f"quest_aligned_{suffix}.stl")
    quest_aligned.export(quest_out)

    left_mesh = finger_mesh.copy()
    left_mesh.apply_transform(t_finger_left)
    left_path = os.path.join(out_dir, f"finger_left_aligned_{suffix}.stl")
    left_mesh.export(left_path)

    right_mesh = finger_mesh.copy()
    right_mesh.apply_transform(t_finger_right)
    right_path = os.path.join(out_dir, f"finger_right_aligned_{suffix}.stl")
    right_mesh.export(right_path)

    result = {
        "quest_path": quest_path,
        "finger_path": finger_path,
        "assem_path": assem_path,
        "samples": samples,
        "icp_iters": icp_iters,
        "residual_thresh": residual_thresh,
        "split_axis": split_axis,
        "split_value": split_value,
        "transform_quest_to_assem": t_quest_to_assem.tolist(),
        "transform_finger_left": t_finger_left.tolist(),
        "transform_finger_right": t_finger_right.tolist(),
        # New hierarchy: quest -> gripper_base -> fingers
        "transform_quest_to_gripper_base": t_quest_to_gripper_base.tolist(),
        "transform_gripper_base_to_left_finger": t_gripper_base_to_left_finger.tolist(),
        "transform_gripper_base_to_right_finger": t_gripper_base_to_right_finger.tolist(),
        # Backward compatibility (old naming: no_finger = quest)
        "transform_finger_left_to_no_finger": t_finger_left_to_quest.tolist(),
        "transform_finger_right_to_no_finger": t_finger_right_to_quest.tolist(),
        "transform_finger_base_to_no_finger": t_quest_to_gripper_base.tolist(),
        "tip_axis": tip_axis,
        "symmetric": symmetric,
        "nudge": nudge,
        "outputs": {
            "quest_aligned": quest_out,
            "finger_left_aligned": left_path,
            "finger_right_aligned": right_path,
        },
    }

    json_path = os.path.join(out_dir, json_name or f"calibration_result_{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return json_path

def _nudge_initial_loop(quest_aligned, assem, finger, center_pose, left_rel, inward_axis="x",
                        save_dir="src/finger_calibrate/output", tip_axis="z",
                        quest_path="", finger_path="", assem_path="",
                        t_quest_to_assem=None, symmetric=True, samples=0,
                        icp_iters=0, residual_thresh=0.0, split_axis="y",
                        split_value=0.0, align_inward=True):
    """Interactive nudge loop for adjusting finger positions.
    
    Args:
        quest_aligned: The quest (gripper base without fingers) mesh aligned to assem
        align_inward: If True, align the finger's inward axis on entry.
                     Set to False when entering nudge after optimization
                     to preserve current positions.
    """
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

    # Only align inward axis on initial entry, not after optimization rounds
    if align_inward:
        inward_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        left_rel = _align_local_axis(left_rel, inward_axis, inward_dir)

    left_tf = center_pose @ left_rel
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    right_tf = center_pose @ mirror @ left_rel

    scene = trimesh.Scene()
    scene.add_geometry(_colorize(quest_aligned, [0.7, 0.7, 0.7, 0.9]), node_name="quest_base")
    scene.add_geometry(_colorize(assem, [0.2, 0.6, 0.9, 0.35]), node_name="assem")
    scene.add_geometry(_colorize(finger, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left", transform=left_tf)
    scene.add_geometry(_colorize(finger, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right", transform=right_tf)
    axis_mesh = _thick_axis_mesh(scale=axis_scale)
    scene.add_geometry(axis_mesh.copy(), node_name="axis_quest", transform=_center_pose_for_mesh(quest_aligned))
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
        "quest_path": quest_path,
        "finger_path": finger_path,
        "assem_path": assem_path,
        "t_quest_to_assem": t_quest_to_assem if t_quest_to_assem is not None else np.eye(4),
        "symmetric": symmetric,
        "samples": samples,
        "icp_iters": icp_iters,
        "residual_thresh": residual_thresh,
        "split_axis": split_axis,
        "split_value": split_value,
        "quest_mesh": quest_aligned,
        "finger_mesh": finger,
    }
    viewer = _NudgeViewer(scene, state)
    pyglet.app.run()
    if not viewer.accepted:
        return None, None
    return state["center_pose"], state["left_rel"]


class _RoundViewer(SceneViewer):
    """Interactive viewer for round results with keypress handling."""
    
    def __init__(self, scene, round_idx):
        self.round_idx = round_idx
        self.choice = None  # Will be 'c', 's', or 'd'
        super().__init__(scene, start_loop=False, smooth=False)
    
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet_key.C:
            self.choice = 'c'
            self.close()
            pyglet.app.exit()
            return
        if symbol == pyglet_key.S:
            self.choice = 's'
            self.close()
            pyglet.app.exit()
            return
        if symbol == pyglet_key.D or symbol == pyglet_key.Q or symbol == pyglet_key.ESCAPE:
            self.choice = 'd'
            self.close()
            pyglet.app.exit()
            return
        # Let parent handle other keys (camera controls etc)
        super().on_key_press(symbol, modifiers)


def _pre_optimization_prompt(round_idx, center_pose, left_rel, quest_aligned, assem, finger,
                             out_dir, quest_path, finger_path, assem_path, t_quest_to_assem,
                             tip_axis, symmetric, samples, icp_iters, residual_thresh,
                             split_axis, split_value):
    """Display current state BEFORE optimization and prompt user.
    
    Allows user to:
    - Continue with optimization
    - Save current state and quit (without running optimization)
    - Discard and quit
    """
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    left_tf = center_pose @ left_rel
    right_tf = center_pose @ (mirror @ left_rel)
    
    round_scene = trimesh.Scene()
    round_scene.add_geometry(_colorize(quest_aligned, [0.7, 0.7, 0.7, 0.6]), node_name="quest_base")
    round_scene.add_geometry(_colorize(assem, [0.2, 0.6, 0.9, 0.2]), node_name="assem")
    round_scene.add_geometry(_colorize(finger, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left", transform=left_tf)
    round_scene.add_geometry(_colorize(finger, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right", transform=right_tf)
    
    print(f"\n→ BEFORE Round {round_idx} optimization. Press key in viewer window:")
    print("  [C] continue with optimization")
    print("  [S] SAVE current state and quit (skip optimization)")
    print("  [D/Q/Esc] discard and quit")
    
    viewer = _RoundViewer(round_scene, round_idx)
    pyglet.app.run()
    
    choice = viewer.choice or 'd'
    
    if choice == 's':
        # Save current state as final result
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = _write_calibration_result(
            out_dir,
            quest_path,
            finger_path,
            assem_path,
            t_quest_to_assem,
            left_tf,
            right_tf,
            tip_axis,
            symmetric,
            True,  # nudge=True since we're saving from interactive mode
            samples,
            icp_iters,
            residual_thresh,
            split_axis,
            split_value,
            quest_aligned,
            finger,
            f"pre_opt_r{round_idx}_{ts}",
            json_name="calibration_result.json",
        )
        print(f"  ✓ Saved pre-optimization state to: {json_path}")
        return 'saved'
    
    return choice


def _round_prompt(round_idx, left_tf, right_tf, quest_aligned, assem, finger):
    """Display optimization round result and prompt user for next action.
    
    Uses interactive viewer with keypress handling (no need to close window first).
    """
    round_scene = trimesh.Scene()
    round_scene.add_geometry(_colorize(quest_aligned, [0.7, 0.7, 0.7, 0.6]), node_name="quest_base")
    round_scene.add_geometry(_colorize(assem, [0.2, 0.6, 0.9, 0.2]), node_name="assem")
    # Use transform= parameter for consistency with nudge mode visualization
    round_scene.add_geometry(_colorize(finger, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left", transform=left_tf)
    round_scene.add_geometry(_colorize(finger, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right", transform=right_tf)
    
    print(f"\n→ Round {round_idx} result. Press key in viewer window:")
    print("  [C] continue optimization")
    print("  [S] save and finish")
    print("  [D/Q/Esc] discard and quit")
    
    viewer = _RoundViewer(round_scene, round_idx)
    pyglet.app.run()
    
    return viewer.choice or 'd'


def calibrate(quest_path, finger_path, assem_path, out_dir,
              samples=6000, icp_iters=1500, residual_thresh=0.1,
              split_axis="y", split_value=0.0,
              tip_axis="z", symmetric=True, nudge=True,
              inward_axis="x", keep_best=True,
              cont_rounds=1,
              interactive_rounds=True, max_samples=6000,
              max_target_samples=8000, score_samples=2000,
              max_combined_target=12000, show_stages=False):
    
    print("="*70)
    print("FINGER CALIBRATION - Enhanced Multi-Stage Optimization")
    print("="*70)
    
    # Load meshes
    print("\n[1/7] Loading meshes...")
    quest = trimesh.load(quest_path)
    finger = trimesh.load(finger_path)
    assem = trimesh.load(assem_path)
    print(f"  ✓ quest (gripper base): {len(quest.vertices)} vertices")
    print(f"  ✓ finger: {len(finger.vertices)} vertices")
    print(f"  ✓ assem: {len(assem.vertices)} vertices")

    # Sample points
    print(f"\n[2/7] Sampling {samples} points from each mesh...")
    sample_count = int(min(samples, max_samples))
    quest_pts = _sample_points(quest, sample_count)
    assem_pts = _sample_points(assem, sample_count)
    finger_pts = _sample_points(finger, sample_count)
    print(f"  ✓ Sampled {len(quest_pts)} points from quest")
    print(f"  ✓ Sampled {len(assem_pts)} points from assem")
    print(f"  ✓ Sampled {len(finger_pts)} points from finger")

    # Align quest to assem
    print(f"\n[3/7] Aligning quest mesh to assembled reference...")
    print(f"  ICP iterations: {icp_iters}")
    t_quest_to_assem = _rigid_transform(_run_icp(quest_pts, assem_pts, max_iterations=icp_iters))
    quest_aligned = quest.copy()
    quest_aligned.apply_transform(t_quest_to_assem)
    alignment_score = _mean_nn_distance(quest_pts @ t_quest_to_assem[:3, :3].T + t_quest_to_assem[:3, 3], assem_pts)
    print(f"  ✓ Alignment score: {alignment_score:.6f}")

    # Extract finger-only points (subtract quest from assem to get just the finger regions)
    print(f"\n[4/8] Extracting finger-only points (residual_thresh={residual_thresh})...")
    print("  Using aligned quest mesh to identify finger regions only")
    # Extract finger-only points (subtract quest from assem to get just the finger regions)
    print(f"\n[4/8] Extracting finger-only points (residual_thresh={residual_thresh})...")
    print("  Using aligned quest mesh to identify finger regions only")
    finger_only_pts = _extract_finger_only_points(assem_pts, quest_aligned, residual_thresh, verbose=True)
    if finger_only_pts.size == 0:
        raise RuntimeError("❌ No finger-only points found! Try lowering --residual-thresh")
    
    if finger_only_pts.shape[0] > max_target_samples:
        idx = np.random.choice(finger_only_pts.shape[0], max_target_samples, replace=False)
        finger_only_pts = finger_only_pts[idx]
        print(f"  ✓ Downsampled to {len(finger_only_pts)} points")

    # Cluster into left and right using spatial connectivity
    print(f"\n  Clustering finger points into two separate fingers...")
    try:
        left_pts, right_pts = _cluster_finger_points(finger_only_pts, verbose=True)
    except RuntimeError as e:
        print(f"  ⚠ Clustering failed: {e}")
        print(f"  Falling back to axis-based split (axis={split_axis}, value={split_value})")
        left_pts, right_pts = _split_points(finger_only_pts, axis=split_axis, value=split_value)
        if left_pts.size == 0 or right_pts.size == 0:
            print("  ⚠ Warning: One side empty, using all finger-only points for both")
            left_pts = finger_only_pts
            right_pts = finger_only_pts
    
    # Interactive finger selection - let user confirm which side is left
    print(f"\n[5/8] Select which cluster is the LEFT finger...")
    left_pts, right_pts = _select_left_finger(left_pts, right_pts, quest_aligned, assem)
    if left_pts is None:
        print("❌ Calibration canceled by user during finger selection")
        return None

    # Initial alignment - new approach:
    # 1. Set center frame at midpoint of finger-only point clouds
    # 2. Align left finger to left points
    # 3. Optimize center pose so mirrored right matches right points
    print(f"\n[6/8] Computing initial finger alignment...")
    
    print("  Stage 1: Initial center frame from finger-only point centroids")
    center_pose_assem = _initial_center_from_finger_points(left_pts, right_pts, tip_axis=tip_axis)
    print(f"    Center position: {center_pose_assem[:3, 3]}")
    
    print("  Stage 2: Principal component alignment for left finger")
    init_left = _initial_align(finger_pts, left_pts, verbose=True)
    
    print("  Stage 3: Multi-start ICP to fully merge left finger with left points")
    init_left_refined = _multi_start_icp(finger_pts, left_pts, init_left, max_iterations=min(100, icp_iters), verbose=True)
    
    print("  Stage 4: Optimize center pose for symmetric mirroring")
    center_pose_assem = _optimize_center_pose(init_left_refined, finger_pts, right_pts, tip_axis=tip_axis, verbose=True)
    
    left_rel = np.linalg.inv(center_pose_assem) @ init_left_refined
    
    # Interactive nudging
    if nudge:
        print(f"\n[7/8] Interactive positioning (nudge mode)...")
        print("  → Opening 3D viewer for manual adjustment")
        print("  → Press ENTER when satisfied to start optimization")
        center_pose_assem, left_rel = _nudge_initial_loop(
            quest_aligned,
            assem,
            finger,
            center_pose_assem,
            left_rel,
            inward_axis=inward_axis,
            save_dir=out_dir,
            tip_axis=tip_axis,
            quest_path=quest_path,
            finger_path=finger_path,
            assem_path=assem_path,
            t_quest_to_assem=t_quest_to_assem,
            symmetric=symmetric,
            samples=samples,
            icp_iters=icp_iters,
            residual_thresh=residual_thresh,
            split_axis=split_axis,
            split_value=split_value,
        )
        if center_pose_assem is None:
            print("❌ Calibration canceled by user")
            return None
    else:
        print(f"\n[7/8] Skipping interactive nudge (--no-nudge)")
    
    initial_left = center_pose_assem @ left_rel
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    initial_right = center_pose_assem @ (mirror @ left_rel)

    # Main optimization
    print(f"\n[8/8] Running optimization...")
    
    if symmetric:
        print(f"  Mode: Symmetric (joint optimization of both fingers)")
        print(f"  Optimization rounds: {cont_rounds}")
        round_idx = 0
        
        # Store the best result from nudge as initial
        best_center = center_pose_assem.copy()
        best_left_rel = left_rel.copy()
        
        while True:
            round_idx += 1
            print(f"\n  --- Round {round_idx} ---")
            
            # Show pre-optimization state and allow user to save/quit before optimization
            if interactive_rounds:
                pre_choice = _pre_optimization_prompt(
                    round_idx, best_center, best_left_rel,
                    quest_aligned, assem, finger,
                    out_dir, quest_path, finger_path, assem_path, t_quest_to_assem,
                    tip_axis, symmetric, samples, icp_iters, residual_thresh,
                    split_axis, split_value,
                )
                if pre_choice == 'saved':
                    print("  ✓ Saved current state, skipping optimization")
                    # Return the path to indicate success
                    return os.path.join(out_dir, "calibration_result.json")
                if pre_choice == 'd':
                    print("❌ Discarded by user before optimization")
                    return None
                # pre_choice == 'c' means continue with optimization
            
            # Joint optimization: optimize center_pose and left_rel together
            # Score is combined distance of both fingers to their targets
            print(f"  Joint optimization of center pose and finger position...")
            
            best_center, best_left_rel = _optimize_symmetric_fingers(
                best_center,
                best_left_rel,
                finger_pts,
                left_pts,
                right_pts,
                tip_axis=tip_axis,
                icp_iters=min(50, icp_iters),
                verbose=True,
                show_stages=show_stages,
                no_finger_aligned=quest_aligned,
                assem=assem,
                finger=finger,
            )
            
            # Compute final transforms
            t_finger_left = best_center @ best_left_rel
            t_finger_right = best_center @ (mirror @ best_left_rel)
            
            # Compute scores
            transformed_left = finger_pts @ t_finger_left[:3, :3].T + t_finger_left[:3, 3]
            left_score = _mean_nn_distance(transformed_left, left_pts, max_samples=score_samples)
            transformed_right = finger_pts @ t_finger_right[:3, :3].T + t_finger_right[:3, 3]
            right_score = _mean_nn_distance(transformed_right, right_pts, max_samples=score_samples)
            print(f"  ✓ Round {round_idx} scores: L={left_score:.6f}, R={right_score:.6f}, combined={(left_score + right_score) / 2:.6f}")
            
            if not interactive_rounds and round_idx >= cont_rounds:
                break
            
            if interactive_rounds:
                # Update state variables for nudge
                center_pose_assem = best_center
                left_rel = best_left_rel
                
                choice = _round_prompt(round_idx, t_finger_left, t_finger_right, quest_aligned, assem, finger)
                if choice == "s":
                    break
                if choice == "d":
                    print("❌ Result discarded by user")
                    return None
                if nudge:
                    # Pass current state to nudge - positions will be consistent
                    # align_inward=False to preserve current positions after optimization
                    center_pose_assem, left_rel = _nudge_initial_loop(
                        quest_aligned,
                        assem,
                        finger,
                        center_pose_assem,
                        left_rel,
                        inward_axis=inward_axis,
                        save_dir=out_dir,
                        tip_axis=tip_axis,
                        quest_path=quest_path,
                        finger_path=finger_path,
                        assem_path=assem_path,
                        t_quest_to_assem=t_quest_to_assem,
                        symmetric=symmetric,
                        samples=samples,
                        icp_iters=icp_iters,
                        residual_thresh=residual_thresh,
                        split_axis=split_axis,
                        split_value=split_value,
                        align_inward=False,
                    )
                    if center_pose_assem is None:
                        return None
                    # Update best_center and best_left_rel from nudge
                    best_center = center_pose_assem
                    best_left_rel = left_rel
                continue
        
        # Final state
        center_pose_assem = best_center
        left_rel = best_left_rel
        t_finger_left = best_center @ best_left_rel
        t_finger_right = best_center @ (mirror @ best_left_rel)
    else:
        print(f"  Mode: Independent (separate ICP for each finger)")
        init_right = _initial_align(finger_pts, right_pts, verbose=True)
        t_finger_left = _multi_start_icp(finger_pts, left_pts, initial_left, icp_iters, verbose=True)
        t_finger_right = _multi_start_icp(finger_pts, right_pts, init_right, icp_iters, verbose=True)

    t_finger_left_opt = t_finger_left
    t_finger_right_opt = t_finger_right
    if keep_best:
        if symmetric:
            # Compare combined scores (left + right)
            initial_left_score = _mean_nn_distance(
                finger_pts @ initial_left[:3, :3].T + initial_left[:3, 3],
                left_pts,
                max_samples=score_samples,
            )
            initial_right_score = _mean_nn_distance(
                finger_pts @ initial_right[:3, :3].T + initial_right[:3, 3],
                right_pts,
                max_samples=score_samples,
            )
            initial_score = initial_left_score + initial_right_score
            
            opt_left_score = _mean_nn_distance(
                finger_pts @ t_finger_left[:3, :3].T + t_finger_left[:3, 3],
                left_pts,
                max_samples=score_samples,
            )
            opt_right_score = _mean_nn_distance(
                finger_pts @ t_finger_right[:3, :3].T + t_finger_right[:3, 3],
                right_pts,
                max_samples=score_samples,
            )
            opt_score = opt_left_score + opt_right_score
            
            print(f"keep_best (symmetric) initial={initial_score:.6f} opt={opt_score:.6f}")
            if opt_score > initial_score:
                print("Optimization increased residuals; keeping initial pose.")
                t_finger_left = initial_left
                t_finger_right = initial_right
        else:
            initial_left_dist = _mean_nn_distance(
                finger_pts @ initial_left[:3, :3].T + initial_left[:3, 3],
                left_pts,
                max_samples=score_samples,
            )
            opt_left_dist = _mean_nn_distance(
                finger_pts @ t_finger_left[:3, :3].T + t_finger_left[:3, 3],
                left_pts,
                max_samples=score_samples,
            )
            initial_right_dist = _mean_nn_distance(
                finger_pts @ initial_right[:3, :3].T + initial_right[:3, 3],
                right_pts,
                max_samples=score_samples,
            )
            opt_right_dist = _mean_nn_distance(
                finger_pts @ t_finger_right[:3, :3].T + t_finger_right[:3, 3],
                right_pts,
                max_samples=score_samples,
            )
            print(
                f"keep_best left initial={initial_left_dist:.6f} opt={opt_left_dist:.6f} "
                f"right initial={initial_right_dist:.6f} opt={opt_right_dist:.6f}"
            )
            if opt_left_dist + opt_right_dist > initial_left_dist + initial_right_dist:
                print("Optimization increased residuals; keeping initial pose.")
                t_finger_left = initial_left
                t_finger_right = initial_right

    if interactive_rounds and not symmetric:
        choice = _round_prompt("final", t_finger_left, t_finger_right, quest_aligned, assem, finger)
        if choice == "d":
            return None

    t_quest_to_assem_inv = np.linalg.inv(t_quest_to_assem)
    t_finger_left_to_quest = t_quest_to_assem_inv @ t_finger_left
    t_finger_right_to_quest = t_quest_to_assem_inv @ t_finger_right
    t_finger_base_to_quest = t_quest_to_assem_inv @ center_pose_assem

    if keep_best:
        if t_finger_left_opt is not t_finger_left:
            # Show optimized pose using same style as nudge mode for consistency
            opt_scene = trimesh.Scene()
            opt_scene.add_geometry(_colorize(quest_aligned, [0.7, 0.7, 0.7, 0.6]), node_name="quest_base")
            opt_scene.add_geometry(_colorize(assem, [0.2, 0.6, 0.9, 0.2]), node_name="assem")
            opt_scene.add_geometry(_colorize(finger, [0.9, 0.4, 0.4, 0.95]), node_name="finger_left", transform=t_finger_left_opt)
            opt_scene.add_geometry(_colorize(finger, [0.4, 0.9, 0.4, 0.95]), node_name="finger_right", transform=t_finger_right_opt)
            print("Showing optimized pose (kept initial due to worse score). Close window to continue.")
            opt_scene.show()

    json_path = _write_calibration_result(
        out_dir,
        quest_path,
        finger_path,
        assem_path,
        t_quest_to_assem,
        t_finger_left,
        t_finger_right,
        tip_axis,
        symmetric,
        nudge,
        samples,
        icp_iters,
        residual_thresh,
        split_axis,
        split_value,
        quest,
        finger,
        "final",
        json_name="calibration_result.json",
    )

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate finger meshes against an assembled reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic calibration with default settings
  python calibrate_finger.py
  
  # Lower residual threshold if fingers are very close to base
  python calibrate_finger.py --residual-thresh 0.3
  
  # Adjust split axis if fingers are oriented differently
  python calibrate_finger.py --split-axis x --split-value 0.0
  
  # Disable interactive mode for batch processing
  python calibrate_finger.py --no-nudge --no-interactive-rounds
  
  # Increase samples for higher accuracy (slower)
  python calibrate_finger.py --samples 10000 --icp-iters 150
        """
    )
    parser.add_argument("--quest", "--no-finger", dest="quest", default="src/meshes/left_no_finger.STL",
                        help="Path to quest/gripper base mesh (without fingers)")
    parser.add_argument("--finger", default="src/meshes/finger.STL",
                        help="Path to single finger mesh")
    parser.add_argument("--assem", default="src/meshes/left_assem.STL",
                        help="Path to assembled gripper mesh (base + both fingers)")
    parser.add_argument("--out", default="src/finger_calibrate/output",
                        help="Output directory for calibration results")
    parser.add_argument("--samples", type=int, default=6000,
                        help="Number of points to sample from each mesh (default: 6000)")
    parser.add_argument("--icp-iters", type=int, default=100,
                        help="ICP iterations per optimization (default: 100, was 80)")
    parser.add_argument("--residual-thresh", type=float, default=0.5,
                        help="Distance threshold to identify finger regions (default: 0.5mm)")
    parser.add_argument("--split-axis", choices=["x", "y", "z"], default="y",
                        help="Axis to split left/right fingers (default: y)")
    parser.add_argument("--split-value", type=float, default=0.0,
                        help="Value along split-axis to divide left/right (default: 0.0)")
    parser.add_argument("--tip-axis", choices=["x", "-x", "y", "-y", "z", "-z"], default="z",
                        help="Local axis pointing toward fingertip (default: z)")
    parser.add_argument("--inward-axis", choices=["x", "-x", "y", "-y", "z", "-z"], default="x",
                        help="Local axis pointing inward toward center (default: x)")
    parser.add_argument("--no-symmetric", action="store_true",
                        help="Disable symmetric solve (run separate ICP for each finger)")
    parser.add_argument("--no-nudge", action="store_false", dest="nudge", default=True,
                        help="Disable interactive positioning before optimization")
    parser.add_argument("--no-keep-best", action="store_true",
                        help="Allow worse ICP results (disable quality check)")
    parser.add_argument("--continue-rounds", type=int, default=3,
                        help="Number of optimization rounds (default: 3, was 1)")
    parser.add_argument("--no-interactive-rounds", action="store_false", dest="interactive_rounds", default=True,
                        help="Disable interactive prompt after each round")
    parser.add_argument("--show-stages", action="store_true",
                        help="Show visualization after each optimization stage (for debugging)")
    parser.add_argument("--max-samples", type=int, default=6000,
                        help="Clamp for mesh point sampling (default: 6000)")
    parser.add_argument("--max-target-samples", type=int, default=8000,
                        help="Clamp for residual target points (default: 8000)")
    parser.add_argument("--score-samples", type=int, default=2000,
                        help="Points used for quality scoring (default: 2000)")
    parser.add_argument("--max-combined-target", type=int, default=12000,
                        help="Clamp for combined symmetric target (default: 12000)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  FINGER CALIBRATION TOOL")
    print("="*70)
    print(f"  quest (gripper base): {args.quest}")
    print(f"  finger:               {args.finger}")
    print(f"  assem:                {args.assem}")
    print(f"  output:               {args.out}")
    print("="*70 + "\n")

    result = calibrate(
        args.quest,
        args.finger,
        args.assem,
        args.out,
        samples=args.samples,
        icp_iters=args.icp_iters,
        residual_thresh=args.residual_thresh,
        split_axis=args.split_axis,
        split_value=args.split_value,
        tip_axis=args.tip_axis,
        symmetric=(not args.no_symmetric),
        nudge=args.nudge,
        inward_axis=args.inward_axis,
        keep_best=(not args.no_keep_best),
        cont_rounds=max(1, args.continue_rounds),
        interactive_rounds=args.interactive_rounds,
        max_samples=max(1000, args.max_samples),
        max_target_samples=max(1000, args.max_target_samples),
        score_samples=max(500, args.score_samples),
        max_combined_target=max(1000, args.max_combined_target),
        show_stages=args.show_stages,
    )
    
    if result:
        print("\n" + "="*70)
        print("  ✓ CALIBRATION COMPLETE")
        print("="*70)
        print(f"  Saved to: {args.out}/calibration_result.json")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("  ✗ CALIBRATION FAILED OR CANCELED")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
