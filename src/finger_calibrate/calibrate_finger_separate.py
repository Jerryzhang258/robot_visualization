#!/usr/bin/env python3
"""
Calibrate left and right finger meshes separately (non-symmetric approach).

This script performs independent ICP alignment for each finger without assuming
symmetry or mirroring. Each finger is calibrated using its own mesh file.
"""

import argparse
import json
import os
import datetime

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

# Import helper functions from the original calibrate_finger.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from calibrate_finger import (
    _sample_points,
    _run_icp,
    _rigid_transform,
    _mean_nn_distance,
    _extract_finger_only_points,
    _cluster_finger_points,
    _split_points,
    _initial_align,
    _multi_start_icp,
)


def _select_finger_cluster(cluster1_pts, cluster2_pts, no_finger_aligned, assem, finger_name="LEFT"):
    """Interactive selection of which cluster corresponds to the named finger.
    
    Args:
        cluster1_pts: First cluster of finger points
        cluster2_pts: Second cluster of finger points
        no_finger_aligned: Aligned base mesh (for visualization)
        assem: Assembled reference mesh (for visualization)
        finger_name: Name of finger to select ("LEFT" or "RIGHT")
        
    Returns:
        selected_pts: Points for the selected finger
        other_pts: Points for the other finger
        or (None, None) if user cancels
    """
    print(f"\n{'='*70}")
    print(f"  SELECT {finger_name} FINGER CLUSTER")
    print(f"{'='*70}")
    print(f"  Two finger clusters detected:")
    print(f"    Cluster 1: {len(cluster1_pts)} points, centroid at {cluster1_pts.mean(axis=0)}")
    print(f"    Cluster 2: {len(cluster2_pts)} points, centroid at {cluster2_pts.mean(axis=0)}")
    print()
    
    # Create visualization scene
    scene_meshes = []
    
    # Add base mesh (gray)
    base_colored = no_finger_aligned.copy()
    base_colored.visual.vertex_colors = np.full((len(base_colored.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
    scene_meshes.append(base_colored)
    
    # Add cluster 1 (red points)
    cluster1_cloud = trimesh.PointCloud(cluster1_pts)
    cluster1_cloud.colors = np.full((len(cluster1_pts), 4), [255, 0, 0, 255], dtype=np.uint8)
    scene_meshes.append(cluster1_cloud)
    
    # Add cluster 2 (blue points)
    cluster2_cloud = trimesh.PointCloud(cluster2_pts)
    cluster2_cloud.colors = np.full((len(cluster2_pts), 4), [0, 0, 255, 255], dtype=np.uint8)
    scene_meshes.append(cluster2_cloud)
    
    scene = trimesh.Scene(scene_meshes)
    
    # Show scene and prompt user
    scene.show(caption=f"Select {finger_name} Finger: Red=1, Blue=2")
    
    while True:
        choice = input(f"  Which cluster is the {finger_name} finger? [1=red/2=blue/q=quit]: ").strip().lower()
        if choice == "1":
            return cluster1_pts, cluster2_pts
        elif choice == "2":
            return cluster2_pts, cluster1_pts
        elif choice == "q":
            return None, None
        else:
            print("  Invalid choice. Please enter 1, 2, or q.")


def calibrate_separate_fingers(
    no_finger_path,
    left_finger_path,
    right_finger_path,
    assem_path,
    out_dir,
    samples=6000,
    icp_iters=100,
    residual_thresh=0.5,
    split_axis="y",
    split_value=0.0,
    tip_axis="z",
    max_samples=6000,
    max_target_samples=8000,
    score_samples=2000,
):
    """
    Calibrate left and right fingers independently without symmetry assumptions.
    
    Args:
        no_finger_path: Path to base gripper mesh (without fingers)
        left_finger_path: Path to left finger mesh STL file
        right_finger_path: Path to right finger mesh STL file
        assem_path: Path to assembled gripper reference (base + both fingers)
        out_dir: Output directory for calibration JSON
        samples: Number of points to sample from each mesh
        icp_iters: ICP iterations per optimization
        residual_thresh: Distance threshold to identify finger regions (mm)
        split_axis: Axis for initial left/right split if clustering fails
        split_value: Value along split_axis for initial division
        tip_axis: Local axis pointing toward fingertip
        max_samples: Max points for mesh sampling
        max_target_samples: Max points for target clouds
        score_samples: Points used for quality scoring
        
    Returns:
        Path to output JSON file if successful, None if failed/canceled
    """
    print("\n" + "="*70)
    print("  SEPARATE FINGER CALIBRATION (Non-Symmetric)")
    print("="*70)
    print(f"  Base mesh:    {no_finger_path}")
    print(f"  Left finger:  {left_finger_path}")
    print(f"  Right finger: {right_finger_path}")
    print(f"  Reference:    {assem_path}")
    print(f"  Output:       {out_dir}")
    print("="*70 + "\n")
    
    # Load meshes
    print("[1/8] Loading meshes...")
    no_finger = trimesh.load(no_finger_path)
    left_finger = trimesh.load(left_finger_path)
    right_finger = trimesh.load(right_finger_path)
    assem = trimesh.load(assem_path)
    print(f"  ✓ Base: {len(no_finger.vertices)} vertices")
    print(f"  ✓ Left finger: {len(left_finger.vertices)} vertices")
    print(f"  ✓ Right finger: {len(right_finger.vertices)} vertices")
    print(f"  ✓ Reference: {len(assem.vertices)} vertices")
    
    # Sample points
    print(f"\n[2/8] Sampling {samples} points from each mesh...")
    sample_count = min(samples, max_samples)
    nf_pts = _sample_points(no_finger, sample_count)
    assem_pts = _sample_points(assem, sample_count)
    left_finger_pts = _sample_points(left_finger, sample_count)
    right_finger_pts = _sample_points(right_finger, sample_count)
    print(f"  ✓ Base: {len(nf_pts)} points")
    print(f"  ✓ Assembled: {len(assem_pts)} points")
    print(f"  ✓ Left finger: {len(left_finger_pts)} points")
    print(f"  ✓ Right finger: {len(right_finger_pts)} points")
    
    # Align base mesh to assembled reference
    print(f"\n[3/8] Aligning base mesh to reference...")
    print(f"  ICP iterations: {icp_iters}")
    t_nf_to_assem = _rigid_transform(_run_icp(nf_pts, assem_pts, max_iterations=icp_iters))
    no_finger_aligned = no_finger.copy()
    no_finger_aligned.apply_transform(t_nf_to_assem)
    alignment_score = _mean_nn_distance(
        nf_pts @ t_nf_to_assem[:3, :3].T + t_nf_to_assem[:3, 3],
        assem_pts
    )
    print(f"  ✓ Alignment score: {alignment_score:.6f} mm")
    
    # Extract finger-only points (subtract base from assembled)
    print(f"\n[4/8] Extracting finger-only regions (threshold={residual_thresh} mm)...")
    finger_only_pts = _extract_finger_only_points(
        assem_pts, no_finger_aligned, residual_thresh, verbose=True
    )
    if finger_only_pts.size == 0:
        raise RuntimeError("❌ No finger-only points found! Try lowering --residual-thresh")
    
    if finger_only_pts.shape[0] > max_target_samples:
        idx = np.random.choice(finger_only_pts.shape[0], max_target_samples, replace=False)
        finger_only_pts = finger_only_pts[idx]
        print(f"  ✓ Downsampled to {len(finger_only_pts)} points")
    
    # Cluster into two separate fingers
    print(f"\n[5/8] Clustering finger points into two groups...")
    try:
        cluster1_pts, cluster2_pts = _cluster_finger_points(finger_only_pts, verbose=True)
    except RuntimeError as e:
        print(f"  ⚠ Clustering failed: {e}")
        print(f"  Falling back to axis-based split (axis={split_axis}, value={split_value})")
        cluster1_pts, cluster2_pts = _split_points(finger_only_pts, axis=split_axis, value=split_value)
        if cluster1_pts.size == 0 or cluster2_pts.size == 0:
            raise RuntimeError("❌ Failed to separate fingers! Try adjusting --split-axis or --split-value")
    
    # Interactive selection: user picks which cluster is left finger
    print(f"\n[6/8] Identifying left and right fingers...")
    left_target_pts, right_target_pts = _select_finger_cluster(
        cluster1_pts, cluster2_pts, no_finger_aligned, assem, finger_name="LEFT"
    )
    if left_target_pts is None:
        print("❌ Calibration canceled by user")
        return None
    
    print(f"  ✓ Left finger target: {len(left_target_pts)} points")
    print(f"  ✓ Right finger target: {len(right_target_pts)} points")
    
    # Calibrate LEFT finger
    print(f"\n[7/8] Calibrating LEFT finger...")
    print("  Stage 1: Principal component alignment")
    init_left = _initial_align(left_finger_pts, left_target_pts, verbose=True)
    
    print("  Stage 2: Multi-start ICP (testing multiple orientations)")
    t_left_to_no_finger = _multi_start_icp(
        left_finger_pts, left_target_pts, init_left,
        max_iterations=icp_iters, verbose=True
    )
    
    # Score left finger alignment
    transformed_left = left_finger_pts @ t_left_to_no_finger[:3, :3].T + t_left_to_no_finger[:3, 3]
    left_score = _mean_nn_distance(transformed_left, left_target_pts, max_samples=score_samples)
    print(f"  ✓ Left finger score: {left_score:.6f} mm")
    
    # Calibrate RIGHT finger
    print(f"\n[8/8] Calibrating RIGHT finger...")
    print("  Stage 1: Principal component alignment")
    init_right = _initial_align(right_finger_pts, right_target_pts, verbose=True)
    
    print("  Stage 2: Multi-start ICP (testing multiple orientations)")
    t_right_to_no_finger = _multi_start_icp(
        right_finger_pts, right_target_pts, init_right,
        max_iterations=icp_iters, verbose=True
    )
    
    # Score right finger alignment
    transformed_right = right_finger_pts @ t_right_to_no_finger[:3, :3].T + t_right_to_no_finger[:3, 3]
    right_score = _mean_nn_distance(transformed_right, right_target_pts, max_samples=score_samples)
    print(f"  ✓ Right finger score: {right_score:.6f} mm")
    
    # Display final results
    print(f"\n{'='*70}")
    print("  CALIBRATION RESULTS")
    print(f"{'='*70}")
    print(f"  Left finger:  {left_score:.6f} mm alignment error")
    print(f"  Right finger: {right_score:.6f} mm alignment error")
    print(f"  Average:      {(left_score + right_score) / 2:.6f} mm")
    print(f"{'='*70}\n")
    
    # Save calibration JSON
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "calibration_result.json")
    
    calibration_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "no_finger_path": no_finger_path,
        "left_finger_path": left_finger_path,
        "right_finger_path": right_finger_path,
        "assem_path": assem_path,
        "samples": samples,
        "icp_iters": icp_iters,
        "residual_thresh": residual_thresh,
        "split_axis": split_axis,
        "split_value": split_value,
        "tip_axis": tip_axis,
        "symmetric": False,  # Mark as non-symmetric calibration
        "transform_no_finger_to_assem": t_nf_to_assem.tolist(),
        "transform_left_finger_to_no_finger": t_left_to_no_finger.tolist(),
        "transform_right_finger_to_no_finger": t_right_to_no_finger.tolist(),
        "alignment_scores": {
            "base_to_assem": float(alignment_score),
            "left_finger": float(left_score),
            "right_finger": float(right_score),
            "average": float((left_score + right_score) / 2),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"✓ Saved calibration to: {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate left and right finger meshes separately (non-symmetric).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic calibration with separate finger meshes
  python calibrate_finger_separate.py
  
  # Custom mesh paths
  python calibrate_finger_separate.py \\
      --no-finger src/meshes/left_no_finger.STL \\
      --left-finger src/meshes/left_finger.STL \\
      --right-finger src/meshes/right_finger.STL \\
      --assem src/meshes/left_assem.STL
  
  # Lower residual threshold if fingers are very close to base
  python calibrate_finger_separate.py --residual-thresh 0.3
  
  # Increase samples for higher accuracy (slower)
  python calibrate_finger_separate.py --samples 10000 --icp-iters 150
        """
    )
    parser.add_argument("--no-finger", default="src/meshes/left_no_finger.STL",
                        help="Path to gripper base mesh (without fingers)")
    parser.add_argument("--left-finger", default="src/meshes/left_finger.STL",
                        help="Path to left finger mesh")
    parser.add_argument("--right-finger", default="src/meshes/right_finger.STL",
                        help="Path to right finger mesh")
    parser.add_argument("--assem", default="src/meshes/left_assem.STL",
                        help="Path to assembled gripper mesh (base + both fingers)")
    parser.add_argument("--out", default="src/finger_calibrate/output",
                        help="Output directory for calibration results")
    parser.add_argument("--samples", type=int, default=6000,
                        help="Number of points to sample from each mesh (default: 6000)")
    parser.add_argument("--icp-iters", type=int, default=100,
                        help="ICP iterations per optimization (default: 100)")
    parser.add_argument("--residual-thresh", type=float, default=0.5,
                        help="Distance threshold to identify finger regions (default: 0.5mm)")
    parser.add_argument("--split-axis", choices=["x", "y", "z"], default="y",
                        help="Axis to split left/right fingers if clustering fails (default: y)")
    parser.add_argument("--split-value", type=float, default=0.0,
                        help="Value along split-axis to divide left/right (default: 0.0)")
    parser.add_argument("--tip-axis", choices=["x", "-x", "y", "-y", "z", "-z"], default="z",
                        help="Local axis pointing toward fingertip (default: z)")
    parser.add_argument("--max-samples", type=int, default=6000,
                        help="Clamp for mesh point sampling (default: 6000)")
    parser.add_argument("--max-target-samples", type=int, default=8000,
                        help="Clamp for residual target points (default: 8000)")
    parser.add_argument("--score-samples", type=int, default=2000,
                        help="Points used for quality scoring (default: 2000)")
    args = parser.parse_args()
    
    try:
        result = calibrate_separate_fingers(
            args.no_finger,
            args.left_finger,
            args.right_finger,
            args.assem,
            args.out,
            samples=args.samples,
            icp_iters=args.icp_iters,
            residual_thresh=args.residual_thresh,
            split_axis=args.split_axis,
            split_value=args.split_value,
            tip_axis=args.tip_axis,
            max_samples=max(1000, args.max_samples),
            max_target_samples=max(1000, args.max_target_samples),
            score_samples=max(500, args.score_samples),
        )
        
        if result:
            print("="*70)
            print("  ✓ CALIBRATION COMPLETE")
            print("="*70)
            print(f"  Result saved to: {result}")
            print("="*70 + "\n")
            return 0
        else:
            print("="*70)
            print("  ✗ CALIBRATION FAILED OR CANCELED")
            print("="*70 + "\n")
            return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
