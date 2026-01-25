#!/usr/bin/env python3
"""Verify finger calibration by visualizing the reassembled gripper.

This script loads calibration results and displays the reconstructed gripper
alongside the original assembled mesh for visual comparison.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh


def _load_calibration(json_path):
    """Load calibration data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get transforms (prefer the to_no_finger versions if available)
    if "transform_finger_left_to_no_finger" in data:
        t_left = np.array(data["transform_finger_left_to_no_finger"], dtype=np.float64)
        t_right = np.array(data["transform_finger_right_to_no_finger"], dtype=np.float64)
    else:
        # Fall back to computing from assem-frame transforms
        t_nf_to_assem = np.array(data["transform_no_finger_to_assem"], dtype=np.float64)
        t_left_assem = np.array(data["transform_finger_left"], dtype=np.float64)
        t_right_assem = np.array(data["transform_finger_right"], dtype=np.float64)
        t_nf_to_assem_inv = np.linalg.inv(t_nf_to_assem)
        t_left = t_nf_to_assem_inv @ t_left_assem
        t_right = t_nf_to_assem_inv @ t_right_assem
    
    # Get center/base transform if available
    t_center = None
    if "transform_finger_base_to_no_finger" in data:
        t_center = np.array(data["transform_finger_base_to_no_finger"], dtype=np.float64)
    
    return data, t_left, t_right, t_center


def _colorize(mesh, rgba):
    """Apply color to mesh vertices."""
    colored = mesh.copy()
    rgba_arr = (np.array(rgba) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba_arr, (len(colored.vertices), 1))
    return colored


def _scale_transform(T, scale):
    """Scale the translation component of a transform."""
    out = np.array(T, dtype=np.float64)
    out[:3, 3] *= scale
    return out


def _thick_axis_mesh(scale=0.1):
    """Create colored axis mesh for visualization."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Verify finger calibration by visualizing the reassembled gripper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python verify_calibration.py
  
  # Verify specific calibration file
  python verify_calibration.py --json path/to/calibration_result.json
  
  # Compare with original assembled mesh
  python verify_calibration.py --show-assem
  
  # Show coordinate axes
  python verify_calibration.py --show-axes
"""
    )
    parser.add_argument(
        "--json", 
        default=os.path.join("src", "finger_calibrate", "output", "calibration_result.json"),
        help="Path to calibration JSON file"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=1.0, 
        help="Scale factor for meshes and transforms (default: 1.0)"
    )
    parser.add_argument(
        "--show-assem", 
        action="store_true",
        help="Show original assembled mesh for comparison (semi-transparent)"
    )
    parser.add_argument(
        "--show-axes", 
        action="store_true",
        help="Show coordinate axes at key positions"
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Hide the no_finger base mesh"
    )
    args = parser.parse_args()

    # Check if calibration file exists
    if not os.path.exists(args.json):
        print(f"❌ Calibration file not found: {args.json}")
        print("   Run calibrate_finger.py first to generate calibration data.")
        sys.exit(1)

    print("="*60)
    print("  CALIBRATION VERIFICATION")
    print("="*60)
    print(f"  Loading: {args.json}")
    
    # Load calibration
    try:
        data, t_left, t_right, t_center = _load_calibration(args.json)
    except Exception as e:
        print(f"❌ Failed to load calibration: {e}")
        sys.exit(1)
    
    # Print calibration info
    print(f"\n  Calibration metadata:")
    print(f"    tip_axis: {data.get('tip_axis', 'z')}")
    print(f"    symmetric: {data.get('symmetric', True)}")
    print(f"    samples: {data.get('samples', 'N/A')}")
    print(f"    residual_thresh: {data.get('residual_thresh', 'N/A')}")
    
    # Load meshes
    no_finger_path = data.get("no_finger_path")
    finger_path = data.get("finger_path")
    assem_path = data.get("assem_path")
    
    if not no_finger_path or not os.path.exists(no_finger_path):
        print(f"❌ no_finger mesh not found: {no_finger_path}")
        sys.exit(1)
    if not finger_path or not os.path.exists(finger_path):
        print(f"❌ finger mesh not found: {finger_path}")
        sys.exit(1)
    
    print(f"\n  Loading meshes...")
    no_finger = trimesh.load(no_finger_path)
    finger = trimesh.load(finger_path)
    print(f"    ✓ no_finger: {len(no_finger.vertices)} vertices")
    print(f"    ✓ finger: {len(finger.vertices)} vertices")
    
    assem = None
    if args.show_assem and assem_path and os.path.exists(assem_path):
        assem = trimesh.load(assem_path)
        print(f"    ✓ assem: {len(assem.vertices)} vertices")
    
    # Apply scale if needed
    if args.scale != 1.0:
        print(f"\n  Applying scale factor: {args.scale}")
        no_finger.apply_scale(args.scale)
        finger.apply_scale(args.scale)
        if assem:
            assem.apply_scale(args.scale)
        t_left = _scale_transform(t_left, args.scale)
        t_right = _scale_transform(t_right, args.scale)
        if t_center is not None:
            t_center = _scale_transform(t_center, args.scale)
    
    # Create transformed finger meshes
    left_mesh = finger.copy()
    left_mesh.apply_transform(t_left)
    right_mesh = finger.copy()
    right_mesh.apply_transform(t_right)
    
    # Build scene
    print(f"\n  Building visualization...")
    scene = trimesh.Scene()
    
    # Add base mesh
    if not args.no_base:
        scene.add_geometry(
            _colorize(no_finger, [0.7, 0.7, 0.7, 0.9]), 
            node_name="no_finger"
        )
    
    # Add original assembled mesh for comparison (transparent)
    if assem is not None:
        scene.add_geometry(
            _colorize(assem, [0.2, 0.6, 0.9, 0.25]), 
            node_name="assem_reference"
        )
    
    # Add calibrated fingers
    scene.add_geometry(
        _colorize(left_mesh, [0.9, 0.3, 0.3, 0.95]), 
        node_name="finger_left"
    )
    scene.add_geometry(
        _colorize(right_mesh, [0.3, 0.9, 0.3, 0.95]), 
        node_name="finger_right"
    )
    
    # Add coordinate axes if requested
    if args.show_axes:
        axis_scale = float(max(no_finger.extents)) * 0.15 if hasattr(no_finger, "extents") else 0.1
        axis_scale = max(axis_scale, 0.02)
        axis_mesh = _thick_axis_mesh(scale=axis_scale)
        
        # Origin axis
        scene.add_geometry(axis_mesh.copy(), node_name="axis_origin")
        
        # Left finger axis
        left_axis_pos = np.eye(4)
        left_axis_pos[:3, 3] = t_left[:3, 3]
        left_axis_pos[:3, :3] = t_left[:3, :3]
        scene.add_geometry(axis_mesh.copy(), transform=left_axis_pos, node_name="axis_left")
        
        # Right finger axis
        right_axis_pos = np.eye(4)
        right_axis_pos[:3, 3] = t_right[:3, 3]
        right_axis_pos[:3, :3] = t_right[:3, :3]
        scene.add_geometry(axis_mesh.copy(), transform=right_axis_pos, node_name="axis_right")
        
        # Center axis
        if t_center is not None:
            scene.add_geometry(axis_mesh.copy(), transform=t_center, node_name="axis_center")
    
    print("="*60)
    print("  Showing calibration result...")
    print("  (Red=Left finger, Green=Right finger)")
    if assem is not None:
        print("  (Blue transparent=Original assembled reference)")
    print("="*60)
    
    scene.show()
    
    print("\n  ✓ Verification complete")


if __name__ == "__main__":
    main()
