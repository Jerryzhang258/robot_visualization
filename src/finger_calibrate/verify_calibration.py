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
    """Load calibration data from JSON file.
    
    Uses the same transform hierarchy as visualization:
    quest -> gripper_base -> fingers
    
    Also loads transform_quest_to_assem for proper alignment with assembled reference.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load the hierarchical transforms (required format)
    if "transform_quest_to_gripper_base" not in data:
        raise ValueError(
            "Calibration file missing required key 'transform_quest_to_gripper_base'. "
            "Please re-run calibration to generate the correct format."
        )
    if "transform_gripper_base_to_left_finger" not in data:
        raise ValueError("Calibration file missing 'transform_gripper_base_to_left_finger'")
    if "transform_gripper_base_to_right_finger" not in data:
        raise ValueError("Calibration file missing 'transform_gripper_base_to_right_finger'")
    
    t_quest_to_gb = np.array(data["transform_quest_to_gripper_base"], dtype=np.float64)
    t_gb_to_left = np.array(data["transform_gripper_base_to_left_finger"], dtype=np.float64)
    t_gb_to_right = np.array(data["transform_gripper_base_to_right_finger"], dtype=np.float64)
    
    # Load transform from quest to assembled reference frame (if available)
    t_quest_to_assem = None
    if "transform_quest_to_assem" in data:
        t_quest_to_assem = np.array(data["transform_quest_to_assem"], dtype=np.float64)
    
    # Compute full transforms: quest -> gripper_base -> finger
    t_quest_to_left = t_quest_to_gb @ t_gb_to_left
    t_quest_to_right = t_quest_to_gb @ t_gb_to_right
    
    # Check if this is separate finger calibration
    has_separate_meshes = "left_finger_path" in data and "right_finger_path" in data
    
    return data, t_quest_to_gb, t_quest_to_left, t_quest_to_right, t_quest_to_assem, has_separate_meshes


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
        data, t_quest_to_gripper_base, t_quest_to_left, t_quest_to_right, t_quest_to_assem, has_separate_meshes = _load_calibration(args.json)
    except Exception as e:
        print(f"❌ Failed to load calibration: {e}")
        sys.exit(1)
    
    # Print calibration info
    print(f"\n  Calibration metadata:")
    print(f"    tip_axis: {data.get('tip_axis', 'z')}")
    print(f"    symmetric: {data.get('symmetric', True)}")
    print(f"    separate_meshes: {has_separate_meshes}")
    print(f"    samples: {data.get('samples', 'N/A')}")
    print(f"    residual_thresh: {data.get('residual_thresh', 'N/A')}")
    if "alignment_scores" in data:
        scores = data["alignment_scores"]
        print(f"    alignment_scores:")
        print(f"      left:  {scores.get('left_finger', 'N/A')} mm")
        print(f"      right: {scores.get('right_finger', 'N/A')} mm")
        print(f"      avg:   {scores.get('average', 'N/A')} mm")
    
    # Load meshes - handle both separate and single finger formats
    # Support both old (no_finger_path) and new (quest_path) naming
    quest_path = data.get("quest_path") or data.get("no_finger_path")
    assem_path = data.get("assem_path")
    
    if not quest_path or not os.path.exists(quest_path):
        print(f"❌ quest/gripper base mesh not found: {quest_path}")
        sys.exit(1)
    
    print(f"\n  Loading meshes...")
    quest = trimesh.load(quest_path)
    print(f"    ✓ quest (gripper base): {len(quest.vertices)} vertices")
    
    # Load finger meshes (separate or single)
    left_finger = None
    right_finger = None
    
    if has_separate_meshes:
        # Load separate left and right finger meshes
        left_finger_path = data.get("left_finger_path")
        right_finger_path = data.get("right_finger_path")
        
        if not left_finger_path or not os.path.exists(left_finger_path):
            print(f"❌ left_finger mesh not found: {left_finger_path}")
            sys.exit(1)
        if not right_finger_path or not os.path.exists(right_finger_path):
            print(f"❌ right_finger mesh not found: {right_finger_path}")
            sys.exit(1)
        
        left_finger = trimesh.load(left_finger_path)
        right_finger = trimesh.load(right_finger_path)
        print(f"    ✓ left_finger: {len(left_finger.vertices)} vertices")
        print(f"    ✓ right_finger: {len(right_finger.vertices)} vertices")
    else:
        # Load single symmetric finger mesh
        finger_path = data.get("finger_path")
        if not finger_path or not os.path.exists(finger_path):
            print(f"❌ finger mesh not found: {finger_path}")
            sys.exit(1)
        
        finger = trimesh.load(finger_path)
        left_finger = finger.copy()
        right_finger = finger.copy()
        print(f"    ✓ finger: {len(finger.vertices)} vertices (used for both sides)")
    
    assem = None
    if args.show_assem and assem_path and os.path.exists(assem_path):
        assem = trimesh.load(assem_path)
        print(f"    ✓ assem: {len(assem.vertices)} vertices")
    
    # Apply scale if needed
    if args.scale != 1.0:
        print(f"\n  Applying scale factor: {args.scale}")
        quest.apply_scale(args.scale)
        left_finger.apply_scale(args.scale)
        right_finger.apply_scale(args.scale)
        if assem:
            assem.apply_scale(args.scale)
        t_quest_to_gripper_base = _scale_transform(t_quest_to_gripper_base, args.scale)
        t_quest_to_left = _scale_transform(t_quest_to_left, args.scale)
        t_quest_to_right = _scale_transform(t_quest_to_right, args.scale)
        if t_quest_to_assem is not None:
            t_quest_to_assem = _scale_transform(t_quest_to_assem, args.scale)
    
    # Create transformed meshes
    # If we have the assem reference, use it as the reference frame
    # and transform everything else relative to it
    if t_quest_to_assem is not None and assem is not None:
        # Assem is the reference frame (stays at origin)
        # Transform quest base to assem frame
        quest_in_assem = quest.copy()
        quest_in_assem.apply_transform(t_quest_to_assem)
        
        # Transform fingers: assem -> quest -> finger
        left_mesh = left_finger.copy()
        left_mesh.apply_transform(t_quest_to_assem @ t_quest_to_left)
        right_mesh = right_finger.copy()
        right_mesh.apply_transform(t_quest_to_assem @ t_quest_to_right)
        
        # Transforms for axes (if shown)
        axis_quest = t_quest_to_assem
        axis_gripper_base = t_quest_to_assem @ t_quest_to_gripper_base
        axis_left = t_quest_to_assem @ t_quest_to_left
        axis_right = t_quest_to_assem @ t_quest_to_right
    else:
        # No assem reference, use quest as reference frame (stays at origin)
        quest_in_assem = quest.copy()
        
        # Apply transforms in sequence: quest -> gripper_base -> finger
        left_mesh = left_finger.copy()
        left_mesh.apply_transform(t_quest_to_left)
        right_mesh = right_finger.copy()
        right_mesh.apply_transform(t_quest_to_right)
        
        # Transforms for axes (if shown)
        axis_quest = np.eye(4)
        axis_gripper_base = t_quest_to_gripper_base
        axis_left = t_quest_to_left
        axis_right = t_quest_to_right
    
    # Build scene
    print(f"\n  Building visualization...")
    scene = trimesh.Scene()
    
    # Add base mesh (transformed to assem frame if available)
    if not args.no_base:
        scene.add_geometry(
            _colorize(quest_in_assem, [0.7, 0.7, 0.7, 0.9]), 
            node_name="quest_base"
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
        axis_scale = float(max(quest.extents)) * 0.15 if hasattr(quest, "extents") else 0.1
        axis_scale = max(axis_scale, 0.02)
        axis_mesh = _thick_axis_mesh(scale=axis_scale)
        
        # Quest origin axis (in assem frame if available, otherwise at origin)
        scene.add_geometry(
            axis_mesh.copy(), 
            transform=axis_quest,
            node_name="axis_quest_origin"
        )
        
        # Gripper_base axis
        scene.add_geometry(
            axis_mesh.copy(), 
            transform=axis_gripper_base, 
            node_name="axis_gripper_base"
        )
        
        # Left finger axis
        scene.add_geometry(
            axis_mesh.copy(), 
            transform=axis_left, 
            node_name="axis_left_finger"
        )
        
        # Right finger axis
        scene.add_geometry(
            axis_mesh.copy(), 
            transform=axis_right, 
            node_name="axis_right_finger"
        )
    
    print("="*60)
    print("  Showing calibration result...")
    print("  Color coding:")
    print("    Gray: Quest gripper base")
    print("    Red: Left finger")
    print("    Green: Right finger")
    if assem is not None:
        print("    Blue (transparent): Original assembled reference")
    if args.show_axes:
        print("  Axes:")
        print("    Quest origin (RGB)")
        print("    Gripper base (RGB) - should overlap with quest")
        print("    Left finger (RGB)")
        print("    Right finger (RGB)")
    print("="*60)
    
    scene.show()
    
    print("\n  ✓ Verification complete")


if __name__ == "__main__":
    main()
