#!/usr/bin/env python3
"""Change mesh origin by moving it to a new position.

This script loads a mesh and a JSON file containing a 4x4 homogeneous transformation
matrix. It then moves the mesh's origin point to the position specified by the
transformation, effectively re-centering the mesh's coordinate system.

The transformation defines where the new origin should be. The mesh vertices are
adjusted so that this point becomes the new (0,0,0) origin of the saved mesh.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh


def load_transform_from_json(json_path, key=None):
    """Load a 4x4 transformation matrix from JSON file.
    
    Args:
        json_path: Path to JSON file
        key: Optional key to extract from JSON. If None, assumes JSON is just the matrix.
    
    Returns:
        4x4 numpy array
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if key:
        if key not in data:
            raise ValueError(f"Key '{key}' not found in JSON file. Available keys: {list(data.keys())}")
        matrix = data[key]
    else:
        # If no key specified, try to use the data directly or find a matrix-like structure
        if isinstance(data, list):
            matrix = data
        elif isinstance(data, dict):
            # Try common keys
            for common_key in ["transform", "transformation", "matrix", "T"]:
                if common_key in data:
                    matrix = data[common_key]
                    print(f"  Using key '{common_key}' from JSON")
                    break
            else:
                # If only one key, use it
                if len(data) == 1:
                    matrix = list(data.values())[0]
                else:
                    raise ValueError(f"JSON contains multiple keys. Please specify which one to use: {list(data.keys())}")
        else:
            matrix = data
    
    # Convert to numpy array and validate shape
    transform = np.array(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got shape {transform.shape}")
    
    return transform


def create_axis_mesh(scale=0.1):
    """Create colored axis mesh for visualization.
    
    Args:
        scale: Length of each axis
    
    Returns:
        Combined mesh with X (red), Y (green), Z (blue) axes
    """
    radius = max(scale * 0.02, 0.001)
    length = scale

    def _create_axis(color, axis_name):
        """Create a single axis cylinder."""
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        rgba = (np.array(color) * 255).astype(np.uint8)
        cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
        
        # Rotate and position based on axis
        if axis_name == "x":
            rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
            rot[:3, 3] = [length / 2.0, 0, 0]
        elif axis_name == "y":
            rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
            rot[:3, 3] = [0, length / 2.0, 0]
        else:  # z
            rot = np.eye(4)
            rot[:3, 3] = [0, 0, length / 2.0]
        
        cyl.apply_transform(rot)
        return cyl

    x_axis = _create_axis([1.0, 0.0, 0.0, 1.0], "x")
    y_axis = _create_axis([0.0, 1.0, 0.0, 1.0], "y")
    z_axis = _create_axis([0.0, 0.0, 1.0, 1.0], "z")
    
    return trimesh.util.concatenate([x_axis, y_axis, z_axis])


def colorize_mesh(mesh, rgba):
    """Apply color to mesh vertices.
    
    Args:
        mesh: Trimesh object
        rgba: RGBA color as [r, g, b, a] in range [0, 1]
    
    Returns:
        Colored copy of the mesh
    """
    colored = mesh.copy()
    rgba_arr = (np.array(rgba) * 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba_arr, (len(colored.vertices), 1))
    return colored


def main():
    parser = argparse.ArgumentParser(
        description="Change mesh origin to a new position specified by a transformation matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - move mesh origin to transformation point
  python change_mesh_origin.py mesh.stl transform.json
  
  # Specify output file
  python change_mesh_origin.py mesh.stl transform.json --output mesh_new_origin.stl
  
  # Use specific key from JSON
  python change_mesh_origin.py mesh.stl calib.json --key transform_quest_to_gripper_base
  
  # Show coordinate axes to visualize origin change
  python change_mesh_origin.py mesh.stl transform.json --show-axes
  
  # Don't save, just visualize
  python change_mesh_origin.py mesh.stl transform.json --no-save

Note: The transformation matrix defines where the new origin should be.
The mesh vertices are adjusted so this point becomes the new (0,0,0).
"""
    )
    parser.add_argument(
        "mesh",
        help="Path to input mesh file (STL, OBJ, PLY, etc.)"
    )
    parser.add_argument(
        "json",
        help="Path to JSON file containing 4x4 transformation matrix"
    )
    parser.add_argument(
        "--key",
        help="Key to extract transformation from JSON (if JSON contains multiple entries)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to output mesh file (default: <input>_transformed.<ext>)"
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show coordinate axes in visualization"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the transformed mesh, just visualize"
    )
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.mesh):
        print(f"❌ Mesh file not found: {args.mesh}")
        sys.exit(1)
    
    if not os.path.exists(args.json):
        print(f"❌ JSON file not found: {args.json}")
        sys.exit(1)

    print("="*70)
    print("  MESH ORIGIN TRANSFORMATION")
    print("="*70)
    
    # Load mesh
    print(f"\n📂 Loading mesh: {args.mesh}")
    try:
        mesh = trimesh.load(args.mesh)
        print(f"   ✓ Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"   Original bounds: {mesh.bounds}")
    except Exception as e:
        print(f"❌ Failed to load mesh: {e}")
        sys.exit(1)
    
    # Load transformation
    print(f"\n📂 Loading transformation: {args.json}")
    try:
        transform = load_transform_from_json(args.json, args.key)
        print(f"   ✓ Loaded 4x4 transformation matrix")
        print("   Transform matrix:")
        for row in transform:
            print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")
    except Exception as e:
        print(f"❌ Failed to load transformation: {e}")
        sys.exit(1)
    
    # Change the mesh origin
    # This moves the mesh so that the transformation's translation becomes the new origin
    # We apply the INVERSE of the transformation to move vertices
    print(f"\n🔧 Changing mesh origin...")
    print(f"   Original origin will be moved to: {transform[:3, 3]}")
    
    # Create the new mesh with changed origin
    transformed_mesh = mesh.copy()
    
    # Apply the inverse transformation to move the origin
    # This makes the point at transform[:3, 3] become the new (0,0,0)
    inverse_transform = np.linalg.inv(transform)
    transformed_mesh.apply_transform(inverse_transform)
    
    print(f"   ✓ Origin changed")
    print(f"   Old bounds: {mesh.bounds}")
    print(f"   New bounds: {transformed_mesh.bounds}")
    
    # Create visualization scene
    print(f"\n👁️  Creating visualization...")
    scene = trimesh.Scene()
    
    # Add original mesh (semi-transparent blue)
    scene.add_geometry(
        colorize_mesh(mesh, [0.3, 0.5, 0.9, 0.4]),
        node_name="original_mesh"
    )
    
    # Add transformed mesh (opaque green)
    scene.add_geometry(
        colorize_mesh(transformed_mesh, [0.3, 0.9, 0.3, 0.95]),
        node_name="transformed_mesh"
    )
    
    # Add coordinate axes if requested
    if args.show_axes:
        axis_scale = float(max(mesh.extents)) * 0.2 if hasattr(mesh, "extents") else 0.1
        axis_scale = max(axis_scale, 0.01)
        axis_mesh = create_axis_mesh(scale=axis_scale)
        
        # Original origin (identity)
        scene.add_geometry(
            axis_mesh.copy(),
            node_name="origin_original"
        )
        
        # Transformed origin
        scene.add_geometry(
            axis_mesh.copy(),
            transform=transform,
            node_name="origin_transformed"
        )
    
    print("="*70)
    print("  Visualization:")
    print("    🔵 Blue (transparent): Original mesh at original origin")
    print("    🟢 Green (solid): Mesh with new origin (shifted)")
    if args.show_axes:
        print("    🔴🟢🔵 RGB axes at (0,0,0): Original coordinate system")
        print("    🔴🟢🔵 RGB axes transformed: Where the original origin moved to")
    print("\n  Note: The green mesh's origin (0,0,0) is now at the transformation point")
    print("="*70)
    print("\n  Close the visualization window to continue...")
    
    # Show visualization
    scene.show()
    
    # Save transformed mesh
    if not args.no_save:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Generate default output name
            base, ext = os.path.splitext(args.mesh)
            output_path = f"{base}_transformed{ext}"
        
        print(f"\n💾 Saving transformed mesh: {output_path}")
        try:
            transformed_mesh.export(output_path)
            print(f"   ✓ Saved successfully")
        except Exception as e:
            print(f"❌ Failed to save mesh: {e}")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping save (--no-save flag set)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
