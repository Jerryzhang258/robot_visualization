# Finger Calibration Tool

This tool calibrates finger positions for robotic gripper visualization by aligning individual finger meshes with an assembled gripper reference.

## Overview

The calibration process:
1. Aligns the gripper base (without fingers) to the assembled reference
2. **Extracts finger-only regions** by subtracting the aligned base from the assembly
3. **Interactive finger selection** - user confirms which cluster is the left finger
4. Uses multi-stage ICP optimization to find optimal finger positions
5. Supports interactive adjustment and quality verification

## Quick Start

```bash
python src/finger_calibrate/calibrate_finger.py \
  --no-finger src/meshes/left_no_finger.STL \
  --finger src/meshes/finger.STL \
  --assem src/meshes/left_assem.STL \
  --out src/finger_calibrate/output \
```

```bash
# Basic verification
python src/finger_calibrate/verify_calibration.py

# Compare with original assembled mesh
python src/finger_calibrate/verify_calibration.py --show-assem

# Show coordinate axes
python src/finger_calibrate/verify_calibration.py --show-axes

# Full comparison
python src/finger_calibrate/verify_calibration.py --show-assem --
```

## Input Files

You need three STL mesh files:
- **no_finger**: Gripper base without fingers (default: `src/meshes/left_no_finger.STL`)
- **finger**: Single finger mesh (default: `src/meshes/finger.STL`)
- **assem**: Complete assembled gripper (default: `src/meshes/left_assem.STL`)

## Key Parameters

### Essential Parameters

- `--residual-thresh` (default: 0.5): Distance threshold to identify finger regions
  - Lower values (0.2-0.4): Use when fingers are very close to the base
  - Higher values (0.5-1.0): Use when there's clear separation

- `--split-axis` (default: y): Axis used to separate left/right fingers
  - Common choices: `x`, `y`, or `z` depending on gripper orientation

- `--tip-axis` (default: z): Local axis pointing toward fingertip
  - Use `-z` if fingers point in negative Z direction

### Optimization Parameters

- `--samples` (default: 6000): Points sampled from each mesh
  - Higher = more accurate but slower (try 10000-15000 for final calibration)
  
- `--icp-iters` (default: 100): ICP iterations per optimization round
  - Increase to 150-200 if optimization gets stuck
  
- `--continue-rounds` (default: 3): Number of optimization rounds
  - More rounds = better convergence but slower

### Modes

- `--no-nudge`: Skip interactive positioning (for batch processing)
- `--no-interactive-rounds`: Skip review prompts between rounds
- `--no-symmetric`: Optimize left/right fingers independently

## Interactive Controls

### Finger Selection (After Point Extraction)

After extracting finger-only points, a viewer opens showing two point clusters:
- **Red** = Assumed left finger
- **Green** = Assumed right finger

**Controls:**
- `Space` - Swap left/right assignment
- `Enter` - Confirm selection and continue
- `q`/`Esc` - Cancel calibration

This step ensures the calibration matches your gripper's actual left/right configuration.

### Nudge Mode (Initial Positioning)

When the 3D viewer opens, use these controls:

**Target Selection:**
- `m` - Toggle between adjusting center frame or finger position

**Translation:**
- `w`/`s` - Move along +Y/-Y
- `a`/`d` - Move along -X/+X
- `r`/`f` - Move along +Z/-Z

**Rotation:**
- `i`/`k` - Rotate around +Rx/-Rx
- `j`/`l` - Rotate around +Ry/-Ry
- `u`/`o` - Rotate around +Rz/-Rz

**Step Size:**
- `[`/`]` - Decrease/increase translation step
- `;`/`'` - Decrease/increase rotation step
- `v` - View current step sizes

**Alignment Helpers:**
- `x`/`y`/`z` - Align finger tip axis to center X/Y/Z (toggles +/-)

**Actions:**
- `p` - Save current pose as snapshot
- `Enter` - Accept and start optimization
- `q`/`Esc` - Cancel calibration

### Round Review

After each optimization round:
- View the result in 3D viewer (close when done)
- Choose action:
  - `c` - Continue to next round
  - `s` - Save and finish
  - `d` - Discard and quit

## Troubleshooting

### Problem: Optimization gets stuck in local minimum

**Solutions:**
1. Use interactive nudge mode to get closer initial position
2. Increase `--icp-iters` to 150-200
3. Increase `--continue-rounds` to 5-7
4. Try different `--residual-thresh` values

### Problem: Wrong finger regions detected

**Solutions:**
1. Adjust `--residual-thresh` (try 0.3 or 0.7)
2. Check `--split-axis` and `--split-value` parameters
3. Verify mesh files are correct and properly oriented

### Problem: Fingers aligned incorrectly

**Solutions:**
1. Check `--tip-axis` parameter (may need `-z` instead of `z`)
2. Verify `--inward-axis` parameter
3. Use nudge mode to manually correct orientation

### Problem: Calibration is too slow

**Solutions:**
1. Reduce `--samples` to 4000-5000
2. Reduce `--icp-iters` to 80
3. Use `--no-interactive-rounds` for batch mode
4. Reduce `--continue-rounds` to 1-2

## Output

The calibration produces:

1. **calibration_result.json**: Transform matrices and metadata
   - `transform_finger_left_to_no_finger`: Left finger relative to base
   - `transform_finger_right_to_no_finger`: Right finger relative to base
   - `transform_finger_base_to_no_finger`: Center frame relative to base

2. **Aligned STL files**:
   - `left_no_finger_aligned_final.stl`: Aligned base mesh
   - `finger_left_aligned_final.stl`: Aligned left finger
   - `finger_right_aligned_final.stl`: Aligned right finger

3. **Snapshots** (if saved during nudge):
   - `nudge_snapshot_YYYYMMDD_HHMMSS.json`: Saved poses for recovery

## Advanced Usage

### Manual Initial Transform (for Large Mesh Offsets)

When meshes have large centroid offsets (>100mm), automatic alignment may fail. You can provide an initial transformation guess:

```bash
# Using calibrate_finger_separate.py with initial transform
python src/finger_calibrate/calibrate_finger_separate.py \
    --no-finger src/meshes/left_no_finger.STL \
    --left-finger src/meshes/left_finger.STL \
    --right-finger src/meshes/right_finger.STL \
    --assem src/meshes/left_assem.STL \
    --out src/finger_calibrate/output \
    --initial-transform src/finger_calibrate/output/initial_transform.json

```

**Transform Format** (see `example_initial_transform.json`):
```json
{
  "transform": [
    [1.0, 0.0, 0.0, 100.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

This is a 4x4 homogeneous transformation matrix:
- Top-left 3x3: Rotation matrix (must be orthonormal)
- Right column (tx, ty, tz): Translation in millimeters
- Bottom row: Always [0, 0, 0, 1]

**How to find the initial transform:**
1. Open both meshes in a 3D viewer (e.g., MeshLab, Blender)
2. Manually align the quest mesh to the assembled mesh
3. Record the rotation and translation applied
4. Format as a JSON file with the 4x4 matrix

**Note:** The initial transform is used as a starting point for ICP refinement. It doesn't need to be perfect, but should get the meshes roughly aligned (within ~20mm).

### Batch Processing

```bash
# Non-interactive mode for automated calibration
python src/finger_calibrate/calibrate_finger.py \
  --no-nudge \
  --no-interactive-rounds \
  --continue-rounds 5 \
  --icp-iters 150
```

### High-Accuracy Calibration

```bash
# Maximum quality settings (slow)
python src/finger_calibrate/calibrate_finger.py \
  --samples 15000 \
  --icp-iters 200 \
  --continue-rounds 7 \
  --max-samples 15000
```

### Custom Mesh Paths

```bash
python src/finger_calibrate/calibrate_finger.py \
  --no-finger path/to/base.stl \
  --finger path/to/finger.stl \
  --assem path/to/assembled.stl \
  --out path/to/output
```

## Algorithm Details

The calibration uses a multi-stage approach to avoid local minima:

1. **Base Alignment**: ICP to align no_finger mesh to assembled reference
2. **Finger Region Extraction**: Subtracts aligned base from assembly to isolate only finger geometry (prevents fingers from being pushed into base during optimization)
3. **Finger Selection**: Interactive viewer lets user confirm which point cluster is the left finger (swap with Space if needed)
4. **Initial Center Frame**: Set at midpoint between left/right finger-only point centroids
5. **Left Finger Alignment**: 
   - Principal component analysis with 8 sign combinations for coarse alignment
   - Multi-start ICP (24+ rotation candidates) to fully merge with left target points
6. **Center Pose Optimization**: Search along the left-right axis to find optimal center position so mirrored right finger aligns with right target points
7. **Iterative Optimization**: Each round:
   - Step 1: Optimize left finger against left finger-only points (full 6-DOF)
   - Step 2: Optimize center pose so mirrored right finger matches right points

This two-step approach ensures:
- Left finger fully merges with its target geometry
- Right finger position is derived by symmetric mirroring
- Finger spacing is automatically optimized for best fit

## See Also

- `verify_calibration.py`: Visualize calibration results
- `../viz_3d_enhanced.py`: Uses calibration for robot visualization