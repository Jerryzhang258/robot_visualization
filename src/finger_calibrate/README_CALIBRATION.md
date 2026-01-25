# Separate Finger Calibration

This directory contains scripts for calibrating gripper fingers with two approaches:

## 1. Symmetric Calibration (Original)
**Script**: `calibrate_finger.py`

Assumes left and right fingers are mirror images. Calibrates a single finger mesh and mirrors it for the opposite side.

**Usage**:
```bash
python calibrate_finger.py \
    --no-finger src/meshes/left_no_finger.STL \
    --finger src/meshes/finger.STL \
    --assem src/meshes/left_assem.STL \
    --out src/finger_calibrate/output
```

**Output**: `calibration_result.json` with symmetric transforms

**Pros**:
- Faster (only calibrates one finger)
- Ensures perfect symmetry
- Works well for symmetric grippers

**Cons**:
- May introduce orientation errors due to mirroring
- Cannot handle asymmetric finger designs

---

## 2. Separate Finger Calibration (New)
**Script**: `calibrate_finger_separate.py`

Calibrates left and right fingers independently using separate mesh files. No symmetry assumptions or mirroring.

**Usage**:
```bash
python calibrate_finger_separate.py \
    --no-finger src/meshes/left_no_finger.STL \
    --left-finger src/meshes/left_finger.STL \
    --right-finger src/meshes/right_finger.STL \
    --assem src/meshes/left_assem.STL \
    --out src/finger_calibrate/output
```

**Output**: `calibration_result.json` with separate left/right transforms

**JSON Structure**:
```json
{
  "symmetric": false,
  "no_finger_path": "src/meshes/left_no_finger.STL",
  "left_finger_path": "src/meshes/left_finger.STL",
  "right_finger_path": "src/meshes/right_finger.STL",
  "transform_left_finger_to_no_finger": [[...], [...], [...], [...]],
  "transform_right_finger_to_no_finger": [[...], [...], [...], [...]],
  "alignment_scores": {
    "left_finger": 0.123,
    "right_finger": 0.145,
    "average": 0.134
  }
}
```

**Pros**:
- More accurate - no mirroring artifacts
- Handles asymmetric finger geometries
- True independent optimization for each finger
- All transforms are right-handed (determinant ≈ 1.0)

**Cons**:
- Requires separate STL files for each finger
- Takes ~2x longer (calibrates both fingers)

---

## Verification

**Script**: `verify_calibration.py`

Visualizes calibration results by loading the JSON and reconstructing the gripper.

**Usage**:
```bash
# Basic verification
python verify_calibration.py

# With original assembled mesh overlay
python verify_calibration.py --show-assem

# With coordinate axes
python verify_calibration.py --show-axes

# Custom calibration file
python verify_calibration.py --json path/to/calibration_result.json

# Apply scale (e.g., mm to meters)
python verify_calibration.py --scale 0.001
```

**Features**:
- ✅ Supports both symmetric and separate finger calibrations
- ✅ Automatically detects format from JSON
- ✅ Shows alignment scores if available
- ✅ Color-coded: Red=Left finger, Green=Right finger, Gray=Base
- ✅ Optional reference overlay (blue transparent)

---

## Calibration Workflow

### For Separate Finger Calibration:

1. **Prepare STL files**:
   - `left_no_finger.STL` or `right_no_finger.STL` - Base gripper without fingers
   - `left_finger.STL` - Left finger geometry
   - `right_finger.STL` - Right finger geometry  
   - `left_assem.STL` or `right_assem.STL` - Fully assembled reference

2. **Run calibration**:
   ```bash
   cd src/finger_calibrate
   python calibrate_finger_separate.py --samples 8000 --icp-iters 150
   ```

3. **Interactive steps**:
   - View two detected finger clusters (red and blue points)
   - Select which cluster is the LEFT finger (1 or 2)
   - Script calibrates each finger independently

4. **Output**:
   - `output/calibration_result.json` with separate transforms
   - Alignment scores for quality assessment

5. **Verify**:
   ```bash
   python verify_calibration.py --show-assem --show-axes
   ```
   - Check that fingers align with reference mesh
   - Red (left) and green (right) should match blue overlay

---

## Common Parameters

### Calibration Scripts

- `--samples N`: Points to sample from each mesh (default: 6000)
  - Higher = more accurate but slower
  - Recommended: 6000-10000

- `--icp-iters N`: ICP iterations (default: 100)
  - Higher = more thorough alignment
  - Recommended: 100-200

- `--residual-thresh X`: Distance threshold for finger detection (mm) (default: 0.5)
  - Lower = stricter finger separation from base
  - Increase if "No finger-only points found" error
  - Decrease if base geometry included in fingers

- `--split-axis {x,y,z}`: Fallback axis for left/right split (default: y)
  - Used if automatic clustering fails

- `--tip-axis {x,y,z,-x,-y,-z}`: Direction fingertips point (default: z)

### Troubleshooting

**"No finger-only points found"**:
- Increase `--residual-thresh` (try 0.7 or 1.0)
- Fingers may be very close to base

**"Could not find 2 finger clusters"**:
- Adjust `--residual-thresh`
- Use `--split-axis` and `--split-value` for manual split

**Poor alignment scores** (>0.5 mm):
- Increase `--samples` and `--icp-iters`
- Check that meshes are properly aligned in origin frame
- Verify STL files are clean (no disconnected geometry)

**Left-handed coordinate frames** (det < 0):
- Should not happen with separate calibration
- If it does, the visualization pipeline's `_fix_rotation()` will correct it

---

## Technical Details

### Coordinate Frame Convention

Both scripts produce transforms in the **no_finger frame**:
- **Origin**: Base gripper coordinate system
- **Orientation**: Right-handed (determinant ≈ 1.0)
- **Z-axis**: Points toward fingertip (default)
- **X-axis**: Opening direction (finger spread)
- **Y-axis**: Perpendicular (completes right-handed system)

### Transform Hierarchy

```
world → no_finger → finger (left/right)
```

The calibration outputs:
- `transform_left_finger_to_no_finger`: Left finger in base frame
- `transform_right_finger_to_no_finger`: Right finger in base frame

### Multi-Start ICP

Both scripts use multi-start ICP to avoid local minima:
- Tests 24 different rotation candidates
- Covers all major orientations (90°, 180°, 270° around each axis)
- Selects best result based on alignment score

### Clustering Algorithm

Separate calibration uses spatial connectivity clustering:
1. Build KD-tree of finger-only points
2. Estimate connectivity threshold from local point density
3. Flood-fill to find connected components
4. Select two largest clusters as left/right fingers
5. User confirms which cluster is left via interactive viewer

---

## Files Generated

### Calibration Output
- `calibration_result.json` - Transform matrices and metadata
- Console output with alignment scores

### Verification
- No files generated (visualization only)
- Interactive 3D viewer with colored meshes

---

## Integration with Visualization

The visualization pipeline (`viz_vb_data.py`, `viz_3d_enhanced.py`) automatically:
- Detects separate vs symmetric calibration
- Loads appropriate finger meshes
- Applies transforms without additional processing
- Handles gripper opening animations

See `DUAL_MESH_IMPLEMENTATION.md` for visualization system details.

---

## Quick Start

For left gripper with separate fingers:
```bash
# Calibrate
python src/finger_calibrate/calibrate_finger_separate.py \
    --no-finger src/meshes/left_no_finger.STL \
    --left-finger src/meshes/left_finger.STL \
    --right-finger src/meshes/right_finger.STL \
    --assem src/meshes/left_assem.STL

# Verify
python src/finger_calibrate/verify_calibration.py \
    --show-assem --show-axes

# Use in visualization (automatic)
python src/viz_vb_data.py <episode_path>
```

For right gripper, use corresponding `right_*` mesh files.
