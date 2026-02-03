# Quest Hand-Eye Calibration

This tool performs hand-eye calibration to find the constant transformation between the end-effector (EE) frame and the Quest headset tracking frame.

## Problem Statement

We want to find the constant transformation `X = ^EE T_Q` (Quest position in end-effector frame) that relates:
- **Quest tracking measurements**: `^W T_Q` (Quest pose in world frame, in meters)
- **End-effector motions**: `^EE_i T_EE_{i+1}` (relative motion between consecutive EE poses, in millimeters)

The hand-eye calibration equation is:
```
A^-1 @ X = X @ B
```

Where:
- `A = ^EE_i T_EE_{i+1}` (known EE motion from robot)
- `B = inv(^W T_Q,i) @ (^W T_Q,j)` (measured Quest motion)
- `X = ^EE T_Q` (unknown constant mount we want to find)

## Input Data Format

The input JSON file should contain one or more datasets:

```json
{
  "datasets": [
    {
      "W_T_Q": [
        [[R11, R12, R13, tx], [R21, R22, R23, ty], [R31, R32, R33, tz], [0, 0, 0, 1]],
        ...
      ],
      "EEi_T_EEi1": [
        [[R11, R12, R13, tx], [R21, R22, R23, ty], [R31, R32, R33, tz], [0, 0, 0, 1]],
        ...
      ]
    },
    {
      // Additional datasets...
    }
  ]
}
```

### Dataset Structure

Each dataset contains:

1. **`W_T_Q`**: Array of 4x4 transformation matrices
   - Quest pose at each time step in world frame
   - **Translation units**: METERS (will be converted to mm internally)
   - Length: `n` poses
   - Each dataset can have its own arbitrary world frame

2. **`EEi_T_EEi1`**: Array of 4x4 transformation matrices
   - Relative motion from EE pose `i` to pose `i+1`
   - **Translation units**: MILLIMETERS
   - Length: `n-1` transforms (one less than `W_T_Q`)
   - These are the **known, accurate** motions from robot kinematics

### Multiple Datasets

- You can provide multiple datasets (e.g., different recording sessions)
- Each dataset can have its own world frame (they don't need to be aligned)
- The algorithm uses **within-dataset** relative motions only
- All datasets constrain the same unknown `X` transformation
- More datasets with diverse motions improve calibration accuracy

## Usage

### Basic Usage (Recommended)

```bash
python calculate_quest_to_base.py data/your_data.json
```

This uses **consecutive-only mode** (default), which:
- Creates pairs only from consecutive poses: (i, i+1)
- Avoids composing multiple transforms
- Minimizes amplification of Quest tracking noise
- **Includes automatic sanity check** to detect Quest tracking errors
- **Recommended for best results**

The sanity check automatically removes pairs where Quest tracking distance differs from EE motion by >25%, helping filter out bad data.

### Advanced Usage

#### Analyzing Per-Dataset Residuals

```bash
python calculate_quest_to_base.py data/your_data.json --per-dataset-residual
```

This shows residuals for each dataset separately, helping you identify problematic data:

```
Dataset 1 (4 pairs):
  Physical Alignment Errors:
    Translation RMS: 4.17mm     ← Excellent!
    Rotation RMS: 3.06°

Dataset 2 (3 pairs):
  Physical Alignment Errors:
    Translation RMS: 206.93mm   ← Terrible! Consider removing this dataset
    Rotation RMS: 2.11°
```

In this example, Dataset 2 has Quest tracking issues and should be removed or re-recorded.

#### Using Non-Consecutive Pairs (Not Recommended)

```bash
python calculate_quest_to_base.py data/your_data.json --all-pairs --max-gap 3
```

**WARNING**: This mode composes multiple transforms and can significantly amplify rotation errors when Quest tracking has noise. Only use if you have very accurate Quest measurements.

Options:
- `--all-pairs`: Enable non-consecutive pair generation
- `--max-gap N`: Maximum gap between poses (default: 1)
  - `--max-gap 1`: Only consecutive pairs (i, i+1)
  - `--max-gap 2`: Pairs (i, i+1) and (i, i+2)
  - `--max-gap 3`: Pairs up to (i, i+3)

#### Sanity Check Options

The sanity check automatically filters out pairs with large Quest tracking errors:

```bash
# Adjust threshold (default is 25%)
python calculate_quest_to_base.py data/your_data.json --sanity-threshold 0.20

# Disable sanity check (not recommended)
python calculate_quest_to_base.py data/your_data.json --no-sanity-check
```

**How it works**:
- For pairs where **EE motion has low rotation** (<5°), the sanity check compares translation distances
- If Quest translation distance differs from EE translation by more than the threshold, the pair is removed
- Only checks EE rotation (not Quest rotation), since Quest tracking may add spurious rotations
- This catches major Quest tracking errors (e.g., 100mm EE motion but Quest reports 500mm)

**Why only low-rotation EE motions?**
- When EE performs combined rotation+translation, comparing translation distances is geometrically meaningless
- Pure translations allow direct distance comparison between Quest and EE measurements
- This ensures the sanity check only validates motion where the comparison is valid

Options:
- `--sanity-threshold 0.25`: Relative error threshold (default: 25%)
  - Pairs with `|quest_dist - ee_dist| / ee_dist > 0.25` are removed
  - Lower values (e.g., 0.15) are more strict, higher values (e.g., 0.35) are more permissive
- `--no-sanity-check`: Disable automatic filtering (use if you're confident in data quality)

Example output with sanity check:
```
⚠️  Dataset 4: Sanity check failed for 1 pair(s)
   Removed pairs where Quest/EE translation distance differs by >25%:
   - Pair (2, 3): EE=100.00mm, Quest=513.06mm, Error=413.1%
                     (EE_rot=0.00°, Quest_rot=6.00°)
```



### 1. Pair Generation

For each dataset with `n` poses:

**Consecutive-only mode** (default):
- Generates `n-1` pairs: (0,1), (1,2), ..., (n-2, n-1)
- Each pair uses direct measurements without composition

**All-pairs mode** (with `max_gap`):
- Generates multiple pairs by composing transforms
- Example with `n=5` and `max_gap=3`:
  - Gap 1: (0,1), (1,2), (2,3), (3,4)
  - Gap 2: (0,2), (1,3), (2,4)
  - Gap 3: (0,3), (1,4)
  - Total: 9 pairs

**Sanity Check** (applied to all modes):
- For each pair where EE rotation < 5°, compares translation distances
- Removes pairs where Quest distance error > threshold (default 25%)
- Only checks EE rotation magnitude, ignoring Quest rotation (which may be noisy)
- Helps automatically filter out Quest tracking glitches or drift

For each pair `(i, j)`:
```python
A_ij = EEi_T_EEi1[i] @ EEi_T_EEi1[i+1] @ ... @ EEi_T_EEi1[j-1]
B_ij = inv(W_T_Q[i]) @ W_T_Q[j]
```

### 2. Initialization

**Rotation Initialization**:
- Use dataset with highest rotational variance
- Extract rotation vectors from all pairs: `a = log(A.R)`, `b = log(B.R)`
- Use `scipy.spatial.transform.Rotation.align_vectors()` to find initial rotation

**Translation Initialization**:
- Solve least-squares problem: `(A.R - I) @ t_X = R_X @ B.t - A.t`
- Combines all pairs to get initial translation estimate

### 3. SE(3) Refinement

Uses non-linear least-squares optimization on the manifold:
- Parameterize `X` update as: `X_new = X_old @ Exp(delta)`
- `delta` is a 6-DOF vector in se(3) tangent space: `[v_x, v_y, v_z, w_x, w_y, w_z]`
- Minimize residual: `sum_i || log(A_i @ X @ inv(X @ B_i)) ||^2`
- Uses Huber loss for robustness to outliers
- Implemented with `scipy.optimize.least_squares`

## Output

### Calibration Result

The script outputs:

1. **Initial Estimate**: Pre-refinement transformation from rotation/translation initialization
2. **Refined Transformation**: Final calibrated `^EE T_Q` (translation in mm)
   - 4x4 matrix
   - Rotation matrix (3x3)
   - Euler angles (XYZ convention, degrees)
   - Translation vector (mm)
3. **Inverse Transformation**: `^Q T_EE` for convenience

### Residual Statistics

The script computes and displays comprehensive residual analysis:

#### Overall Residuals
Statistics across all pairs from all datasets combined.

#### Per-Dataset Residuals
Individual statistics for each dataset to identify problematic data.

### Residual Calculation

For each pair `(A, B)` and calibrated `X`, we compute two types of residuals:

#### 1. Physical Alignment Errors (What You See in Visualization)

These measure the actual position and rotation differences:

```python
AX = A @ X  # Apply EE motion, then transform to Quest frame
XB = X @ B  # Transform to Quest frame, then apply Quest motion
# Ideal: AX should equal XB

translation_error = ||AX.t - XB.t||  # Euclidean distance in mm
rotation_error = ||log(AX.R^T @ XB.R)||  # Angle in radians
```

**These are the errors you see when you visualize the calibrated result.** They represent how well the Quest and EE motions agree after applying the calibration `X`.

#### 2. Equation Closure Errors (SE(3) Logarithm)

These are the mathematical residuals used during optimization:

```python
E = (A @ X) @ inv(X @ B)  # Should be identity if equation is satisfied
residual = log(E)  # 6-DOF vector in se(3)
```

The residual vector `[v_x, v_y, v_z, w_x, w_y, w_z]` contains:
- **Translation component** `[v_x, v_y, v_z]`: Related to but not equal to physical translation error
- **Rotation component** `[w_x, w_y, w_z]`: Axis-angle representation (radians)

**Note**: The equation closure translation component can differ from physical translation error because it's coupled with rotation through the SE(3) logarithm.

#### Residual Metrics

For each component (translation/rotation):

- **RMS (Root Mean Square)**: `sqrt(mean(||residual||^2))`
  - Overall magnitude of errors
  - Good calibration: Translation RMS < 10mm, Rotation RMS < 5°

- **Max**: Maximum error across all pairs
  - Identifies worst-case outliers

- **Mean per axis**: `mean(|residual_x|), mean(|residual_y|), mean(|residual_z|)`
  - Shows if errors are biased in specific directions

- **Std per axis**: Standard deviation per axis
  - Measures error consistency/variance

#### Interpreting Residuals

**Good Calibration** (consecutive-only mode):
- **Equation Closure Translation RMS: < 10mm** ✅
- **Equation Closure Rotation RMS: < 5°** ✅
- Low standard deviations indicate consistent measurements
- **Check per-dataset residuals** - all datasets should have similar error levels

**Note on Residual Values**: The script now reports only "Equation Closure Errors" (SE(3) logarithm), which are the technical residuals used during optimization. These values may appear larger than physical alignment errors you would see in visualization, especially when there are systematic biases in Quest tracking.

**Poor Calibration** (often seen with non-consecutive pairs or bad data):
- Equation Closure Translation RMS: > 50mm
- Equation Closure Rotation RMS: > 20°
- High rotation errors suggest:
  - Quest tracking noise amplified by transform composition
  - Insufficient rotational diversity in data
  - Potential synchronization issues between Quest and EE measurements

**Mixed Results** (some datasets good, others bad):
- Overall RMS dominated by worst dataset
- Use `--per-dataset-residual` to identify problematic datasets
- Example: Dataset 1 has 4mm RMS (excellent), Dataset 2 has 200mm RMS (terrible)
- This indicates **Quest tracking issues** or **synchronization problems** in specific recording sessions
- **Important**: Large residuals in datasets with pure-translation motions often indicate systematic Quest tracking biases that vary between pairs
  - Example: Pair 1 has Quest +8mm error, Pair 2 has Quest -22mm error
  - These contradictory measurements cannot be reconciled by a single rigid transform X
  - The optimizer tries to compromise, resulting in large residuals

**Action Items for Poor Calibration**:
1. **Check sanity check output** - see which pairs were automatically removed
2. **Use `--per-dataset-residual`** to identify which dataset(s) are problematic
3. If one dataset is much worse, consider removing it and re-running calibration
4. Use consecutive-only mode (default) to avoid error amplification
5. If residuals are still high (>100mm) even with sanity check:
   - Quest tracking may have systematic drift that varies between motions
   - Consider lowering `--sanity-threshold` to be more aggressive (e.g., 0.15 or 0.20)
   - Collect new data with better Quest tracking conditions
6. Collect more datasets with diverse rotational motions
7. Ensure Quest and EE measurements are properly synchronized in time
8. Check for Quest tracking loss or drift during data collection

## Data Collection Tips

For best calibration results:

1. **Rotational Diversity**:
   - Move the end-effector through varied orientations
   - Include rotations around all axes (X, Y, Z)
   - Avoid purely translational motions

2. **Multiple Datasets**:
   - Collect 2-3 separate recording sessions
   - Each with 4-6 poses minimum
   - Different motion patterns per dataset

3. **Smooth Motions**:
   - Move slowly between poses for stable Quest tracking
   - Allow Quest tracking to settle before recording each pose
   - Avoid fast rotations that may cause tracking loss

4. **Synchronization**:
   - Ensure Quest pose and EE pose are captured at the same time
   - Use proper time synchronization between systems

## Example

```bash
# Run calibration with default settings (consecutive-only)
python calculate_quest_to_base.py data/left_hand_right_quest.json

# Output:
# Found 2 dataset(s).
# Using ONLY consecutive pairs (recommended)
# Dataset 1: 5 poses -> 4 consecutive pairs
# Dataset 2: 4 poses -> 3 consecutive pairs
# ...
# Final Residual Statistics:
#   Translation errors (mm):
#     RMS:     3.47 mm
#     Max:     6.14 mm
#   Rotation errors (degrees):
#     RMS:     3.06°
#     Max:     3.74°
```

## Troubleshooting

### Large Rotation Errors with --all-pairs

**Problem**: Rotation RMS > 50° when using `--all-pairs`

**Solution**: Use default consecutive-only mode. Quest tracking noise accumulates when composing multiple transforms.

### Large Translation Errors

**Problem**: Translation RMS > 50mm even with sanity check enabled

**Possible Causes**:
- **Systematic Quest tracking drift** that varies between motions (most common)
  - Example: Some pairs have Quest over-reporting distance (+8mm), others under-reporting (-22mm)
  - These contradictory measurements cannot be satisfied by a single rigid transform
- Insufficient rotational diversity in data
- Incorrect unit conversion (check meters vs millimeters)
- Desynchronized Quest and EE measurements

**Solutions**:
1. **Lower sanity threshold** to be more aggressive:
   ```bash
   python calculate_quest_to_base.py data.json --sanity-threshold 0.15
   ```
2. Use `--per-dataset-residual` to identify worst datasets and remove them
3. Collect new data with more varied rotations
4. Ensure proper time synchronization
5. Check that `W_T_Q` is in meters and `EEi_T_EEi1` is in millimeters
6. Improve Quest tracking conditions (lighting, occlusion, etc.)

### Sanity Check Removing Too Many Pairs

**Problem**: Most pairs are being removed by sanity check

**Solution**: 
- This indicates serious Quest tracking issues
- Review Quest tracking quality during data collection
- Consider collecting new data in better conditions
- If you're confident the data is good, you can raise the threshold:
  ```bash
  python calculate_quest_to_base.py data.json --sanity-threshold 0.35
  ```
- Or disable it entirely (not recommended):
  ```bash
  python calculate_quest_to_base.py data.json --no-sanity-check
  ```

### One Dataset Has Much Worse Residuals

**Problem**: Per-dataset residuals show one dataset with 10x larger errors

**Solution**: 
- Review that specific dataset for data quality issues
- Consider removing the problematic dataset
- Re-record that session with better Quest tracking conditions

## Technical Details

### SE(3) Manifold Optimization

The calibration uses proper Lie group optimization on SE(3):
- **SE(3)**: Special Euclidean group (rigid transformations)
- **se(3)**: Tangent space (6-DOF velocities)
- **Exponential map**: `Exp: se(3) -> SE(3)`
- **Logarithm map**: `Log: SE(3) -> se(3)`

Benefits:
- Maintains rotation matrix orthonormality
- Proper metric on transformation space
- Stable optimization without gimbal lock

### Unit Conversions

The code handles two different unit conventions:
- **Quest tracking**: Typically in meters (VR/AR convention)
- **Robot kinematics**: Typically in millimeters (robotics convention)

Internal processing:
- Convert all to millimeters for consistency
- Final output `X` has translation in millimeters
- Residuals reported in millimeters and degrees

## Files

- `calculate_quest_to_base.py`: Main calibration script
- `data/`: Directory for input JSON files
  - `left_hand_right_quest.json`: Example dataset
  - `right_hand_left_quest.json`: Example dataset
- `README.md`: This documentation
