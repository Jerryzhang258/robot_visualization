#!/usr/bin/env python3
"""
Hand–eye calibration (AX = XB) with SE(3) refinement.

Unknown constant mount:
    X =  ^EE T_Q   (constant across all datasets)

For each dataset d with poses i=0..n-1:
    - measured (noisy):  ^W_d T_{Q,i}   as 4x4 matrices, translation in METERS
    - known EE relative motion between consecutive poses:
            A_{d,i} = ^EE_{i} T_{EE_{i+1}}   as 4x4 matrices, translation in MILLIMETERS
Datasets may be disjoint.

We use only within-dataset relative motions:
    B_{d,ij} = inv(^W T_Q,i) * (^W T_Q,j)      (translation in meters)
    A_{d,ij} = A_{d,i} * ... * A_{d,j-1}       (translation in millimeters)

To solve consistently, we convert B translations to millimeters (meters * 1000).
The refined X is output as a 4x4 matrix with translation in MILLIMETERS.
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
    from scipy.optimize import least_squares
except ImportError as e:
    raise SystemExit(
        "This script requires scipy. Install with: pip install scipy\n"
        f"ImportError: {e}"
    )

M_TO_MM = 1000.0


# ----------------------------- SE(3) utilities -----------------------------

def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

@dataclass
class SE3:
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3,

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inv(self) -> "SE3":
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t)

    def __matmul__(self, other: "SE3") -> "SE3":
        return SE3(self.R @ other.R, self.t + self.R @ other.t)

def so3_exp(w: np.ndarray) -> np.ndarray:
    return R.from_rotvec(w).as_matrix()

def so3_log(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_rotvec()

def left_jacobian_SO3(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    W = _skew(w)
    I = np.eye(3)
    if theta < 1e-8:
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    A = math.sin(theta) / theta
    B = (1.0 - math.cos(theta)) / (theta * theta)
    C = (1.0 - A) / (theta * theta)
    return I + B * W + C * (W @ W)

def left_jacobian_inv_SO3(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    W = _skew(w)
    I = np.eye(3)
    if theta < 1e-8:
        return I - 0.5 * W + (1.0 / 12.0) * (W @ W)
    half = 0.5 * theta
    cot_half = math.cos(half) / math.sin(half)
    return I - 0.5 * W + ((1.0 / (theta * theta)) * (1.0 - half * cot_half)) * (W @ W)

def se3_exp(xi: np.ndarray) -> SE3:
    v = xi[:3]
    w = xi[3:]
    Rm = so3_exp(w)
    J = left_jacobian_SO3(w)
    t = J @ v
    return SE3(Rm, t)

def se3_log(T: SE3) -> np.ndarray:
    w = so3_log(T.R)
    Jinv = left_jacobian_inv_SO3(w)
    v = Jinv @ T.t
    return np.hstack([v, w])


# ----------------------------- IO conversions -----------------------------

def _check_homogeneous(T: np.ndarray) -> None:
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=float), atol=1e-6):
        raise ValueError(f"Bottom row is not [0,0,0,1]: {T[3, :]}")

def se3_from_matrix(T: np.ndarray, translation_scale: float = 1.0) -> SE3:
    """
    translation_scale multiplies the translation component.
    Use translation_scale=1000 for meters->millimeters conversion.
    """
    T = np.asarray(T, dtype=float)
    _check_homogeneous(T)
    Rm = T[:3, :3]
    t = T[:3, 3] * translation_scale
    return SE3(Rm, t)

def matrix_from_nested_list(M) -> np.ndarray:
    T = np.array(M, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4, got {T.shape}")
    return T

def euler_xyz_deg_from_R(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_euler("xyz", degrees=True)


# ----------------------------- Hand–eye build -----------------------------

def compose_chain(A_list: List[SE3], i: int, j: int) -> SE3:
    """Compose A_i ... A_{j-1} (i<j)."""
    T = A_list[i]
    for k in range(i+1, j):
        T = T @ A_list[k]
    return T

def build_pairs_from_dataset(W_T_Q: List[SE3], EEi_T_EEi1: List[SE3], use_all_pairs: bool = True, max_gap: int = 3, 
                            dataset_idx: int = 0, sanity_check: bool = True, sanity_threshold: float = 0.25) -> List[Tuple[SE3, SE3]]:
    """
    Returns list of (A_ij, B_ij) pairs.
    
    If use_all_pairs=True (default):
      - Uses pairs (i, j) where i < j and j-i <= max_gap
      - A_ij is computed by composing the chain of consecutive transforms
      - max_gap limits how many transforms are composed to avoid error accumulation
      - For n poses with max_gap=3, generates ~3*(n-1) pairs
    
    If use_all_pairs=False:
      - Uses only consecutive pairs (i, i+1)
      - Avoids potential error accumulation from composing multiple motions
      - For n poses, generates n-1 pairs
    
    Sanity check (if enabled):
      - Compares EE translation distance with Quest translation distance
      - Removes pairs where they differ by more than sanity_threshold (default 25%)
      - Helps detect Quest tracking errors or desynchronization
    
    Units:
      - W_T_Q translations are already converted to mm
      - EEi_T_EEi1 translations are already in mm
      => A and B are both in mm.
    """
    n = len(W_T_Q)
    if len(EEi_T_EEi1) != n - 1:
        raise ValueError(f"EEi_T_EEi1 length must be n-1. Got n={n}, len(EE)={len(EEi_T_EEi1)}")

    pairs = []
    removed_pairs = []
    skipped_rotation_pairs = []
    
    if use_all_pairs:
        # Use pairs with limited gap to avoid error accumulation
        for i in range(n - 1):
            for j in range(i + 1, min(n, i + max_gap + 1)):
                # Compose A_i, A_{i+1}, ..., A_{j-1}
                A_ij = compose_chain(EEi_T_EEi1, i, j)
                # IMPORTANT: Use A_ij^-1 to match the correct hand-eye equation
                # The equation is: A^-1 @ X = X @ B
                A_ij_inv = A_ij.inv()
                B_ij = W_T_Q[i].inv() @ W_T_Q[j]
                
                # Sanity check: compare translation distances (only for low-rotation EE pairs)
                if sanity_check:
                    # Check EE rotation magnitude only (Quest rotation may be noisy)
                    ee_rot_angle = np.linalg.norm(so3_log(A_ij.R))
                    quest_rot_angle = np.linalg.norm(so3_log(B_ij.R))
                    
                    # Only check translation if EE rotation is small (<5 degrees = 0.087 radians)
                    # This ensures we're comparing translation distances in a meaningful way
                    if ee_rot_angle < 0.087:
                        ee_dist = np.linalg.norm(A_ij.t)
                        quest_dist = np.linalg.norm(B_ij.t)
                        
                        if ee_dist > 1e-3:  # Only check if there's significant EE motion
                            rel_error = abs(ee_dist - quest_dist) / ee_dist
                            if rel_error > sanity_threshold:
                                removed_pairs.append((i, j, ee_dist, quest_dist, rel_error, 
                                                     np.rad2deg(ee_rot_angle), np.rad2deg(quest_rot_angle)))
                                continue
                    else:
                        # Track pairs skipped due to EE rotation
                        skipped_rotation_pairs.append((i, j, np.rad2deg(ee_rot_angle), np.rad2deg(quest_rot_angle)))
                
                pairs.append((A_ij_inv, B_ij))
    else:
        # Only use consecutive pairs
        for i in range(n - 1):
            j = i + 1
            A_ij = EEi_T_EEi1[i]  # Direct consecutive motion
            # IMPORTANT: Use A_ij^-1 to match the correct hand-eye equation
            # The equation is: A^-1 @ X = X @ B
            A_ij_inv = A_ij.inv()
            B_ij = W_T_Q[i].inv() @ W_T_Q[j]
            
            # Sanity check: compare translation distances (only for low-rotation EE pairs)
            if sanity_check:
                # Check EE rotation magnitude only (Quest rotation may be noisy)
                ee_rot_angle = np.linalg.norm(so3_log(A_ij.R))
                quest_rot_angle = np.linalg.norm(so3_log(B_ij.R))
                
                # Only check translation if EE rotation is small (<5 degrees = 0.087 radians)
                # This ensures we're comparing translation distances in a meaningful way
                if ee_rot_angle < 0.087:
                    ee_dist = np.linalg.norm(A_ij.t)
                    quest_dist = np.linalg.norm(B_ij.t)
                    
                    if ee_dist > 1e-3:  # Only check if there's significant EE motion
                        rel_error = abs(ee_dist - quest_dist) / ee_dist
                        if rel_error > sanity_threshold:
                            removed_pairs.append((i, j, ee_dist, quest_dist, rel_error,
                                                 np.rad2deg(ee_rot_angle), np.rad2deg(quest_rot_angle)))
                            continue
                else:
                    # Track pairs skipped due to EE rotation
                    skipped_rotation_pairs.append((i, j, np.rad2deg(ee_rot_angle), np.rad2deg(quest_rot_angle)))
            
            pairs.append((A_ij_inv, B_ij))
    
    # Print warnings for removed pairs
    if removed_pairs:
        print(f"\n⚠️  Dataset {dataset_idx + 1}: Sanity check failed for {len(removed_pairs)} pair(s)")
        print(f"   Removed pairs where Quest/EE translation distance differs by >{sanity_threshold*100:.0f}%:")
        for i, j, ee_dist, quest_dist, rel_error, ee_rot, quest_rot in removed_pairs:
            print(f"   - Pair ({i}, {j}): EE={ee_dist:.2f}mm, Quest={quest_dist:.2f}mm, Error={rel_error*100:.1f}%")
            print(f"                     (EE_rot={ee_rot:.2f}°, Quest_rot={quest_rot:.2f}°)")
    
    if skipped_rotation_pairs and sanity_check:
        print(f"   ℹ️  Skipped sanity check for {len(skipped_rotation_pairs)} pair(s) with rotation >5°")
    
    return pairs


# ----------------------------- Initialization -----------------------------

def init_rotation_align_vectors(pairs: List[Tuple[SE3, SE3]]) -> np.ndarray:
    a_vecs = []
    b_vecs = []
    for A, B in pairs:
        a = so3_log(A.R)
        b = so3_log(B.R)
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            continue
        a_vecs.append(a)
        b_vecs.append(b)

    if len(a_vecs) < 2:
        return np.eye(3)

    a_mat = np.vstack(a_vecs)
    b_mat = np.vstack(b_vecs)
    rot, _ = R.align_vectors(a_mat, b_mat)  # rot.apply(b) ≈ a
    return rot.as_matrix()

def init_translation_least_squares(Rx: np.ndarray, pairs: List[Tuple[SE3, SE3]]) -> np.ndarray:
    I = np.eye(3)
    M_blocks = []
    y_blocks = []

    for A, B in pairs:
        M = (A.R - I)
        y = (Rx @ B.t - A.t)
        if np.linalg.norm(M) < 1e-9:
            continue
        M_blocks.append(M)
        y_blocks.append(y.reshape(3, 1))

    if not M_blocks:
        return np.zeros(3)

    M_all = np.vstack(M_blocks)
    y_all = np.vstack(y_blocks).reshape(-1)
    tX, *_ = np.linalg.lstsq(M_all, y_all, rcond=None)
    return tX


# ----------------------------- Refinement -----------------------------

def compute_residuals(pairs: List[Tuple[SE3, SE3]], X: SE3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute residuals for each pair under the hand-eye equation A^-1 @ X = X @ B.
    
    Returns:
        equation_translation_errors: (n, 3) SE(3) logarithm translation component
        equation_rotation_errors: (n, 3) SE(3) logarithm rotation component (axis-angle, radians)
        physical_translation_errors: (n,) physical position difference in mm
        physical_rotation_errors: (n,) physical rotation angle difference in radians
    """
    equation_translation_errors = []
    equation_rotation_errors = []
    physical_translation_errors = []
    physical_rotation_errors = []
    
    for A, B in pairs:
        # Equation closure error: E = (A @ X) @ (X @ B)^-1
        XB = X @ B
        AX = A @ X
        E = AX @ XB.inv()
        
        # Extract SE(3) logarithm components (used for optimization)
        xi = se3_log(E)
        equation_translation_errors.append(xi[:3])
        equation_rotation_errors.append(xi[3:])
        
        # Physical errors: direct comparison of AX vs XB
        # Translation: Euclidean distance between positions
        pos_error = np.linalg.norm(AX.t - XB.t)
        physical_translation_errors.append(pos_error)
        
        # Rotation: angle between rotation matrices
        R_error = AX.R.T @ XB.R  # Relative rotation
        rot_angle = np.linalg.norm(so3_log(R_error))
        physical_rotation_errors.append(rot_angle)
    
    return (np.array(equation_translation_errors), 
            np.array(equation_rotation_errors),
            np.array(physical_translation_errors),
            np.array(physical_rotation_errors))

def print_residual_statistics(pairs: List[Tuple[SE3, SE3]], X: SE3, label: str = ""):
    """
    Compute and print statistics about residuals.
    """
    eq_trans_err, eq_rot_err, phys_trans_err, phys_rot_err = compute_residuals(pairs, X)
    
    # Physical errors (what you see when visualizing)
    phys_trans_rms = np.sqrt(np.mean(phys_trans_err**2))
    phys_trans_max = np.max(phys_trans_err)
    phys_trans_mean = np.mean(phys_trans_err)
    
    phys_rot_deg = np.rad2deg(phys_rot_err)
    phys_rot_rms_deg = np.sqrt(np.mean(phys_rot_deg**2))
    phys_rot_max_deg = np.max(phys_rot_deg)
    phys_rot_mean_deg = np.mean(phys_rot_deg)
    
    # Equation closure errors (SE(3) logarithm - used for optimization)
    eq_trans_norms = np.linalg.norm(eq_trans_err, axis=1)
    eq_trans_rms = np.sqrt(np.mean(eq_trans_norms**2))
    eq_trans_max = np.max(eq_trans_norms)
    eq_trans_mean_per_axis = np.mean(np.abs(eq_trans_err), axis=0)
    eq_trans_std_per_axis = np.std(eq_trans_err, axis=0)
    
    eq_rot_norms_rad = np.linalg.norm(eq_rot_err, axis=1)
    eq_rot_norms_deg = np.rad2deg(eq_rot_norms_rad)
    eq_rot_rms_deg = np.sqrt(np.mean(eq_rot_norms_deg**2))
    eq_rot_max_deg = np.max(eq_rot_norms_deg)
    eq_rot_mean_per_axis_deg = np.rad2deg(np.mean(np.abs(eq_rot_err), axis=0))
    eq_rot_std_per_axis_deg = np.rad2deg(np.std(eq_rot_err, axis=0))
    
    prefix = f"{label} " if label else ""
    print(f"\n{prefix}Residual Statistics:")
    
    # # Show physical errors first (most intuitive)
    # print(f"\n  Physical Alignment Errors (direct comparison of A@X vs X@B):")
    # print(f"    Translation (position difference):")
    # print(f"      RMS:     {phys_trans_rms:.6f} mm")
    # print(f"      Max:     {phys_trans_max:.6f} mm")
    # print(f"      Mean:    {phys_trans_mean:.6f} mm")
    
    # print(f"    Rotation (angle difference):")
    # print(f"      RMS:     {phys_rot_rms_deg:.6f}°")
    # print(f"      Max:     {phys_rot_max_deg:.6f}°")
    # print(f"      Mean:    {phys_rot_mean_deg:.6f}°")
    
    # Show equation closure errors (technical details)
    print(f"\n  Equation Closure Errors (SE(3) logarithm - used for optimization):")
    print(f"    Translation component:")
    print(f"      RMS:     {eq_trans_rms:.6f} mm")
    print(f"      Max:     {eq_trans_max:.6f} mm")
    print(f"      Mean per axis [X, Y, Z]: [{eq_trans_mean_per_axis[0]:.6f}, {eq_trans_mean_per_axis[1]:.6f}, {eq_trans_mean_per_axis[2]:.6f}] mm")
    print(f"      Std per axis  [X, Y, Z]: [{eq_trans_std_per_axis[0]:.6f}, {eq_trans_std_per_axis[1]:.6f}, {eq_trans_std_per_axis[2]:.6f}] mm")
    
    print(f"    Rotation component:")
    print(f"      RMS:     {eq_rot_rms_deg:.6f}°")
    print(f"      Max:     {eq_rot_max_deg:.6f}°")
    print(f"      Mean per axis [X, Y, Z]: [{eq_rot_mean_per_axis_deg[0]:.6f}, {eq_rot_mean_per_axis_deg[1]:.6f}, {eq_rot_mean_per_axis_deg[2]:.6f}]°")
    print(f"      Std per axis  [X, Y, Z]: [{eq_rot_std_per_axis_deg[0]:.6f}, {eq_rot_std_per_axis_deg[1]:.6f}, {eq_rot_std_per_axis_deg[2]:.6f}]°")

def refine_X(pairs: List[Tuple[SE3, SE3]], X0: SE3) -> SE3:
    """
    Optimize delta in se(3): X = X0 * Exp(delta)
    All translations are in mm.
    """
    def residual(delta: np.ndarray) -> np.ndarray:
        X = X0 @ se3_exp(delta)
        res_list = []
        for A, B in pairs:
            XB = X @ B
            E = (A @ X) @ XB.inv()
            res_list.append(se3_log(E))
        return np.concatenate(res_list, axis=0)

    sol = least_squares(
        residual,
        np.zeros(6),
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=200
    )
    return X0 @ se3_exp(sol.x)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hand-eye calibration (AX=XB) with SE(3) refinement.",
        epilog="\nIMPORTANT: Using non-consecutive pairs can amplify rotation errors when Quest "
               "tracking has noise. Consecutive-only mode (default) is strongly recommended.\n"
    )
    ap.add_argument("input_json", help="Path to input JSON (4x4 matrices).")
    ap.add_argument("--all-pairs", action="store_true",
                    help="Use all possible pairs with max-gap limit. WARNING: This significantly "
                         "amplifies rotation errors when Quest tracking is noisy. Not recommended.")
    ap.add_argument("--max-gap", type=int, default=1,
                    help="Maximum gap between poses for pairing (default: 1 = consecutive only). "
                         "Higher values compose more transforms and amplify noise. Use with caution.")
    ap.add_argument("--per-dataset-residual", action="store_true",
                    help="Print per-dataset residuals to identify problematic datasets.")
    ap.add_argument("--no-sanity-check", action="store_true",
                    help="Disable sanity check that removes pairs with large Quest/EE distance mismatch.")
    ap.add_argument("--sanity-threshold", type=float, default=0.25,
                    help="Relative error threshold for sanity check (default: 0.25 = 25%%). "
                         "Pairs with Quest/EE distance difference >threshold are removed.")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = data.get("datasets", [])
    if not datasets:
        raise SystemExit("No datasets found.")

    print(f"Found {len(datasets)} dataset(s).")
    
    # Determine pairing strategy
    if args.all_pairs:
        use_consecutive_only = False
        max_gap = args.max_gap
        print(f"⚠️  WARNING: Using non-consecutive pairs (max_gap={max_gap})")
        print(f"   This can significantly amplify rotation errors from Quest tracking noise!")
        print(f"   Consider using consecutive-only mode (default) for better results.")
    else:
        use_consecutive_only = True
        max_gap = 1
        print("Using ONLY consecutive pairs (recommended)")
    
    # Sanity check settings
    sanity_check = not args.no_sanity_check
    if sanity_check:
        print(f"Sanity check enabled: removing pairs with Quest/EE distance mismatch >{args.sanity_threshold*100:.0f}%")
    else:
        print("Sanity check disabled")
    
    # Collect pairs from ALL datasets (each dataset has its own consistent frame)
    all_dataset_pairs = []
    
    # use_all_pairs = False
    for d_idx, d in enumerate(datasets):
        W_T_Q_raw = d["W_T_Q"]
        EE_raw = d["EEi_T_EEi1"]

        # W_T_Q translation: meters -> mm
        W_T_Q = [se3_from_matrix(matrix_from_nested_list(M), translation_scale=M_TO_MM) for M in W_T_Q_raw]

        # EEi_T_EEi1 translation: already in mm
        EEi_T_EEi1 = [se3_from_matrix(matrix_from_nested_list(M), translation_scale=1.0) for M in EE_raw]

        pairs = build_pairs_from_dataset(W_T_Q, EEi_T_EEi1, 
                                        use_all_pairs=not use_consecutive_only, 
                                        max_gap=max_gap,
                                        dataset_idx=d_idx,
                                        sanity_check=sanity_check,
                                        sanity_threshold=args.sanity_threshold)
        n_poses = len(W_T_Q)
        if not use_consecutive_only:
            print(f"Dataset {d_idx + 1}: {n_poses} poses -> {len(pairs)} pairs (max_gap={max_gap})")
        else:
            print(f"Dataset {d_idx + 1}: {n_poses} poses -> {len(pairs)} consecutive pairs")
        all_dataset_pairs.append(pairs)
    
    # Strategy: Use dataset with most rotational motion for initialization
    # Then use ALL datasets for refinement (they all constrain the same X)
    
    # Find dataset with most rotational variation
    best_rotation_dataset_idx = 0
    max_rotation_variance = 0
    
    for d_idx, pairs in enumerate(all_dataset_pairs):
        if len(pairs) == 0:
            continue
        rotation_angles = []
        for A, B in pairs:
            angle_a = np.linalg.norm(so3_log(A.R))
            angle_b = np.linalg.norm(so3_log(B.R))
            rotation_angles.append(angle_a)
            rotation_angles.append(angle_b)
        if len(rotation_angles) > 0:
            variance = np.var(rotation_angles)
            if variance > max_rotation_variance:
                max_rotation_variance = variance
                best_rotation_dataset_idx = d_idx
    
    print(f"\nUsing dataset {best_rotation_dataset_idx + 1} for translation initialization (highest rotational variance)")

    # find dataset with most translational variation
    best_translation_dataset_idx = 0
    max_translation_variance = 0

    for d_idx, pairs in enumerate(all_dataset_pairs):
        if len(pairs) == 0:
            continue
        translation_magnitudes = []
        for A, B in pairs:
            mag_a = np.linalg.norm(A.t)
            mag_b = np.linalg.norm(B.t)
            translation_magnitudes.append(mag_a)
            translation_magnitudes.append(mag_b)
        if len(translation_magnitudes) > 0:
            variance = np.var(translation_magnitudes)
            if variance > max_translation_variance:
                max_translation_variance = variance
                best_translation_dataset_idx = d_idx
    print(f"Using dataset {best_translation_dataset_idx + 1} for rotation initialization (highest translational variance)")

    # Flatten all pairs for refinement
    all_pairs = []
    for pairs in all_dataset_pairs:
        all_pairs.extend(pairs)
    
    print(f"Total pairs across all datasets: {len(all_pairs)}\n")

    if len(all_pairs) < 3:
        raise SystemExit(f"Not enough (A,B) pairs. Got {len(all_pairs)}; need more data / more poses per dataset.")

    # Init using best rotation dataset
    Rx = init_rotation_align_vectors(all_dataset_pairs[best_rotation_dataset_idx])
    tx = init_translation_least_squares(Rx, all_dataset_pairs[best_rotation_dataset_idx])
    X0 = SE3(Rx, tx)
    
    # Output initial estimate
    X0_mat = X0.as_matrix()
    eul_deg_0 = euler_xyz_deg_from_R(X0.R)
    
    np.set_printoptions(precision=6, suppress=True)
    print("Initial estimate ^EE T_Q (4x4), translation in mm:")
    print(X0_mat)
    print("\nInitial rotation matrix:")
    print(X0.R)
    print("\nInitial euler_xyz_deg:")
    print(eul_deg_0.tolist())
    print()

    # Refine using ALL pairs from ALL datasets
    print("Refining with all datasets combined...")
    X = refine_X(all_pairs, X0)

    # Output: translation in mm
    X_mat = X.as_matrix()
    eul_deg = euler_xyz_deg_from_R(X.R)

    # take inverse of X
    X_inv = X.inv()
    X_inv_mat = X_inv.as_matrix()

    print("\n" + "="*60)
    print("Refined ^EE T_Q (4x4), translation in mm:")
    print(X_mat)
    print("\nRefined rotation matrix:")
    print(X.R)
    print("\nRefined euler_xyz_deg:")
    print(eul_deg.tolist())
    print("\nRefined translation (mm):")
    print(X.t.tolist())
    print("="*60)

    print("\nInverse ^EE T_Q (4x4), translation in mm:")
    print(X_inv_mat)
    print("\nInverse rotation matrix:")
    print(X_inv.R)
    print("\nInverse euler_xyz_deg:")
    print(euler_xyz_deg_from_R(X_inv.R).tolist())
    print("\nInverse translation (mm):")
    print(X_inv.t.tolist())
    
    # Print residual statistics
    print_residual_statistics(all_pairs, X, label="Final")
    
    if args.per_dataset_residual:
        # Print per-dataset residuals to identify problematic datasets
        print("\n" + "="*60)
        print("Per-Dataset Residuals:")
        print("="*60)
        for d_idx, pairs in enumerate(all_dataset_pairs):
            if len(pairs) > 0:
                print(f"\nDataset {d_idx + 1} ({len(pairs)} pairs):")
                print_residual_statistics(pairs, X, label="")


if __name__ == "__main__":
    main()
