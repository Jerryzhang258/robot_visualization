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
    T = SE3(np.eye(3), np.zeros(3))
    for k in range(i, j):
        T = T @ A_list[k]
    return T

def build_pairs_from_dataset(W_T_Q: List[SE3], EEi_T_EEi1: List[SE3]) -> List[Tuple[SE3, SE3]]:
    """
    Returns list of (A_ij, B_ij) for ONLY CONSECUTIVE pairs (i, i+1).
    Using only consecutive pairs avoids accumulated errors from composing multiple motions.
    
    Units:
      - W_T_Q translations are already converted to mm
      - EEi_T_EEi1 translations are already in mm
      => A and B are both in mm.
    """
    n = len(W_T_Q)
    if len(EEi_T_EEi1) != n - 1:
        raise ValueError(f"EEi_T_EEi1 length must be n-1. Got n={n}, len(EE)={len(EEi_T_EEi1)}")

    pairs = []
    # Only use consecutive pairs to avoid accumulated errors
    for i in range(n - 1):
        j = i + 1
        A_ij = EEi_T_EEi1[i]  # Direct consecutive motion
        # IMPORTANT: Use A_ij^-1 to match the correct hand-eye equation
        # The equation is: A^-1 @ X = X @ B
        A_ij_inv = A_ij.inv()
        B_ij = W_T_Q[i].inv() @ W_T_Q[j]
        pairs.append((A_ij_inv, B_ij))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="Path to input JSON (4x4 matrices).")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = data.get("datasets", [])
    if not datasets:
        raise SystemExit("No datasets found.")

    print(f"Found {len(datasets)} dataset(s).")
    
    # Collect pairs from ALL datasets (each dataset has its own consistent frame)
    all_dataset_pairs = []
    
    for d_idx, d in enumerate(datasets):
        W_T_Q_raw = d["W_T_Q"]
        EE_raw = d["EEi_T_EEi1"]

        # W_T_Q translation: meters -> mm
        W_T_Q = [se3_from_matrix(matrix_from_nested_list(M), translation_scale=M_TO_MM) for M in W_T_Q_raw]

        # EEi_T_EEi1 translation: already in mm
        EEi_T_EEi1 = [se3_from_matrix(matrix_from_nested_list(M), translation_scale=1.0) for M in EE_raw]

        pairs = build_pairs_from_dataset(W_T_Q, EEi_T_EEi1)
        print(f"Dataset {d_idx + 1}: Generated {len(pairs)} consecutive pairs")
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
    
    print(f"\nUsing dataset {best_rotation_dataset_idx + 1} for rotation initialization (highest rotational variance)")
    
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

    print("Inverse ^EE T_Q (4x4), translation in mm:")
    print(X_inv_mat)
    print("\nInverse rotation matrix:")
    print(X_inv.R)
    print("\nInverse euler_xyz_deg:")
    print(euler_xyz_deg_from_R(X_inv.R).tolist())
    print("\nInverse translation (mm):")
    print(X_inv.t.tolist())


if __name__ == "__main__":
    main()
