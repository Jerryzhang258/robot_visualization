#!/usr/bin/env python3
"""
Manually solve hand-eye calibration to understand the correct formulation.
"""

import numpy as np
import json
from scipy.spatial.transform import Rotation as R

# Load data
with open("src/finger_calibrate/quest_calibrate/data/trial1.json", "r") as f:
    data = json.load(f)

dataset = data["datasets"][0]
W_T_Q = [np.array(m, dtype=float) for m in dataset["W_T_Q"]]
EEi_T_EEi1 = [np.array(m, dtype=float) for m in dataset["EEi_T_EEi1"]]

print("=== Manual Hand-Eye Calibration ===\n")

# Expected rotation
expected_R = np.array([
    [ 0.0859359024475257, -0.692080469355965, -0.717657359150919],
    [ 0.467267593954552,  -0.622605638745331,  0.628465481383785],
    [-0.867300425779312,  -0.425287225517091,  0.261701704972375]
])

# Let's manually construct A and B for the first pair and solve
# using scipy.spatial.transform.Rotation.align_vectors

a_vecs = []
b_vecs = []

for i in range(len(EEi_T_EEi1)):
    j = i + 1
    
    # A: EE motion from i to j
    A = EEi_T_EEi1[i].copy()
    
    # B: Quest motion from i to j
    B = np.linalg.inv(W_T_Q[i]) @ W_T_Q[j]
    
    # Extract rotation axis-angles
    a = R.from_matrix(A[:3, :3]).as_rotvec()
    b = R.from_matrix(B[:3, :3]).as_rotvec()
    
    print(f"Pair {i}->{j}:")
    print(f"  A axis-angle: {a}")
    print(f"  B axis-angle: {b}")
    print(f"  ||a||: {np.linalg.norm(a):.4f}, ||b||: {np.linalg.norm(b):.4f}")
    
    if np.linalg.norm(a) > 1e-3 and np.linalg.norm(b) > 1e-3:
        a_vecs.append(a)
        b_vecs.append(b)

print(f"\nUsing {len(a_vecs)} pairs for rotation estimation\n")

# Method 1: align_vectors(a, b) -> finds R such that R @ b ≈ a
# For AX = XB: log(A) ≈ Ad_X(log(B)) (adjoint action, not simple)
# So align_vectors doesn't directly apply...

# Method 2: Let's try direct correspondence
# If A_i @ X = X @ B_i for all i, then for rotations:
# R_A[i] @ R_X = R_X @ R_B[i]
# This means R_X commutes with certain rotations - very constrained!

# Method 3: Maybe the correspondence is: R_X aligns axis(B) to axis(A)?
print("Method: Aligning B rotation axes to A rotation axes")
rot1, _= R.align_vectors(np.array(a_vecs), np.array(b_vecs))
X1 = rot1.as_matrix()
print("X from align_vectors(a, b):")
print(X1)
print()

# Test it
errors1 = []
for i in range(len(EEi_T_EEi1)):
    A = EEi_T_EEi1[i][:3, :3]
    B = (np.linalg.inv(W_T_Q[i]) @ W_T_Q[i+1])[:3, :3]
    
    left = A @ X1
    right = X1 @ B
    error = np.linalg.norm(left - right, 'fro')
    errors1.append(error)
    
print(f"Average error: {np.mean(errors1):.6f}\n")

# Method 4: align_vectors(b, a) -> finds R such that R @ a ≈ b
print("Method: Aligning A rotation axes to B rotation axes")
rot2, _ = R.align_vectors(np.array(b_vecs), np.array(a_vecs))
X2 = rot2.as_matrix()
print("X from align_vectors(b, a):")
print(X2)
print()

errors2 = []
for i in range(len(EEi_T_EEi1)):
    A = EEi_T_EEi1[i][:3, :3]
    B = (np.linalg.inv(W_T_Q[i]) @ W_T_Q[i+1])[:3, :3]
    
    left = A @ X2
    right = X2 @ B
    error = np.linalg.norm(left - right, 'fro')
    errors2.append(error)
    
print(f"Average error: {np.mean(errors2):.6f}\n")

# Let's also try with A inverted
print("Method: Using A^-1, align_vectors(a, b)")
a_vecs_inv = []
for i in range(len(EEi_T_EEi1)):
    A = EEi_T_EEi1[i][:3, :3]
    A_inv = np.linalg.inv(A)
    a_inv = R.from_matrix(A_inv).as_rotvec()
    if np.linalg.norm(a_inv) > 1e-3:
        a_vecs_inv.append(a_inv)

rot3, _ = R.align_vectors(np.array(a_vecs_inv), np.array(b_vecs))
X3 = rot3.as_matrix()
print("X from align_vectors(a_inv, b):")
print(X3)
print()

errors3 = []
for i in range(len(EEi_T_EEi1)):
    A = EEi_T_EEi1[i][:3, :3]
    B = (np.linalg.inv(W_T_Q[i]) @ W_T_Q[i+1])[:3, :3]
    A_inv = np.linalg.inv(A)
    
    left = A_inv @ X3
    right = X3 @ B
    error = np.linalg.norm(left - right, 'fro')
    errors3.append(error)
    
print(f"Average error (for A^-1 @ X = X @ B): {np.mean(errors3):.6f}\n")

# Compare with expected
print("=== Comparison with Expected ===")
print(f"X1 vs expected: {np.linalg.norm(X1 - expected_R, 'fro'):.6f}")
print(f"X2 vs expected: {np.linalg.norm(X2 - expected_R, 'fro'):.6f}")
print(f"X3 vs expected: {np.linalg.norm(X3 - expected_R, 'fro'):.6f}")
