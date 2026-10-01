"""
From-scratch linear system solvers: Gauss elimination and LU
decomposition with partial pivoting, plus a thin scipy-sparse
fallback for production-size runs.

This module is the direct numerical-analysis contribution the
course project extends the base paper with: PDE-Agents calls out
to DOLFINx/PETSc as an opaque solver, whereas here the linear
system produced by FEM assembly (K + M/dt) T = b is solved with
the same algorithms taught in CSE401 (systems of equations unit).

Both solvers are vectorized with numpy row operations (not pure
Python triple loops) so they stay usable up to a few thousand
unknowns -- enough to reproduce a mesh-refinement convergence
study without needing sparse-matrix machinery for correctness
validation.
"""
from __future__ import annotations
import numpy as np


def gauss_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b via Gauss elimination with partial pivoting.
    A: (n,n) dense array, b: (n,) or (n,1). Returns x: (n,)."""
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D matrix")
    if b.shape[0] != n:
        raise ValueError("dimension mismatch between A and b")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("A and b must contain finite values")

    scale = max(1.0, float(np.max(np.abs(A))))  # for scale-aware singularity check

    # Forward elimination with partial pivoting
    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
        if abs(A[k, k]) < 1e-14 * scale:
            raise np.linalg.LinAlgError(f"singular matrix (zero pivot at column {k})")
        factors = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    # Check the final diagonal before back-substitution
    if abs(A[n - 1, n - 1]) < 1e-14 * scale:
        raise np.linalg.LinAlgError("singular matrix (zero final diagonal)")

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def lu_decompose(A: np.ndarray):
    """LU decomposition with partial pivoting: PA = LU.
    Returns (L, U, P) with P as a permutation matrix."""
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain finite values")
    L = np.eye(n)
    P = np.eye(n)
    U = A.copy()

    scale = max(1.0, float(np.max(np.abs(U))))

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        if abs(U[k, k]) < 1e-14 * scale:
            raise np.linalg.LinAlgError(f"singular matrix (zero pivot at column {k})")
        factors = U[k + 1:, k] / U[k, k]
        L[k + 1:, k] = factors
        U[k + 1:, k:] -= np.outer(factors, U[k, k:])

    # Check the final diagonal
    if abs(U[n - 1, n - 1]) < 1e-14 * scale:
        raise np.linalg.LinAlgError("singular matrix (zero final diagonal)")
    return L, U, P


def lu_solve(L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b given PA=LU, via forward then back substitution."""
    n = L.shape[0]
    b = P @ np.array(b, dtype=float).reshape(-1)
    # Forward substitution Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    # Back substitution Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_dense(A: np.ndarray, b: np.ndarray, method: str = "lu") -> np.ndarray:
    """Dispatch to a chosen from-scratch solver. method in {'gauss','lu'}."""
    if method == "gauss":
        return gauss_elimination(A, b)
    elif method == "lu":
        L, U, P = lu_decompose(A)
        return lu_solve(L, U, P, b)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'gauss' or 'lu'")


def power_method(A: np.ndarray, n_iter: int = 200, tol: float = 1e-10):
    """Dominant eigenvalue/eigenvector via the power method
    (CSE401 eigenvalue unit) -- used by the Simulation Agent to
    estimate the spectral radius of the discrete diffusion operator
    for an automatic stable-timestep check (see agents/tools.py)."""
    n = A.shape[0]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(n_iter):
        w = A @ v
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-300:
            break
        v = w / w_norm
        lam = v @ (A @ v)
        if abs(lam - lam_old) < tol * max(1.0, abs(lam)):
            lam_old = lam
            break
        lam_old = lam
    return lam_old, v
